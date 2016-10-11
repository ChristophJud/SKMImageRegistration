/*
 * Copyright 2016 University of Basel, Medical Image Analysis Center
 *
 * Author: Christoph Jud     (christoph.jud@unibas.ch)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#pragma once

#include <iostream>
#include <fstream>

#include <itkImageRegionConstIteratorWithIndex.h>

#include "itkRegularizedImageToImageMetricv4.h"
#include "itkGlue.h"

namespace itk{

template <class TFixedImage, class TMovingImage, class TVirtualImage, typename TInternalComputationValueType>
RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::RegularizedImageToImageMetricv4() : 
    m_SubSample(1),
    m_SubSampleNeighborhood(1),
    m_UseNccMetric(false),
    m_UseLccMetric(false),
    m_RegularizerL1(0),
    m_RegularizerL21(0),
    m_RegularizerL2(0),
    m_RegularizerRKHS(0),
    m_RegularizerRD(0),
    m_RegularizerPG(0),
    m_RegularizerRDScaling(1.0),
    m_RegularizerPGScaling(1.0),
    m_DoResampling(true),
    m_UseNegativeGradient(true)
{
    m_FixedImage  = ITK_NULLPTR;
    m_MovingImage = ITK_NULLPTR;

    m_OutputFilename = "/tmp/output.txt";

#if SpaceDimensions == 3
    m_SubSample = 4;
    m_SubSampleNeighborhood = 4;
#endif

    // set default print function
    printFunction = [](const RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>::TemporaryValuesType v){};

    // TODO: set number of threads to 1
}

template <class TFixedImage, class TMovingImage, class TVirtualImage, typename TInternalComputationValueType>
RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::~RegularizedImageToImageMetricv4()
{
}

template <class TFixedImage, class TMovingImage, class TVirtualImage, typename TInternalComputationValueType>
void
RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::Initialize()
throw ( ExceptionObject )
{
    Superclass::Initialize();

    if(m_RegularizerL1<0 || m_RegularizerL21<0  ||
       m_RegularizerL2<0 || m_RegularizerRKHS<0 ||
       m_RegularizerRD<0 || m_RegularizerPG<0   ){
        itkExceptionMacro("Regularization weights must be positive."); return;
    } 

    if(m_RegularizerRDScaling<=0 || m_RegularizerPGScaling<=0){
        itkExceptionMacro("Robust regularization scales must be strictly positive."); return;
    }

    if(m_SubSample==0 || m_SubSample>100){
        itkExceptionMacro("Sampling rate inproper."); return;
    }

    if(m_SubSampleNeighborhood==0 || m_SubSampleNeighborhood>100){
        itkExceptionMacro("Neighborhood sampling rate inproper."); return;
    }

    if( !m_FixedImage ){
        itkExceptionMacro("Fixed image not set.");
    }

    if( !m_MovingImage ){
        itkExceptionMacro("Moving image not set.");
    }

    this->SetVirtualDomainFromImage(m_MovingImage);
}

template <class TFixedImage, class TMovingImage, class TVirtualImage, typename TInternalComputationValueType>
typename RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::MeasureType
RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::GetValue() const
{
    MeasureType value = 0;

        /** typedefs for cuda image types */
    typedef Vec<ScalarType, SpaceDimensions> VecF;
    typedef Mat<ScalarType, SpaceDimensions> MatF;

    /** update parameters of transform */
    auto parameters = this->m_MovingTransform->GetParameters();

    /** regularization values */
    double reg_value_l1     = 0;    // l1 regularization value (switch: m_RegularizerL1)
    double reg_value_l21    = 0;    // l2,1 regularization value (switch: m_RegularizerL21)
    double reg_value_l2     = 0;    // l2 regularization value (switch: m_RegularizerL2)
    double reg_value_rkhs   = 0;    // rkhs norm regularization value (switch: m_RegularizerRKHS)
    double reg_value_rd     = 0;    // radiometric differences regularization value (switch: m_RegularizerRD)
    double reg_value_pg     = 0;    // parallelogram regularization value (switch: m_RegularizerPG)

    unsigned num_pixels_counted = 0;

    /** prepare transform pointers */
    const TransformAdapterType *transform;
    const SummationTransformType *sumTransform = 
                dynamic_cast<const SummationTransformType *>((const MovingTransformType *)this->m_MovingTransform);
    if(sumTransform){
        transform = dynamic_cast<const TransformAdapterType *>(sumTransform->GetBackTransform());
    }
    else{
        transform = dynamic_cast<const TransformAdapterType *>((const MovingTransformType *)this->m_MovingTransform);
    }
    if(!transform){
        itkExceptionMacro("Transform type not supported."); return 0;
    }

    /** evaluate loss function value and bilateral regularizers which are calculated on GPU*/
    {      
        /** setup cuda displacement field */
        ImageNearest<VecF, SpaceDimensions> displacementField;
        if(sumTransform && sumTransform->GetNumberOfTransforms() > 1){
            const DisplacementFieldTransformType *dfTransform =
                       dynamic_cast<const DisplacementFieldTransformType *>(sumTransform->GetFrontTransform()); 
            if(dfTransform){
                displacementField = fromItk<ImageNearest<VecF, SpaceDimensions>>(*dfTransform->GetDisplacementField());
                displacementField.assignData(reinterpret_cast<const VecF *>(&(*dfTransform->GetDisplacementField()->GetPixelContainer())[0]));
            }
        }

        /** set regularizers for gpu evaluator */
        if(m_RegularizerRKHS>0){
            transform->GetGpuKernelEvaluator()->SetRegularizerRKHS(m_RegularizerRKHS);
        }
        if(m_RegularizerRD>0){
            transform->GetGpuKernelEvaluator()->SetRegularizerRD(m_RegularizerRD,m_RegularizerRDScaling);
        }
        if(m_RegularizerPG>0){
            transform->GetGpuKernelEvaluator()->SetRegularizerPG(m_RegularizerPG,m_RegularizerPGScaling);
        }

        /** get current control points and weights */
        const auto  cpImage = transform->GetControlPointImage();
        const auto cpwImage = transform->GetControlPointWeightImage();

        /** run evaluation */
        GpuKernelEvaluator::EvaluationResult result;
        if(!m_UseNccMetric && !m_UseLccMetric){
            result = transform->GetGpuKernelEvaluator()->getValue(
                                                 GpuKernelEvaluator::MEASURE::MSE,
                                                 reinterpret_cast<const VecF *>(cpImage->GetBufferPointer()),
                                                 cpwImage->GetBufferPointer(),
                                                 displacementField,
                                                 m_DoResampling);
        }
        else if(m_UseNccMetric && !m_UseLccMetric){
            result = transform->GetGpuKernelEvaluator()->getValue(
                                                 GpuKernelEvaluator::MEASURE::NCC,
                                                 reinterpret_cast<const VecF *>(cpImage->GetBufferPointer()),
                                                 cpwImage->GetBufferPointer(),
                                                 displacementField,
                                                 m_DoResampling);
        }
        else if(!m_UseNccMetric && m_UseLccMetric){
            result = transform->GetGpuKernelEvaluator()->getValue(
                                                 GpuKernelEvaluator::MEASURE::LCC,
                                                 reinterpret_cast<const VecF *>(cpImage->GetBufferPointer()),
                                                 cpwImage->GetBufferPointer(),
                                                 displacementField,
                                                 m_DoResampling);
        }
        else{
            std::cout << "Do not know which metric to execute" << std::endl;
        }


        /** gather GPU results */
        value              = result.measure;
        reg_value_rkhs     = result.reg_value_rkhs;
        reg_value_rd       = result.reg_value_rd;
        reg_value_pg       = result.reg_value_pg;
        num_pixels_counted = result.numPixelsCountedGpu;

    }

    /** calculate L1 regularization values/derivative */
    if(m_RegularizerL1>0){
        // calculate l1 norm of parameter vector
        // (do not normalize l1 with number of parameters)
        for(unsigned long p=0; p<parameters.Size(); p++){
            reg_value_l1 += std::abs(parameters[p]);
        }
    }

    if(m_RegularizerL2>0){
    /// calculate l2 norm of parameter vector
    /// (do not normalize with number of parameters)
        ImageRegionConstIteratorWithIndex<TransformAdapterType::ControlPointImageType> iterator(
                    transform->GetControlPointImage(), transform->GetControlPointImage()->GetLargestPossibleRegion());
        while(!iterator.IsAtEnd()){
            typename ControlPointImageType::ValueType pc;
            typename ControlPointImageType::IndexType current_index = iterator.GetIndex();
            for(unsigned d=0;d<SpaceDimensions;d++){
                /// See GetFlatIndex() for a detailed description of the indexing
                unsigned long param_index = transform->GetFlatIndex(current_index, d);
                pc.SetElement(d,parameters[param_index]);
            }
            double pc_norm = pc.GetSquaredNorm();
            reg_value_l2 += pc_norm;
            ++iterator;
        }
    }

    if(m_RegularizerL21>0){
    /// calculate l2,1 norm of parameter vector
    /// (do not normalize with number of parameters)
        ImageRegionConstIteratorWithIndex<TransformAdapterType::ControlPointImageType> iterator(
                    transform->GetControlPointImage(), transform->GetControlPointImage()->GetLargestPossibleRegion());
        while(!iterator.IsAtEnd()){
            typename ControlPointImageType::ValueType pc;
            typename ControlPointImageType::IndexType current_index = iterator.GetIndex();
            for(unsigned d=0;d<SpaceDimensions;d++){
                unsigned long param_index = transform->GetFlatIndex(current_index, d);
                pc.SetElement(d,parameters[param_index]);
            }
            double pc_norm = pc.GetNorm();
            reg_value_l21 += pc_norm;
            ++iterator;
        }
    }

    /** sum up regularization values */
    double reg_value = 0;
    if(m_RegularizerL1   > 0) reg_value += m_RegularizerL1   * reg_value_l1;
    if(m_RegularizerL21  > 0) reg_value += m_RegularizerL21  * reg_value_l21;
    if(m_RegularizerL2   > 0) reg_value += m_RegularizerL2   * reg_value_l2;
    if(m_RegularizerRKHS > 0) reg_value += m_RegularizerRKHS * reg_value_rkhs;
    if(m_RegularizerRD   > 0) reg_value += m_RegularizerRD   * reg_value_rd;
    if(m_RegularizerPG   > 0) reg_value += m_RegularizerPG   * reg_value_pg;

    /** print out to console and file */
    if(m_Verbosity>=1){
        std::stringstream out_string;
        out_string << "metric: " << value+reg_value << ",\t value: " << value;
        if(m_RegularizerL1   > 0) out_string << ",\t l1: "   << m_RegularizerL1   * reg_value_l1;
        if(m_RegularizerL21  > 0) out_string << ",\t l21: "  << m_RegularizerL21  * reg_value_l21;
        if(m_RegularizerL2   > 0) out_string << ",\t l2: "   << m_RegularizerL2   * reg_value_l2;
        if(m_RegularizerRKHS > 0) out_string << ",\t rkhs: " << m_RegularizerRKHS * reg_value_rkhs;
        if(m_RegularizerRD   > 0) out_string << ",\t rd: "   << m_RegularizerRD   * reg_value_rd;
        if(m_RegularizerPG   > 0) out_string << ",\t pg: "   << m_RegularizerPG   * reg_value_pg;
        if(m_Verbosity>=2) std::cout << out_string.str() << std::endl;

        std::fstream fs;
        fs.open(m_OutputFilename, std::fstream::in | std::fstream::out | std::fstream::app);
        fs << out_string.str() << std::endl;
        fs.close();
    }

    /** prepare data for callbacks */
    TemporaryValuesType tmp_values;
    tmp_values.push_back(num_pixels_counted);
    tmp_values.push_back(value);
    tmp_values.push_back(reg_value);
    tmp_values.push_back(reg_value_l1);
    tmp_values.push_back(reg_value_l21);
    tmp_values.push_back(reg_value_l2);
    tmp_values.push_back(reg_value_rkhs);
    tmp_values.push_back(reg_value_rd);
    tmp_values.push_back(reg_value_pg);
    printFunction(tmp_values);

    /** finally add regularization to loss function value */
    value += reg_value;
    return value;
}

template <class TFixedImage, class TMovingImage, class TVirtualImage, typename TInternalComputationValueType>
void
RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::GetDerivative(DerivativeType & derivative) const
{
    MeasureType value;
    // call the combined version
    this->GetValueAndDerivative(value, derivative);
}

template <class TFixedImage, class TMovingImage, class TVirtualImage, typename TInternalComputationValueType>
void
RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::GetValueAndDerivative(MeasureType & value,
                        DerivativeType & derivative) const
{
    /** typedefs for cuda image types */
    typedef Vec<ScalarType, SpaceDimensions> VecF;
    typedef Mat<ScalarType, SpaceDimensions> MatF;

    /** update parameters of transform */
    auto parameters = this->m_MovingTransform->GetParameters();

    /** create derivative vector if it has not the right size */
    if( derivative.GetSize() != this->m_MovingTransform->GetNumberOfParameters() ){
        derivative = DerivativeType(this->m_MovingTransform->GetNumberOfParameters());
    }

    /** regularization values */
    double reg_value_l1     = 0;    // l1 regularization value (switch: m_RegularizerL1)
    double reg_value_l21    = 0;    // l2,1 regularization value (switch: m_RegularizerL21)
    double reg_value_l2     = 0;    // l2 regularization value (switch: m_RegularizerL2)
    double reg_value_rkhs   = 0;    // rkhs norm regularization value (switch: m_RegularizerRKHS)
    double reg_value_rd     = 0;    // radiometric differences regularization value (switch: m_RegularizerRD)
    double reg_value_pg     = 0;    // parallelogram regularization value (switch: m_RegularizerPG)

    unsigned num_pixels_counted = 0;

    /** prepare transform pointers */
    const TransformAdapterType *transform;
    const SummationTransformType *sumTransform = 
                dynamic_cast<const SummationTransformType *>((const MovingTransformType *)this->m_MovingTransform);
    if(sumTransform){
        transform = dynamic_cast<const TransformAdapterType *>(sumTransform->GetBackTransform());
    }
    else{
        transform = dynamic_cast<const TransformAdapterType *>((const MovingTransformType *)this->m_MovingTransform);
    }
    if(!transform){
        itkExceptionMacro("Transform type not supported."); return;
    }

    /** evaluate loss function value and bilateral regularizers which are calculated on GPU*/
    {      
        /** setup cuda displacement field */
        ImageNearest<VecF, SpaceDimensions> displacementField;
        if(sumTransform && sumTransform->GetNumberOfTransforms() > 1){
            const DisplacementFieldTransformType *dfTransform =
                       dynamic_cast<const DisplacementFieldTransformType *>(sumTransform->GetFrontTransform()); 
            if(dfTransform){
                displacementField = fromItk<ImageNearest<VecF, SpaceDimensions>>(*dfTransform->GetDisplacementField());
                displacementField.assignData(reinterpret_cast<const VecF *>(&(*dfTransform->GetDisplacementField()->GetPixelContainer())[0]));
            }
        }

        /** set regularizers for gpu evaluator */
        if(m_RegularizerRKHS>0){
            transform->GetGpuKernelEvaluator()->SetRegularizerRKHS(m_RegularizerRKHS);
        }
        if(m_RegularizerRD>0){
            transform->GetGpuKernelEvaluator()->SetRegularizerRD(m_RegularizerRD,m_RegularizerRDScaling);
        }
        if(m_RegularizerPG>0){
            transform->GetGpuKernelEvaluator()->SetRegularizerPG(m_RegularizerPG,m_RegularizerPGScaling);
        }

        /** get current control points and weights */
        const auto  cpImage = transform->GetControlPointImage();
        const auto cpwImage = transform->GetControlPointWeightImage();


        /** run evaluation */
        GpuKernelEvaluator::EvaluationResult result;
        if(!m_UseNccMetric && !m_UseLccMetric){
            result = transform->GetGpuKernelEvaluator()->getValueAndDerivative(
                                                 GpuKernelEvaluator::MEASURE::MSE,
                                                 reinterpret_cast<const VecF *>(cpImage->GetBufferPointer()),
                                                 cpwImage->GetBufferPointer(),
                                                 displacementField,
                                                 derivative.data_block(),
                                                 m_DoResampling);
        }
        else if(m_UseNccMetric && !m_UseLccMetric){
            result = transform->GetGpuKernelEvaluator()->getValueAndDerivative(
                                                 GpuKernelEvaluator::MEASURE::NCC,
                                                 reinterpret_cast<const VecF *>(cpImage->GetBufferPointer()),
                                                 cpwImage->GetBufferPointer(),
                                                 displacementField,
                                                 derivative.data_block(),
                                                 m_DoResampling);
        }
        else if(!m_UseNccMetric && m_UseLccMetric){
            result = transform->GetGpuKernelEvaluator()->getValueAndDerivative(
                                                 GpuKernelEvaluator::MEASURE::LCC,
                                                 reinterpret_cast<const VecF *>(cpImage->GetBufferPointer()),
                                                 cpwImage->GetBufferPointer(),
                                                 displacementField,
                                                 derivative.data_block(),
                                                 m_DoResampling);
        }
        else{
            std::cout << "Do not know which metric to execute" << std::endl;
        }


        /** gather GPU results */
        value              = result.measure;
        reg_value_rkhs     = result.reg_value_rkhs;
        reg_value_rd       = result.reg_value_rd;
        reg_value_pg       = result.reg_value_pg;
        num_pixels_counted = result.numPixelsCountedGpu;

    }

    /** calculate L1 regularization values/derivative */
    if(m_RegularizerL1>0){
        // calculate l1 norm of parameter vector
        // (do not normalize l1 with number of parameters)
        for(unsigned long p=0; p<parameters.Size(); p++){
            reg_value_l1 += std::abs(parameters[p]);
        }

        /** signum function for calculating subgradient */
        auto sgn = [](ParametersValueType x){
            if(x<0) return -1;
            else if(x==0) return 0;
            else return 1;
        };
        for( unsigned int parameter = 0; parameter < this->m_MovingTransform->GetNumberOfParameters(); parameter++ ){
            /// calculate subgradient of l1 norm
            ParametersValueType par = parameters[parameter];
            if(std::abs(par) > 0){
                derivative[parameter] += m_RegularizerL1 * sgn(par);
            }
            else if(par==0 && derivative[parameter] < -m_RegularizerL1){
                derivative[parameter] += m_RegularizerL1;
            }
            else if(par==0 && derivative[parameter] > m_RegularizerL1){
                derivative[parameter] -= m_RegularizerL1;
            }
            else if(par==0 && -m_RegularizerL1 <= derivative[parameter] && derivative[parameter] <= m_RegularizerL1){
                derivative[parameter] = 0;
            }
            else{
                itkExceptionMacro("Error in calculating subgradient " << derivative[parameter]);
            }
        }
    }

    if(m_RegularizerL2>0){
    /// calculate l2 norm of parameter vector
    /// (do not normalize with number of parameters)
        ImageRegionConstIteratorWithIndex<TransformAdapterType::ControlPointImageType> iterator(
                    transform->GetControlPointImage(), transform->GetControlPointImage()->GetLargestPossibleRegion());
        while(!iterator.IsAtEnd()){
            typename ControlPointImageType::ValueType pc;
            typename ControlPointImageType::IndexType current_index = iterator.GetIndex();
            for(unsigned d=0;d<SpaceDimensions;d++){
                /// See GetFlatIndex() for a detailed description of the indexing
                unsigned long param_index = transform->GetFlatIndex(current_index, d);
                pc.SetElement(d,parameters[param_index]);
            }
            double pc_norm = pc.GetSquaredNorm();
            reg_value_l2 += pc_norm;

            for(unsigned d=0;d<SpaceDimensions;d++){
                unsigned long param_index = transform->GetFlatIndex(current_index, d);
                derivative[param_index] += m_RegularizerL2 * 2.0*pc.GetElement(d);
            }
            ++iterator;
        }
    }

    if(m_RegularizerL21>0){
    /// calculate l2,1 norm of parameter vector
    /// (do not normalize with number of parameters)
        ImageRegionConstIteratorWithIndex<TransformAdapterType::ControlPointImageType> iterator(
                    transform->GetControlPointImage(), transform->GetControlPointImage()->GetLargestPossibleRegion());
        while(!iterator.IsAtEnd()){
            typename ControlPointImageType::ValueType pc;
            typename ControlPointImageType::IndexType current_index = iterator.GetIndex();
            for(unsigned d=0;d<SpaceDimensions;d++){
                unsigned long param_index = transform->GetFlatIndex(current_index, d);
                pc.SetElement(d,parameters[param_index]);
            }
            double pc_norm = pc.GetNorm();
            reg_value_l21 += pc_norm;

            // subgradient
            for(unsigned d=0;d<SpaceDimensions;d++){
                unsigned long param_index = transform->GetFlatIndex(current_index, d);
                ParametersValueType par = parameters[param_index];
                if(pc_norm>0){
                    derivative[param_index] += m_RegularizerL21 * par / pc_norm;
                }
                else if(pc_norm==0 && derivative[param_index] < -m_RegularizerL21){
                    derivative[param_index] += m_RegularizerL21;
                }
                else if(pc_norm==0 && derivative[param_index] > m_RegularizerL21){
                    derivative[param_index] -= m_RegularizerL21;
                }
                else if(pc_norm==0 && -m_RegularizerL21 <= derivative[param_index] && derivative[param_index] <= m_RegularizerL21){
                    derivative[param_index] = 0;
                }
                else{
                    itkExceptionMacro("Error in calculating subgradient");
                }
            }
            ++iterator;
        }
    }

    if(m_UseNegativeGradient)
        derivative = static_cast<ScalarType>(-1.0) * derivative;

    /** sum up regularization values */
    double reg_value = 0;
    if(m_RegularizerL1   > 0) reg_value += m_RegularizerL1   * reg_value_l1;
    if(m_RegularizerL21  > 0) reg_value += m_RegularizerL21  * reg_value_l21;
    if(m_RegularizerL2   > 0) reg_value += m_RegularizerL2   * reg_value_l2;
    if(m_RegularizerRKHS > 0) reg_value += m_RegularizerRKHS * reg_value_rkhs;
    if(m_RegularizerRD   > 0) reg_value += m_RegularizerRD   * reg_value_rd;
    if(m_RegularizerPG   > 0) reg_value += m_RegularizerPG   * reg_value_pg;

    /** print out to console and file */
    if(m_Verbosity>=1){
        std::stringstream out_string;
        out_string << "metric: " << value+reg_value << ",\t value: " << value;
        if(m_RegularizerL1   > 0) out_string << ",\t l1: "   << m_RegularizerL1   * reg_value_l1;
        if(m_RegularizerL21  > 0) out_string << ",\t l21: "  << m_RegularizerL21  * reg_value_l21;
        if(m_RegularizerL2   > 0) out_string << ",\t l2: "   << m_RegularizerL2   * reg_value_l2;
        if(m_RegularizerRKHS > 0) out_string << ",\t rkhs: " << m_RegularizerRKHS * reg_value_rkhs;
        if(m_RegularizerRD   > 0) out_string << ",\t rd: "   << m_RegularizerRD   * reg_value_rd;
        if(m_RegularizerPG   > 0) out_string << ",\t pg: "   << m_RegularizerPG   * reg_value_pg;
        if(m_Verbosity>=2) std::cout << out_string.str() << std::endl;

        std::fstream fs;
        fs.open(m_OutputFilename, std::fstream::in | std::fstream::out | std::fstream::app);
        fs << out_string.str() << std::endl;
        fs.close();
    }

    /** prepare data for callbacks */
    TemporaryValuesType tmp_values;
    tmp_values.push_back(num_pixels_counted);
    tmp_values.push_back(value);
    tmp_values.push_back(reg_value);
    tmp_values.push_back(reg_value_l1);
    tmp_values.push_back(reg_value_l21);
    tmp_values.push_back(reg_value_l2);
    tmp_values.push_back(reg_value_rkhs);
    tmp_values.push_back(reg_value_rd);
    tmp_values.push_back(reg_value_pg);
    printFunction(tmp_values);

    /** finally add regularization to loss function value */
    value += reg_value;
}

template <class TFixedImage, class TMovingImage, class TVirtualImage, typename TInternalComputationValueType>
bool 
RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::SupportsArbitraryVirtualDomainSamples( void ) const{
    return false;
}

template <class TFixedImage, class TMovingImage, class TVirtualImage, typename TInternalComputationValueType>
template<typename Printer>
void
RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::SetPrintFunction(Printer&& p){
    printFunction = p;
}

template <class TFixedImage, class TMovingImage, class TVirtualImage, typename TInternalComputationValueType>
inline typename RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>::MeasureType
RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::GetTRE() const{
    if(m_FixedLandmarks.size()==0 || m_MovingLandmarks.size()==0 ||
            m_FixedLandmarks.size() != m_MovingLandmarks.size()){
        return -1;
    }

    PointListType points;
    // transform target points
    for(unsigned i=0; i<m_FixedLandmarks.size(); i++){
        points.push_back(this->m_MovingTransform->TransformPoint(m_FixedLandmarks[i]));
    }

    MeasureType tre = 0;
    for(unsigned i=0; i<m_FixedLandmarks.size(); i++){
        tre += (points[i]-m_MovingLandmarks[i]).GetNorm();
    }
    return tre/m_FixedLandmarks.size();
}

template <class TFixedImage, class TMovingImage, class TVirtualImage, typename TInternalComputationValueType>
void
RegularizedImageToImageMetricv4<TFixedImage,TMovingImage,TVirtualImage,TInternalComputationValueType>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

} // namespace itk