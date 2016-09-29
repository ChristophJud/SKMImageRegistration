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

#include <limits>

// ITK includes
#include <itkDisplacementFieldTransform.h>
#include <itkTransformToDisplacementFieldFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkStatisticsImageFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkCurvatureAnisotropicDiffusionImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkWarpImageFilter.h>
#include <itkDisplacementFieldJacobianDeterminantFilter.h>


// own includes
#include "CommonTypes.h"
#include "GpuEvaluator.h"
#include "itkSummationTransform.h"
#include "itkTransformAdapter.h"


template<typename TImageType>
typename TImageType::Pointer
ThresholdImage(typename TImageType::Pointer image, double thresh_max, double thresh_min)
{
    // estimate maximum below threshold
    double max = std::numeric_limits<double>::lowest();
    double min = std::numeric_limits<double>::max();
    typename itk::ImageRegionConstIterator<TImageType> iterator(image,image->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd()){
        double value = static_cast<double>(iterator.Get());
        if(max<value && value<thresh_max){
            max=value;
        }
        if(min>value && value>thresh_min){
            min=value;
        }
        ++iterator;
    }

    // get minimum value
    typedef typename itk::StatisticsImageFilter<TImageType> StatisticsImageFilterType;
    typename StatisticsImageFilterType::Pointer statistics = StatisticsImageFilterType::New();
    statistics->SetInput(image);
    statistics->Update();

    typename TImageType::Pointer result;
    {
        // threshold image for max threshold
        typedef typename itk::ThresholdImageFilter<TImageType> ThresholdImageFilterType;
        typename ThresholdImageFilterType::Pointer thresher = ThresholdImageFilterType::New();
        thresher->SetInput(image);
        thresher->ThresholdAbove(thresh_max);
        thresher->ThresholdBelow(statistics->GetMinimum());
        thresher->ThresholdOutside(statistics->GetMinimum(), max);
        thresher->SetOutsideValue(max);
        thresher->Update();
        result = thresher->GetOutput();
    }

    {
        // threshold image for min threshold
        typedef typename itk::ThresholdImageFilter<TImageType> ThresholdImageFilterType;
        typename ThresholdImageFilterType::Pointer thresher = ThresholdImageFilterType::New();
        thresher->SetInput(result);
        thresher->ThresholdAbove(statistics->GetMaximum());
        thresher->ThresholdBelow(thresh_min);
        thresher->ThresholdOutside(min, statistics->GetMaximum());
        thresher->SetOutsideValue(min);
        thresher->Update();
        result = thresher->GetOutput();
    }

    return result;
}

template<typename TImageType>
typename TImageType::Pointer
RescaleImage(typename TImageType::Pointer image, typename TImageType::ValueType min, typename TImageType::ValueType max){
    typedef typename itk::RescaleIntensityImageFilter<TImageType,TImageType> RescaleIntensityImageFilter;
    typename RescaleIntensityImageFilter::Pointer filter = RescaleIntensityImageFilter::New();
    filter->SetInput(image);
    filter->SetOutputMinimum(min);
    filter->SetOutputMaximum(max);
    filter->Update();
    return filter->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer
AnisotropicDiffusion(typename TImageType::Pointer image, double timestep=0.0625, double conductance=3.0,
                     double conductance_scaling=1, unsigned num_iterations=5){
    typedef typename itk::CurvatureAnisotropicDiffusionImageFilter<TImageType,TImageType> AnisotropicDiffusionFilterType;
    typename AnisotropicDiffusionFilterType::Pointer filter = AnisotropicDiffusionFilterType::New();
    filter->SetTimeStep(timestep);
    filter->SetConductanceParameter(conductance);
    filter->SetConductanceScalingParameter(conductance_scaling);
    filter->SetNumberOfIterations(num_iterations);
    filter->SetInput(image);
    filter->Update();
    return filter->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer
MultiplyConstant(typename TImageType::Pointer image, typename TImageType::ValueType constant){

    typedef typename itk::MultiplyImageFilter<TImageType, TImageType, TImageType> MultiplyImageFilterType;
    typename MultiplyImageFilterType::Pointer multiplier = MultiplyImageFilterType::New();
    multiplier->SetInput(image);
    multiplier->SetConstant(constant);
    multiplier->Update();

    return multiplier->GetOutput();
}

template<typename TImageType>
typename TImageType::Pointer
GaussianSmoothing(typename TImageType::Pointer image, double variance=1, bool use_image_spacing=true){
    typedef typename itk::DiscreteGaussianImageFilter<TImageType,TImageType> DiscreteGaussianImageFilterType;
    typename DiscreteGaussianImageFilterType::Pointer filter = DiscreteGaussianImageFilterType::New();
    filter->SetUseImageSpacing(use_image_spacing);
    filter->SetInput(image);
    filter->SetVariance(variance);
    filter->SetMaximumKernelWidth(64);
    filter->Update();
    return filter->GetOutput();
}

template<typename TDisplacementFieldType, typename TImageType>
typename TImageType::Pointer
JacobianDeterminant(typename TDisplacementFieldType::Pointer df){
    typedef typename itk::DisplacementFieldJacobianDeterminantFilter<TDisplacementFieldType, ScalarType, TImageType> DisplacementFieldJacobianDeterminantFilterType;
    typename DisplacementFieldJacobianDeterminantFilterType::Pointer filter = DisplacementFieldJacobianDeterminantFilterType::New();
    filter->SetInput(df);
    filter->Update();
    return filter->GetOutput();
}

template<typename TImageType, typename TDisplacementFieldType>
typename TImageType::Pointer
WarpImage(typename TImageType::Pointer image, const typename TDisplacementFieldType::Pointer df, unsigned order=3){
    typedef typename itk::BSplineInterpolateImageFunction<TImageType, double> InterpolatorType;
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    interpolator->SetSplineOrder(order);

    typedef typename itk::WarpImageFilter<TImageType, TImageType, TDisplacementFieldType> WarpImageFilterType;
    typename WarpImageFilterType::Pointer warper = WarpImageFilterType::New();
    warper->SetInterpolator(interpolator);
    warper->SetDisplacementField(df);
    warper->SetOutputParametersFromImage(df);
    warper->SetInput(image);
    warper->Update();

    return warper->GetOutput();
}

template<typename TImageType, typename TDisplacementFieldType>
typename TDisplacementFieldType::Pointer GenerateDisplacementFieldGpu(
    itk::TransformAdapter<ScalarType, SpaceDimensions>::Pointer transform, 
    typename TImageType::Pointer reference_image)
{
    GpuKernelEvaluator *evaluator = transform->GetGpuKernelEvaluator();

    const auto  cpImage = transform->GetControlPointImage();
    const auto cpwImage = transform->GetControlPointWeightImage();

    typedef Vec<ScalarType, SpaceDimensions> VecF;

    typename TDisplacementFieldType::Pointer df = TDisplacementFieldType::New();
    df->SetRegions(reference_image->GetLargestPossibleRegion());
    df->SetLargestPossibleRegion(reference_image->GetLargestPossibleRegion());
    df->SetSpacing(reference_image->GetSpacing());
    df->SetOrigin(reference_image->GetOrigin());
    df->SetDirection(reference_image->GetDirection());
    df->Allocate();

    evaluator->evaluateDisplacementField(
        reinterpret_cast<const VecF *>(cpImage->GetBufferPointer()),
        cpwImage->GetBufferPointer(),
        ImageNearest<VecF, SpaceDimensions>(),
        reinterpret_cast<VecF *>(&(*df->GetPixelContainer())[0])
    );

    return df;
}

template<typename TImageType, typename TDisplacementFieldType>
typename TDisplacementFieldType::Pointer GenerateDisplacementFieldGpu(
                itk::SummationTransform<ScalarType, SpaceDimensions>::Pointer sumTransform,
                typename TImageType::Pointer reference_image)
{
    const itk::TransformAdapter<ScalarType, SpaceDimensions> *transform= dynamic_cast<const itk::TransformAdapter<ScalarType, SpaceDimensions> *>(sumTransform->GetBackTransform());
    if (!transform)
        itkGenericExceptionMacro(<< "Summation transform passed to GenerateDisplacementFieldGpu does not contain an RKHS transform");

    GpuKernelEvaluator *evaluator = transform->GetGpuKernelEvaluator();

    const auto  cpImage = transform->GetControlPointImage();
    const auto cpwImage = transform->GetControlPointWeightImage();

    typedef Vec<ScalarType, SpaceDimensions> VecF;
    ImageNearest<VecF, SpaceDimensions> displacementField;
    if (sumTransform->GetNumberOfTransforms() > 1) {
        const itk::DisplacementFieldTransform<ScalarType, SpaceDimensions> *dfTransform =
                dynamic_cast<const itk::DisplacementFieldTransform<ScalarType, SpaceDimensions> *>(sumTransform->GetFrontTransform());
        if (!transform)
            itkGenericExceptionMacro(<< "Summation transform passed to GenerateDisplacementFieldGpu does not contain a displacement field transform");

        displacementField = fromItk<ImageNearest<VecF, SpaceDimensions>>(*dfTransform->GetDisplacementField());
        displacementField.assignData(reinterpret_cast<const VecF *>(&(*dfTransform->GetDisplacementField()->GetPixelContainer())[0]));
    }

    typename TDisplacementFieldType::Pointer df = TDisplacementFieldType::New();
    df->SetRegions(reference_image->GetLargestPossibleRegion());
    df->SetLargestPossibleRegion(reference_image->GetLargestPossibleRegion());
    df->SetSpacing(reference_image->GetSpacing());
    df->SetOrigin(reference_image->GetOrigin());
    df->SetDirection(reference_image->GetDirection());
    df->Allocate();

    evaluator->evaluateDisplacementField(
        reinterpret_cast<const VecF *>(cpImage->GetBufferPointer()),
        cpwImage->GetBufferPointer(),
        displacementField,
        reinterpret_cast<VecF *>(&(*df->GetPixelContainer())[0])
    );

    return df;
}

template<typename TTransformType, typename TImageType, typename TDisplacementFieldType>
typename TDisplacementFieldType::Pointer GenerateDisplacementField(typename TTransformType::Pointer transform,
                                                       typename TImageType::Pointer reference_image){
    typedef typename itk::TransformToDisplacementFieldFilter<TDisplacementFieldType> DisplacementFieldGeneratorType;
    typename DisplacementFieldGeneratorType::Pointer dispfieldGenerator = DisplacementFieldGeneratorType::New();
    dispfieldGenerator->UseReferenceImageOn();
    dispfieldGenerator->SetReferenceImage(reference_image);
    dispfieldGenerator->SetTransform(transform);
    dispfieldGenerator->Update();
    return dispfieldGenerator->GetOutput();
}

template<typename TMetricType, typename TDisplacementFieldType>
typename TMetricType::MeasureType
CalculateTRE(typename TDisplacementFieldType::Pointer df,
             typename TMetricType::PointListType fixed_landmarks, 
             typename TMetricType::PointListType moving_landmarks){
    
    if(fixed_landmarks.size()>0 && fixed_landmarks.size()==moving_landmarks.size()){
        typedef itk::DisplacementFieldTransform<ScalarType, SpaceDimensions>  DisplacementFieldTransform;
        DisplacementFieldTransform::Pointer df_transform = DisplacementFieldTransform::New();
        df_transform->SetDisplacementField(df);

        typename TMetricType::PointListType points;
        // transform target points
        for(unsigned i=0; i<fixed_landmarks.size(); i++){
            points.push_back(df_transform->TransformPoint(moving_landmarks[i]));
        }

        typename TMetricType::MeasureType tre = 0;
        for(unsigned i=0; i<fixed_landmarks.size(); i++){
            tre += (points[i]-fixed_landmarks[i]).GetNorm();
        }
        return tre/fixed_landmarks.size();
    }
    else{ 
        return -1;
    }
}
