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

#include <memory> // std::move

#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionConstIterator.h>

#include "itkGlue.h"

#include <vnl/algo/vnl_symmetric_eigensystem.h>

namespace itk{

template <typename TScalar, unsigned int NDimensions>
TransformAdapter<TScalar, NDimensions>::TransformAdapter() : Superclass(NDimensions),
    m_SigmaC0(1),
    m_SigmaC4(1),
    m_UseWeightImage(false),
    m_UseWeightTensor(false),
    m_MaximumWeight(1),
    m_Subsample(1),
    m_SubsampleNeighborhood(1){
        
    this->m_CPImage         = 0;
    this->m_CPWImage        = 0;
    this->m_WeightImage     = 0;
    this->m_WeightTensor    = 0;

    m_MaximumEigenvalueCovariance = CovarianceMatrixType::InternalMatrixType(SpaceDimensions, SpaceDimensions);
    m_MaximumEigenvalueCovariance.SetIdentity();

    this->m_ReferenceImage  = 0;
    this->m_TargetImage     = 0;

    this->kernelEvaluator.reset();
}

template <typename TScalar, unsigned int NDimensions>
void
TransformAdapter<TScalar, NDimensions>
::SetParameters(const ParametersType & parameters){

    // If SetParameter is called from a Superclass directly after the constructor
    // we set the parameters to an empty set
    if(this->m_CPImage->GetLargestPossibleRegion().GetNumberOfPixels()==0){
        this->m_Parameters = ParametersType(0);
        this->Modified();
        return;
    }

    if(this->m_CPImage->GetLargestPossibleRegion().GetNumberOfPixels()*NDimensions != parameters.Size()){
        itkExceptionMacro (<< "Number of provided parameters (" <<parameters.GetSize() << ") does not correspond to #Dimensions times the number of control points ("<<this->m_CPImage->GetLargestPossibleRegion().GetNumberOfPixels()*NDimensions <<").");
    }

    // Save parameters. Needed for proper operation of TransformUpdateParameters.
    if( &parameters != &(this->m_Parameters) ){
        this->m_Parameters = parameters;
    }

    // store parameters in control point image as well
    unsigned long max = 0;
    ImageRegionIteratorWithIndex<ControlPointImageType> iterator(m_CPImage,m_CPImage->GetLargestPossibleRegion());
    while(!iterator.IsAtEnd()){
        typename ControlPointImageType::ValueType pc;
        typename ControlPointImageType::IndexType current_index = iterator.GetIndex();
        for(unsigned d=0;d<NDimensions;d++){

            /// See GetFlatIndex() for a detailed description of the indexing
            unsigned long param_index = this->GetFlatIndex(current_index, d);
            pc.SetElement(d,this->m_Parameters[param_index]);
            if(param_index>max)max=param_index;
        }
        iterator.Set(pc);
        ++iterator;
    }

    // Modified is always called since we just have a pointer to the
    // parameters and cannot know if the parameters have changed.
    this->Modified();
}

template <typename TScalar, unsigned int NDimensions>
void
TransformAdapter<TScalar, NDimensions>
::UpdateParameters() const{
    // assumption: LargestPossibleRegion of control point image starts at index [0,0,0]
    this->m_Parameters = ParametersType(this->m_CPImage->GetLargestPossibleRegion().GetNumberOfPixels()*ControlPointImageType::ValueType::GetNumberOfComponents());
    ImageRegionConstIterator<ControlPointImageType> iterator(m_CPImage,m_CPImage->GetLargestPossibleRegion());

    unsigned long max = 0;
    while(!iterator.IsAtEnd()){
        typename ControlPointImageType::IndexType current_index = iterator.GetIndex();
        for(unsigned d=0; d<ControlPointImageType::ValueType::GetNumberOfComponents(); d++){

            /// See GetFlatIndex() for a detailed description of the indexing
            unsigned long param_index = this->GetFlatIndex(current_index, d);
            if(param_index>=this->m_Parameters.GetSize()){
                itkExceptionMacro(<<"Parameter indexing exceeds parameter vector size (idx="<<param_index<<", size="<<this->m_Parameters.GetSize()<<")");
            }
            this->m_Parameters[param_index] = iterator.Get().GetElement(d);
            if(param_index>max)max=param_index;
        }
        ++iterator;
    }
}

template <typename TScalar, unsigned int NDimensions>
const typename TransformAdapter<TScalar, NDimensions>::ParametersType &
TransformAdapter<TScalar, NDimensions>
::GetParameters() const{
    this->UpdateParameters();
    return this->m_Parameters;
}

template <typename TScalar, unsigned int NDimensions>
const typename TransformAdapter<TScalar, NDimensions>::FixedParametersType &
TransformAdapter<TScalar, NDimensions>
::GetFixedParameters() const{
    return this->m_FixedParameters;
}

template <typename TScalar, unsigned int NDimensions>
void 
TransformAdapter<TScalar, NDimensions>
::SetParameter(typename ControlPointImageType::PointType p, typename ControlPointImageType::PixelType pv){
    typename ControlPointImageType::IndexType idx;
    this->m_CPImage->TransformPhysicalPointToIndex(p,idx);
    this->SetParameter(idx, pv);
}

template <typename TScalar, unsigned int NDimensions>
void 
TransformAdapter<TScalar, NDimensions>
::SetParameter(typename ControlPointImageType::IndexType idx, typename ControlPointImageType::PixelType pv){
    if(this->m_CPImage->GetLargestPossibleRegion().IsInside(idx)){
        this->m_CPImage->SetPixel(idx,pv);
        this->UpdateParameters();
    }
    else{
        itkExceptionMacro("Index is outside the control points region");
    }
}

template <typename TScalar, unsigned int NDimensions>
void
TransformAdapter<TScalar, NDimensions>
::SetControlPointImage(const typename ControlPointImageType::Pointer image){
    this->m_CPImage = image;
    this->UpdateParameters();
}

template <typename TScalar, unsigned int NDimensions>
typename TransformAdapter<TScalar, NDimensions>::ControlPointImageType::Pointer
TransformAdapter<TScalar, NDimensions>
::GetControlPointImage()const{
    return this->m_CPImage;
}

template <typename TScalar, unsigned int NDimensions>
void
TransformAdapter<TScalar, NDimensions>
::SetControlPointWeightImage(const typename ControlPointWeightImageType::Pointer weight_image){
      if(this->m_CPImage->GetLargestPossibleRegion().GetNumberOfPixels() !=
              weight_image->GetLargestPossibleRegion().GetNumberOfPixels()){
          itkExceptionMacro("Control point weight image must have the same amount of pixels than the control point image.");
      }
      for(unsigned d=0; d<SpaceDimension; d++){
          if(this->m_CPImage->GetLargestPossibleRegion().GetUpperIndex()[d] !=
                  weight_image->GetLargestPossibleRegion().GetUpperIndex()[d]){
              itkExceptionMacro("The largest possible region of the control point weight and the control point image must be equal.");
          }
      }
      this->m_CPWImage = weight_image;
}

template <typename TScalar, unsigned int NDimensions>
const typename TransformAdapter<TScalar, NDimensions>::ControlPointWeightImageType::Pointer
TransformAdapter<TScalar, NDimensions>
::GetControlPointWeightImage()const{
    return this->m_CPWImage;
}


template <typename TScalar, unsigned int NDimensions>
unsigned long
TransformAdapter<TScalar, NDimensions>
::GetFlatIndex(const typename ControlPointImageType::IndexType& current_index, unsigned dimension)const{

    /// Flat indexing of parmeters is done as follows:
    /// suppose we index CPImage by CPImage(i,j,k,d)
    /// where i,j,k are indizes in each space dimension
    /// with dimensionality of N,M,O and d indexes
    /// the vector of the index location with dimensionality
    /// D. CPImage(i,j,k,d) = parameters[i + N(j + M(k + Od))].
    typename ControlPointImageType::IndexType upper_index = m_CPImage->GetLargestPossibleRegion().GetUpperIndex();

    // since the size of each index dimension is needed 1 is added to each upper index
    for(unsigned d=0; d<NDimensions; d++) upper_index.SetElement(d, upper_index.GetElement(d)+1);

    unsigned long param_index = 0;
    if(NDimensions==2){
        // i + N * (j + M * d)
        param_index = current_index.GetElement(0)+
                upper_index.GetElement(0)*(current_index.GetElement(1)+
                                           upper_index.GetElement(1)*dimension);
    }
    else if(NDimensions==3){
        // i + N * (j + M * (k + O * d)
        param_index = current_index.GetElement(0)+
                upper_index.GetElement(0)*(current_index.GetElement(1)+
                                           upper_index.GetElement(1)*(current_index.GetElement(2)+
                                                                      upper_index.GetElement(2)*dimension));
    }
    else{
        itkExceptionMacro ("Only 2 and 3 space dimensions supported");
    }
    return param_index;
}

/**
 * @brief ComputeControlPointImage
 * A contol point image in terms of a regular grid is generated which lies within the reference image.
 * The spacing of the control points is defined as sigma.
 * @param image Reference image to generate the control point image which lies within the reference image
 * @param sigma Spacing of the control points
 * @return Control point image
 */
template <typename TScalar, unsigned int NDimensions>
void
TransformAdapter<TScalar, NDimensions>
::ComputeControlPointImage(
    const typename ImageType::Pointer reference_image, 
    typename ControlPointImageType::SpacingType sigma){

    typename ImageType::PointType lower_extent, upper_extent;
    typename ImageType::RegionType region = reference_image->GetLargestPossibleRegion();
    reference_image->TransformIndexToPhysicalPoint(region.GetIndex(),lower_extent);
    reference_image->TransformIndexToPhysicalPoint(region.GetUpperIndex(),upper_extent);

    typename ControlPointImageType::SpacingType cp_spacing;
    for(unsigned d=0; d<SpaceDimensions; d++) cp_spacing[d] = sigma[d];

    typename ControlPointImageType::SizeType cp_size;
    for(unsigned d=0; d<SpaceDimensions; d++) cp_size[d] = std::ceil(std::abs((upper_extent[d] - lower_extent[d]) / sigma[d]));

    typename ImageType::PointType origin;
    origin = reference_image->GetOrigin();

    typename ControlPointImageType::RegionType cp_region(region.GetIndex(), cp_size);

    typename ControlPointImageType::Pointer cp_image = ControlPointImageType::New();
    cp_image->SetRegions(cp_region);
    cp_image->SetSpacing(cp_spacing);
    cp_image->SetOrigin(origin);
    cp_image->SetDirection(reference_image->GetDirection());
    cp_image->Allocate();

    typename ControlPointImageType::ValueType initial_value;
    for(unsigned d=0; d<SpaceDimensions; d++){
        initial_value.SetElement(d,0);
    }

    cp_image->FillBuffer(initial_value);

    // adjust origin such that cp image is centered with respect to the image
    typename ControlPointImageType::PointType cp_lower_extent, cp_upper_extent;
    cp_image->TransformIndexToPhysicalPoint(cp_region.GetIndex(),cp_lower_extent);
    cp_image->TransformIndexToPhysicalPoint(cp_region.GetUpperIndex(),cp_upper_extent);

    typename ControlPointImageType::PointType cp_origin;
    for(unsigned d=0; d<SpaceDimensions; d++) cp_origin[d] = origin[d] + std::abs(upper_extent[d]-cp_upper_extent[d])/2.0;
    cp_image->SetOrigin(cp_origin);

    this->m_CPImage = cp_image;

    // construct corresponding weight image
    typename ControlPointWeightImageType::Pointer cpw_image = ControlPointWeightImageType::New();
    cpw_image->CopyInformation( cp_image );
    cpw_image->SetRequestedRegion( cp_image->GetRequestedRegion() );
    cpw_image->SetBufferedRegion( cp_image->GetBufferedRegion() );
    cpw_image->Allocate();

    // by default all the weights are one (no influence)
    typename ControlPointWeightImageType::ValueType initial_weight(1);
    cpw_image->FillBuffer(initial_weight);

    this->m_CPWImage = cpw_image;

    this->UpdateParameters();
}

template <typename TScalar, unsigned int NDimensions>
void
TransformAdapter<TScalar, NDimensions>
::SetWeightingImage(const typename TransformAdapter<TScalar, NDimensions>::ImageType::Pointer weight_image){
    this->m_UseWeightImage = true;
    this->m_WeightImage = weight_image;

    // check if weight image is positive
    double maximum = std::numeric_limits<double>::lowest();
    ImageRegionConstIteratorWithIndex<ImageType> iter(this->m_WeightImage, this->m_WeightImage->GetLargestPossibleRegion());
    while(!iter.IsAtEnd()){
        double val = iter.Get();
        if(maximum<val){
            maximum = val;
        }
        if(val <=1e-8){
            itkExceptionMacro ("Weighting image has to be strictly positive everywhere.");
            return;
        }
        ++iter;
    }
    this->m_MaximumWeight = maximum;
}

template <typename TScalar, unsigned int NDimensions>
void
TransformAdapter<TScalar, NDimensions>
::SetWeightingTensor(const typename TransformAdapter<TScalar, NDimensions>::TensorImageType::Pointer weight_tensor, TScalar maximum_tilt){
    this->m_UseWeightTensor = true;
    this->m_WeightTensor = weight_tensor;

    if(maximum_tilt<0){
        // get maximum tilt for kernel support estimation
        double maximum = std::numeric_limits<double>::lowest();
        ImageRegionConstIteratorWithIndex<TensorImageType> iter(this->m_WeightTensor, this->m_WeightTensor->GetLargestPossibleRegion());
        while(!iter.IsAtEnd()){
            TensorImageType::PixelType M = iter.Get();
            // check positive definiteness
            if(vnl_determinant(M.GetVnlMatrix())<=0){
                itkExceptionMacro ("Weighting tensors must be positive definite: \n Determinant: "<< vnl_determinant(M.GetVnlMatrix()) << "\n Matrix: " << M << "\n Inverse: " << M.GetInverse());
                return;
            }
            // largest eigenvalue is relevant for kernel support estimation
            vnl_symmetric_eigensystem<CovarianceMatrixType::ValueType> eig(M.GetVnlMatrix());
            // eigenvalues are in increasing order, thus, we comare the last element corresponding to the largest eigenvalue
            if(std::abs(eig.get_eigenvalue(SpaceDimensions-1)) > maximum){
                maximum = std::abs(eig.get_eigenvalue(SpaceDimensions-1));
                m_MaximumEigenvalueCovariance = M;
            }
            ++iter;
        }
        this->m_MaximumWeight = maximum;
    }
    else{
        this->m_MaximumWeight = maximum_tilt;
    }
}

template <typename TScalar, unsigned int NDimensions>
void 
TransformAdapter<TScalar, NDimensions>
::SetReferenceImage(ImageType * referenceImage){
    if(m_ReferenceImage.GetPointer() != referenceImage){
        m_ReferenceImage = referenceImage;
    }
}

template <typename TScalar, unsigned int NDimensions>
void 
TransformAdapter<TScalar, NDimensions>
::SetTargetImage(ImageType * targetImage){
    if(m_TargetImage.GetPointer() != targetImage){
        m_TargetImage = targetImage;
    }
}

template <typename TScalar, unsigned int NDimensions>
void
TransformAdapter<TScalar, NDimensions>
::Initialize(){

    // check if required images are set
    if(!m_ReferenceImage){
        itkExceptionMacro("Reference image not set."); return;
    }
    if(!m_TargetImage){
        itkExceptionMacro("Target image not set."); return;
    }
    if(!m_CPImage){
        itkExceptionMacro("Control point image has not been computed jet."); return;
    }
    if(!m_CPWImage){
        itkExceptionMacro("Control point weight image has not been computed jet."); return;
    }
    if(!m_WeightImage && m_UseWeightImage){
        itkExceptionMacro("Weight image not provided."); return;
    }
    if(!m_WeightTensor && m_UseWeightTensor){
        itkExceptionMacro("Weight tensor not set or not constructed."); return;
    }

    // check kernel parameters
    if(m_SigmaC0<=0 || m_SigmaC4<=0){
        itkExceptionMacro("Kernel parameters have to be positive."); return;
    }

    /** setup GPU Evaluator */
    typedef Vec<ScalarType, SpaceDimensions> VecF;
    typedef Mat<ScalarType, SpaceDimensions> MatF;

    /** if no weighting is used, build empty weighting image/tensor */
    typename WeightImageType::Pointer weightImageItk  = WeightImageType::New();
    typename TensorImageType::Pointer weightTensorItk = TensorImageType::New();
    if(m_UseWeightImage)  weightImageItk  = m_WeightImage;
    if(m_UseWeightTensor) weightTensorItk = m_WeightTensor;

    MatF maximumEigenvalueCovariance         = m_MaximumEigenvalueCovariance[0];

    auto movingImage  = fromItk<BsplineImage<ScalarType, SpaceDimensions>>(*m_ReferenceImage);
    auto  fixedImage  = fromItk<ImageNearest<ScalarType, SpaceDimensions>>(*m_TargetImage);
    auto     cpImage  = fromItk<ImageNearest<      VecF, SpaceDimensions>>(*m_CPImage);
    auto    cpwImage  = fromItk<ImageNearest<ScalarType, SpaceDimensions>>(*m_CPWImage);
    auto weightImage  = fromItk<ImageNearest<ScalarType, SpaceDimensions>>(*weightImageItk);
    auto weightTensor = fromItk<ImageNearest<      MatF, SpaceDimensions>>(*weightTensorItk);

    const ScalarType *movingImageData = &(*m_ReferenceImage->GetPixelContainer())[0];

    std::unique_ptr<ScalarType[]> movingImageCoeffs;
    movingImage.computeCoeffs(movingImageData, movingImageCoeffs);
    movingImage.assignCoeffs(movingImageCoeffs.get());

    fixedImage  .assignData(&(*m_TargetImage->GetPixelContainer())[0]);
    weightImage .assignData(&(*weightImageItk->GetPixelContainer())[0]);
    weightTensor.assignData(reinterpret_cast<const MatF *>(&(*weightTensorItk->GetPixelContainer())[0]));

    Kernel kernel(
        m_SigmaC0, m_SigmaC4,
        m_UseWeightImage,  m_MaximumWeight,               weightImage,
        m_UseWeightTensor, maximumEigenvalueCovariance,   weightTensor
    );

    kernelEvaluator.reset();
    kernelEvaluator.reset(new GpuKernelEvaluator(
        m_CPWImage->GetLargestPossibleRegion().GetNumberOfPixels()*SpaceDimensions,
        m_Subsample,
        m_SubsampleNeighborhood,
        kernel,
        movingImage,
        fixedImage,
        cpImage,
        cpwImage
    ));
}

template <typename TScalar, unsigned int NDimensions>
GpuKernelEvaluator*
TransformAdapter<TScalar, NDimensions>
::GetGpuKernelEvaluator() const{
    return this->kernelEvaluator.get();
}


template <typename TScalar, unsigned int NDimensions>
typename TransformAdapter<TScalar, NDimensions>::OutputPointType 
TransformAdapter<TScalar, NDimensions>
::TransformPoint(const InputPointType & thisPoint) const{

    /** setup GPU Evaluator */
    typedef Vec<ScalarType, SpaceDimensions> VecF;
    typedef Mat<ScalarType, SpaceDimensions> MatF;

    /** if no weighting is used, build empty weighting image/tensor */
    typename WeightImageType::Pointer weightImageItk  = WeightImageType::New();
    typename TensorImageType::Pointer weightTensorItk = TensorImageType::New();
    if(m_UseWeightImage)  weightImageItk  = m_WeightImage;
    if(m_UseWeightTensor) weightTensorItk = m_WeightTensor;

    MatF maximumEigenvalueCovariance         = m_MaximumEigenvalueCovariance[0];

    auto     cpImage  = fromItk<ImageNearest<      VecF, SpaceDimensions>>(*m_CPImage);
    auto    cpwImage  = fromItk<ImageNearest<ScalarType, SpaceDimensions>>(*m_CPWImage);
    auto weightImage  = fromItk<ImageNearest<ScalarType, SpaceDimensions>>(*weightImageItk);
    auto weightTensor = fromItk<ImageNearest<      MatF, SpaceDimensions>>(*weightTensorItk);

    cpImage     .assignData(reinterpret_cast<const VecF *>(&(*m_CPImage->GetPixelContainer())[0]));
    cpwImage    .assignData(&(*m_CPWImage->GetPixelContainer())[0]);
    weightImage .assignData(&(*weightImageItk->GetPixelContainer())[0]);
    weightTensor.assignData(reinterpret_cast<const MatF *>(&(*weightTensorItk->GetPixelContainer())[0]));

    Kernel kernel(
        m_SigmaC0, m_SigmaC4,
        m_UseWeightImage,  m_MaximumWeight,               weightImage,
        m_UseWeightTensor, maximumEigenvalueCovariance,   weightTensor
    );

    auto imagePoint = fromItk<ScalarType,SpaceDimensions>(thisPoint);

#if SpaceDimensions == 3
    Vec3f displacement(0.0);

    ScalarType support;
    ScalarType sigmaP;
    Mat3f covP;
    if (kernel.useWeightImage()) {
        sigmaP  = kernel.getSigmaAtPoint(imagePoint);
        support = kernel.getRegionSupport(sigmaP);
    } else if (kernel.useWeightTensor()) {
        covP = kernel.getCovarianceAtPoint(imagePoint);
        support = kernel.getRegionSupport(covP);
    } else {
        support = kernel.getRegionSupport();
    }

    Vec3f indexPoint = cpImage.toLocal(imagePoint);
    Vec3i imgLower = std::min(std::max(Vec3i(indexPoint - support/cpImage.scale() + 1.0), Vec3i(0)), cpImage.size() - 1);
    Vec3i imgUpper = std::min(std::max(Vec3i(indexPoint + support/cpImage.scale()      ), Vec3i(0)), cpImage.size() - 1);

    for (int zi = imgLower[2]; zi <= imgUpper[2]; ++zi) {
        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
                Vec3f point_i = cpImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)));
                ScalarType k;
                if (kernel.useWeightImage()){
                    k = kernel.evaluate(imagePoint, point_i, sigmaP);
                }
                else if (kernel.useWeightTensor()){
                    k = kernel.evaluate(imagePoint, point_i, covP);
                }
                else{
                    k = kernel.evaluate(imagePoint, point_i);
                }

                displacement += cpwImage.at(Vec3i(xi, yi, zi))*k*cpImage.at(Vec3i(xi, yi, zi));
            }
        }
    }
#else
    Vec2f displacement(0.0);

    ScalarType support;
    ScalarType sigmaP;
    Mat2f covP;
    if (kernel.useWeightImage()) {
        sigmaP  = kernel.getSigmaAtPoint(imagePoint);
        support = kernel.getRegionSupport(sigmaP);
    } else if (kernel.useWeightTensor()) {
        covP = kernel.getCovarianceAtPoint(imagePoint);
        support = kernel.getRegionSupport(covP);
    } else {
        support = kernel.getRegionSupport();
    }

    Vec2f indexPoint = cpImage.toLocal(imagePoint);
    Vec2i imgLower = std::min(std::max(Vec2i(indexPoint - support/cpImage.scale() + 1.0), Vec2i(0)), cpImage.size() - 1);
    Vec2i imgUpper = std::min(std::max(Vec2i(indexPoint + support/cpImage.scale()      ), Vec2i(0)), cpImage.size() - 1);

    for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
        for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
            Vec2f point_i = cpImage.toGlobal(Vec2f(Vec2i(xi, yi)));
            ScalarType k;
            if (kernel.useWeightImage()){
                k = kernel.evaluate(imagePoint, point_i, sigmaP);
            }
            else if (kernel.useWeightTensor()){
                k = kernel.evaluate(imagePoint, point_i, covP);
            }
            else{
                k = kernel.evaluate(imagePoint, point_i);
            }

            displacement += cpwImage.at(Vec2i(xi, yi))*k*cpImage.at(Vec2i(xi, yi));
        }
    }
#endif

    return OutputPointType(thisPoint + 
            toItkVector<ScalarType,SpaceDimensions>(displacement));
}

}
