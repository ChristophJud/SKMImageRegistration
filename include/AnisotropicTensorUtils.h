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

#include <itkGradientImageFilter.h>

#include "CommonTypes.h"
#include "itkUtils.h"

#include "vnl/vnl_cross.h"
#include "vnl/vnl_inverse.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"


#include <itkDiscreteGaussianDerivativeImageFilter.h>
#include <itkSymmetricEigenDecompositionImageFilter.h>

// alignment_direction: x=0, y=1, z=2, none=negative
itk::SymmetricEigenDecompositionImageFilter<TensorImageType,TensorImageType>::OutputImageType::Pointer
ComputeAnisotropicTensor(ImageType::Pointer input_mask, double sigma, double weight, int alignment_direction, double& maximum_tile, 
                                bool anisotropic_filtering=true, bool write_to_tmp=true, std::string temp_directory="/tmp/"){

    // preprocess mask
    if(anisotropic_filtering)
        input_mask = AnisotropicDiffusion<ImageType>(input_mask, 0.0625, 10.0, 2.0, 50);

    // setup alignment vector md
    double md[SpaceDimensions];
    for(unsigned d=0; d<SpaceDimensions; d++){
        md[d] = 0;
    }
    if(alignment_direction >= 0 && alignment_direction <SpaceDimensions){
          md[alignment_direction] = 1;
    }

    // Multiply by gradient weight to scale gradient magnitude
    input_mask = MultiplyConstant<ImageType>(input_mask,weight);


    /** Compute structure tensor */
    // K_rho ( nabla u_sigma outer_product nabla u_sigma
    typedef itk::SymmetricSecondRankTensor< ImageType::PixelType, SpaceDimensions> SymmetricTensorType;
    typedef itk::Image<SymmetricTensorType::EigenVectorsMatrixType, SpaceDimensions> TensorImageType;


    ImageType::Pointer smoothed = GaussianSmoothing<ImageType>(input_mask, sigma);

    typedef itk::GradientImageFilter<ImageType, ImageType::PointValueType, ImageType::PointValueType >  GradientFilterType;
    GradientFilterType::Pointer gradientFilter = GradientFilterType::New();
    gradientFilter->SetInput( smoothed );
    gradientFilter->Update();

    if(write_to_tmp)
        WriteImage<GradientFilterType::OutputImageType>(gradientFilter->GetOutput(), temp_directory + "/gradient_image.vtk");

    TensorImageType::Pointer struct_tensor = TensorImageType::New();
    struct_tensor->CopyInformation( gradientFilter->GetOutput() );
    struct_tensor->SetRequestedRegion( gradientFilter->GetOutput()->GetRequestedRegion() );
    struct_tensor->SetBufferedRegion( gradientFilter->GetOutput()->GetBufferedRegion() );
    struct_tensor->Allocate();

    typedef TensorImageType::PixelType EigenSystemMatrixType;
    typedef vnl_vector<EigenSystemMatrixType::ValueType> EigenVectorType;

    itk::ImageRegionConstIterator<GradientFilterType::OutputImageType> iter_gradient(gradientFilter->GetOutput(), gradientFilter->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionIterator<TensorImageType> iter_tensor(struct_tensor, struct_tensor->GetLargestPossibleRegion());
    while(!iter_gradient.IsAtEnd() && !iter_tensor.IsAtEnd()){

        auto M(outer_product(iter_gradient.Get().GetVnlVector(),iter_gradient.Get().GetVnlVector()));

        EigenSystemMatrixType T;
        for(unsigned i=0; i<SpaceDimensions; i++){
            for(unsigned j=0; j<SpaceDimensions; j++){
                T(i,j) = M(i,j);
            }
        }

        iter_tensor.Set(T);

        ++iter_gradient;
        ++iter_tensor;
    }
    if(write_to_tmp)
        WriteImage<TensorImageType>(struct_tensor, temp_directory + "/struct_tensor.vtk");


    // filter tensor image elemet-wise
    TensorImageType::Pointer smoothed_tensor = TensorImageType::New();
    smoothed_tensor->CopyInformation( gradientFilter->GetOutput() );
    smoothed_tensor->SetRequestedRegion( gradientFilter->GetOutput()->GetRequestedRegion() );
    smoothed_tensor->SetBufferedRegion( gradientFilter->GetOutput()->GetBufferedRegion() );
    smoothed_tensor->Allocate();


    // each matrix index
    #pragma omp parallel for
    for(unsigned i=0; i<SpaceDimensions; i++){
        for(unsigned j=0; j<SpaceDimensions; j++){

            ImageType::Pointer component_image = ImageType::New();
            component_image->CopyInformation( gradientFilter->GetOutput() );
            component_image->SetRequestedRegion( gradientFilter->GetOutput()->GetRequestedRegion() );
            component_image->SetBufferedRegion( gradientFilter->GetOutput()->GetBufferedRegion() );
            component_image->Allocate();

            // fill up component image with structure tensor index i,j
            {
                itk::ImageRegionConstIterator<TensorImageType> iter_tensor(struct_tensor, struct_tensor->GetLargestPossibleRegion());
                itk::ImageRegionIterator<ImageType> iter_component(component_image, component_image->GetLargestPossibleRegion());

                while(!iter_tensor.IsAtEnd() && !iter_component.IsAtEnd()){

                    iter_component.Set(iter_tensor.Get()(i,j));

                    ++iter_tensor;
                    ++iter_component;
                }
            }

            component_image = GaussianSmoothing<ImageType>(component_image, sigma/4.0);


            // fill up structur tensor element i,j
            {
                itk::ImageRegionIterator<TensorImageType> iter_tensor(smoothed_tensor, smoothed_tensor->GetLargestPossibleRegion());
                itk::ImageRegionConstIterator<ImageType> iter_component(component_image, component_image->GetLargestPossibleRegion());

                while(!iter_tensor.IsAtEnd() && !iter_component.IsAtEnd()){

                    auto M = iter_tensor.Get();
                    M(i,j) = iter_component.Get();
                    iter_tensor.Set(M);

                    ++iter_tensor;
                    ++iter_component;
                }
            }
        }
    }

    if(write_to_tmp)
        WriteImage<TensorImageType>(smoothed_tensor, temp_directory + "/smoothed_struct_tensor.vtk");

    typedef itk::SymmetricEigenDecompositionImageFilter<TensorImageType,TensorImageType>  SymmetricEigenDecompositionImageFilterType;
    SymmetricEigenDecompositionImageFilterType::Pointer symmetricEigenAnalysisFilter = SymmetricEigenDecompositionImageFilterType::New();
    symmetricEigenAnalysisFilter->SetInput(smoothed_tensor);
    symmetricEigenAnalysisFilter->SetDimension(SpaceDimensions);
    symmetricEigenAnalysisFilter->OrderEigenValuesBy(SymmetricEigenDecompositionImageFilterType::EigenValueOrderType::OrderByMagnitude);
    symmetricEigenAnalysisFilter->Update();

    smoothed_tensor = symmetricEigenAnalysisFilter->GetOutput();

    if(write_to_tmp)
        WriteImage<SymmetricEigenDecompositionImageFilterType::OutputImageType>(smoothed_tensor, temp_directory + "/main_structure.vtk");

    // lets define the Y direction as the main direction of motion and damp first component
    typedef TensorImageType::PixelType EigenSystemMatrixType;
    typedef vnl_vector<EigenSystemMatrixType::ValueType> EigenVectorType;


    // main direction is weighted proportional to the cross product with the Y direction
    // the other directions are scaled to unit magnitude
    {
        double max = std::numeric_limits<double>::lowest();

        itk::ImageRegionConstIterator<GradientFilterType::OutputImageType> iter_gradient(gradientFilter->GetOutput(), gradientFilter->GetOutput()->GetLargestPossibleRegion());
        itk::ImageRegionIterator<TensorImageType> iter_tensor(smoothed_tensor, smoothed_tensor->GetLargestPossibleRegion());
        while(!iter_tensor.IsAtEnd()){

            EigenSystemMatrixType M = iter_tensor.Get();
            EigenVectorType yv = EigenVectorType(SpaceDimensions,0);
            for(unsigned d=0; d<SpaceDimensions; d++) yv[d]=md[d];

#if SpaceDimensions==2

            ScalarType magnitude1 = M.GetVnlMatrix().get_row(1).two_norm(); // largest eigenvector
            ScalarType magnitude2 = M.GetVnlMatrix().get_row(0).two_norm();

            // assumption: if first component is greater than 1e-8, the second one is not zero as well
            if(magnitude1>1e-8 && magnitude2>1e-8){

                // Get eigenvector with largest eigenvalue, and normalize it to unit magnitude
                EigenVectorType normalized = M.GetVnlMatrix().get_row(1)/magnitude1;

                if(alignment_direction >= 0 && alignment_direction <SpaceDimensions){
                    // measure alignement to yv vector
                    double factor = std::abs(vnl_cross_2d(normalized, yv)) + 1e-10;
                    //factor /= direction_weight;
                    factor = factor*factor / std::sqrt(SpaceDimensions);

                    // weight the first direction with that measure
                    for(unsigned d=0; d<SpaceDimensions; d++){
                        M(1,d) = M(1,d)*factor;
                    }
                }

                // because we want exactly the opposite (get small magnitudes for large vectors)
                double reverse_scale = M.GetVnlMatrix().get_row(1).two_norm();
                double scaler = std::pow(reverse_scale,1.5);

                if(max<reverse_scale)
                    max = reverse_scale;


                for(unsigned d=0; d<SpaceDimensions; d++){
                    //M(1,d) = normalized[d]/(1+reverse_scale);
                    //M(1,d) = normalized[d]/(1+std::log(1+reverse_scale));
                    //M(1,d) = normalized[d]/(std::exp(c2*std::abs(reverse_scale)));
                    M(1,d) = normalized[d]/(1+scaler);
                }

                // normalize other dimensions to 1
                for(unsigned d=0; d<SpaceDimensions; d++){
                    //M(0,d) = M(0,d)/magnitude2;
                    M(0,d) = M(0,d)/magnitude2 * (1+std::pow(reverse_scale,1.5)/(1+std::pow(reverse_scale,1.5)));
                }

                M = M.GetVnlMatrix().flipud().transpose();

           }
            else{ // just orthogonal unit vectors
                M(0,0) = 1;
                M(0,1) = 0;
                M(1,0) = 0;
                M(1,1) = 1;
            }

#elif SpaceDimensions==3

            ScalarType magnitude1 = M.GetVnlMatrix().get_row(2).two_norm(); // largest eigenvector
            ScalarType magnitude2 = M.GetVnlMatrix().get_row(1).two_norm();
            ScalarType magnitude3 = M.GetVnlMatrix().get_row(0).two_norm();

            // assumption: if first component is greater than 1e-8, the second one is not zero as well
            if(magnitude1>1e-8 && magnitude2>1e-8 && magnitude3>1e-8){

                // Get eigenvector with largest eigenvalue, and normalize it to unit magnitude
                EigenVectorType normalized = M.GetVnlMatrix().get_row(2)/magnitude1;

                if(alignment_direction >= 0 && alignment_direction <SpaceDimensions){
                    // measure alignement to yv vector
                    ScalarType factor = std::abs(vnl_cross_3d(normalized, yv).two_norm()) + 1e-10;
                    //factor /= direction_weight;
                    factor = factor*factor / std::sqrt(SpaceDimensions);

                    // weight the first direction with that measure
                    for(unsigned d=0; d<SpaceDimensions; d++){
                        M(2,d) = M(2,d)*factor;
                    }
                }

                // because we want exactly the opposite (get small magnitudes for large vectors)
                ScalarType reverse_scale = M.GetVnlMatrix().get_row(2).two_norm();
                ScalarType scaler = std::pow(reverse_scale,1.5);

                if(max<reverse_scale)
                    max = reverse_scale;

                for(unsigned d=0; d<SpaceDimensions; d++){
                    //M(2,d) = normalized[d]/(1+reverse_scale);
                    M(2,d) = normalized[d]/(1+scaler);
                }


                // normalize other dimensions to 1
                for(unsigned d=0; d<SpaceDimensions; d++){
                    M(1,d) = M(1,d)/magnitude2 * (1+std::pow(reverse_scale,1.5)/(1+std::pow(reverse_scale,1.5)));
                    M(0,d) = M(0,d)/magnitude3 * (1+std::pow(reverse_scale,1.5)/(1+std::pow(reverse_scale,1.5)));
                }

                M = M.GetVnlMatrix().flipud().transpose();

           }
            else{ // just orthogonal unit vectors
                M(0,0) = 1; M(0,1) = 0; M(0,2) = 0;
                M(1,0) = 0; M(1,1) = 1; M(1,2) = 0;
                M(2,0) = 0; M(2,1) = 0; M(2,2) = 1;
            }
#endif

            iter_tensor.Set(M);

            ++iter_gradient;
            ++iter_tensor;
        }
        maximum_tile = max;
    }

    if(write_to_tmp)
        WriteImage<SymmetricEigenDecompositionImageFilterType::OutputImageType>(smoothed_tensor, temp_directory + "/tensor_directions.vtk");


    TensorImageType::Pointer covariance_tensor = TensorImageType::New();
    covariance_tensor->CopyInformation( gradientFilter->GetOutput() );
    covariance_tensor->SetRequestedRegion( gradientFilter->GetOutput()->GetRequestedRegion() );
    covariance_tensor->SetBufferedRegion( gradientFilter->GetOutput()->GetBufferedRegion() );
    covariance_tensor->Allocate();


    {
        itk::ImageRegionConstIterator<TensorImageType> iter_direction(smoothed_tensor, smoothed_tensor->GetLargestPossibleRegion());
        itk::ImageRegionIterator<TensorImageType> iter_covariance(covariance_tensor, covariance_tensor->GetLargestPossibleRegion());
        while(!iter_direction.IsAtEnd() && !iter_covariance.IsAtEnd()){

            EigenSystemMatrixType M = iter_direction.Get();


            // calculate covariance matrix given eigenvectors
            double s = M.GetVnlMatrix().get_column(0).two_norm();

            if(s<1e-8){
                M.Fill(0);
                M.SetIdentity();
            }
            else{

                EigenSystemMatrixType::InternalMatrixType S(SpaceDimensions,SpaceDimensions);
                S.fill(0);
                S.set_identity();

                EigenSystemMatrixType::InternalMatrixType U(SpaceDimensions,SpaceDimensions);
                U.fill(0);

                for(unsigned d=0; d<SpaceDimensions; d++){
                    ScalarType s = M.GetVnlMatrix().get_column(d).two_norm();
                    EigenVectorType n = M.GetVnlMatrix().get_column(d)/s;

                    S(d,d) = std::sqrt(s);
                    U.set_column(d,n);
                }

                M = U*S*S*U.transpose();

            }

            if(vnl_determinant(M.GetVnlMatrix())<=0){
                std::cout << "Matrix is not positive definite" << std::endl;
                std::cout << M << std::endl;
                return covariance_tensor;
            }



            iter_covariance.Set(M);

            ++iter_direction;
            ++iter_covariance;
        }
    }

    return covariance_tensor;
}

// alignment_direction: x=0, y=1, z=2, none=negative
itk::SymmetricEigenDecompositionImageFilter<TensorImageType,TensorImageType>::OutputImageType::Pointer
ComputeAnisotropicTensor(ImageType::Pointer input_mask, double sigma, double weight, int alignment_direction, 
                            bool anisotropic_filtering=true, bool write_to_tmp=true, std::string temp_directory="/tmp/"){
    double maximum_tile = -1;
    return ComputeAnisotropicTensor(input_mask, sigma, weight, alignment_direction, maximum_tile, anisotropic_filtering, write_to_tmp);
}