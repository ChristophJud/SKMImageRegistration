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
 
#include <chrono>
#include <cmath>
#include <random>

#include "CommonConfig.h"
#include "IOUtils.h"
#include "CommonTypes.h"
#include "itkUtils.h"

#include "itkBSplineInterpolateImageFunction.h"
#include "itkImageRegionIterator.h"

#include "itkTransformAdapter.h"
#include "itkRegularizedImageToImageMetric.h"


int main (int argc, char * argv[]) {  

    ImageType::Pointer fimage, mimage;
    try{
        #if SpaceDimensions == 3
        fimage = ReadImage<ImageType>("../../data/10-P_cropped.mhd");
        mimage = ReadImage<ImageType>("../../data/70-P_cropped.mhd");
        mimage = fimage;
        #else
        fimage = ReadImage<ImageType>("../../data/l0.IMA");
        mimage = ReadImage<ImageType>("../../data/l4.IMA");
        mimage = fimage;
        #endif
    }
    catch( itk::ExceptionObject & err){
        std::cout << err << std::endl;
        return -1;
    }

    std::cout << fimage->GetSpacing() << std::endl;

    WriteImage<ImageType>(fimage, "/tmp/image.vtk");

    typedef itk::TransformAdapter<ScalarType, SpaceDimensions> TransformType;
    TransformType::Pointer transform = TransformType::New();

    /** set kernel parameters */
    float sigmac0 = 128;
    float sigmac4 = 64;
    transform->SetSigmaC0(sigmac0);
    transform->SetSigmaC4(sigmac4);

    // estimate spacing for control point image
    typedef TransformType::ControlPointImageType    ControlPointImageType;
    ControlPointImageType::SpacingType spacing;
    for(unsigned d=0; d<SpaceDimensions; d++){
            spacing[d] = std::min(sigmac4,sigmac0)/3.0;
    }
    transform->ComputeControlPointImage(fimage,spacing);

    WriteImage<ControlPointImageType>(transform->GetControlPointImage(), "/tmp/cp_image.vtk");

    std::cout << "CP LRegion: " << transform->GetControlPointImage()->GetLargestPossibleRegion() << std::endl;
    std::cout << "Number of control points: " << transform->GetControlPointImage()->GetLargestPossibleRegion().GetNumberOfPixels() << std::endl;

    /** set reference/target images which is needed by the GPU evaluator */
    transform->SetReferenceImage(mimage);
    transform->SetTargetImage(fimage);

    /** set the sampling rate for the GPU evaluator */
    #if SpaceDimensions == 3
    unsigned subsample = 10;
    unsigned subsampleneighborhood = 18; // 20 -> 20,
    #else
    unsigned subsample = 2;
    unsigned subsampleneighborhood = 6; // 20 -> 20,
    #endif  
    transform->SetSubsample(subsample);
    transform->SetSubsampleNeighborhood(subsampleneighborhood);

    transform->Initialize();

    /** setup BSpline interpolator used by the metric */
    typedef itk::BSplineInterpolateImageFunction<ImageType, double> InterpolatorType;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    interpolator->SetInputImage(mimage);
    interpolator->SetSplineOrder(3);

    typedef itk::RegularizedImageToImageMetric < ImageType , ImageType > MetricType;
    MetricType::Pointer metric = MetricType::New();
    metric->SetFixedImage       (fimage);
    metric->SetMovingImage      (mimage);
    metric->SetFixedImageRegion (fimage->GetLargestPossibleRegion());
    metric->SetInterpolator     (interpolator);
    metric->SetTransform        (transform);

    metric->UseNccMetricOn();
    metric->UseLccMetricOff();

    bool mse_is_used= false;
    if(!metric->GetUseNccMetric() && !metric->GetUseLccMetric())
        mse_is_used = true;

    fimage = RescaleImage<ImageType>(fimage, 0, 1);
    mimage = RescaleImage<ImageType>(mimage, 0, 1);

    /** set subsampling rate for metric evaluation */
    metric->SetSubSample(subsample);
    metric->SetSubSampleNeighborhood(subsampleneighborhood);

    metric->DoResamplingOff(); // for finite difference test, resampling before each evaluation has to be turned off

    /** set regularizers **/
    //metric->SetRegularizerRKHS(0.01);
    //metric->SetRegularizerRD(0.01);
    //metric->SetRegularizerPG(0.01);


    metric->SetVerbosity(0);
    metric->Initialize();

    double delta = 0.0001;

    // generate random generator for indices
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<>  dis_int(    0, transform->GetControlPointImage()->GetLargestPossibleRegion().GetNumberOfPixels()*SpaceDimensions);
    
    #if SpaceDimensions == 3
    std::uniform_real_distribution<> dis_real(-1.0, 1.0);
    #else
    std::uniform_real_distribution<> dis_real(-10.0, 10.0);
    #endif

    MetricType::ParametersType r_params = transform->GetParameters();
    
    std::vector<double> errors;
    MetricType::DerivativeType deriv_z; // analytical derivative
    for(unsigned i = 0; i<100; i++){
        unsigned idx = dis_int(gen);

        // first run with zero parameters
        if(i==0){
            r_params.fill(0.0);
        }
        // all others with a specific random initialization
        if(i==1){
            for(unsigned i = 0; i<r_params.size(); i++){
                r_params[i] = dis_real(gen);
            }
            ControlPointImageType::PointType p;
            auto cp_idx = transform->GetControlPointImage()->GetLargestPossibleRegion().GetUpperIndex();
            p[0] = 212;
            p[1] = 276;
            p[2] = 38;
            transform->GetControlPointImage()->TransformPhysicalPointToIndex(p,cp_idx);
            idx = transform->GetFlatIndex(cp_idx,0);

            if(false){
                std::cout << "Writing displacement field..." << std::endl;
                transform->SetParameters(r_params);
                DisplacementFieldType::Pointer tmp_df = GenerateDisplacementField<TransformType,ImageType,DisplacementFieldType>(transform, fimage);
                WriteImage<DisplacementFieldType>(tmp_df, "/tmp/random_df.vtk");
            }
        }
        
        
        MetricType::ParametersType pz = r_params;

        MetricType::MeasureType val_z;
        if(i==0 || i==1){
            auto t0 = std::chrono::system_clock::now();
            metric->GetValueAndDerivative(pz, val_z, deriv_z);
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-t0;
            std::cout << "GetValueAndDerivative done in " << elapsed_seconds.count() << "s" << std::endl;
        }

        // if(i==0){
        //     for(unsigned dz=0; dz<deriv_z.size(); dz++){
        //         if(std::abs(deriv_z[dz])>1e-12){
        //             std::cout << "Analytical derivative of the image with zero transform should be zero." << std::endl;
        //             std::cout << "Index: " << dz << ", GetValueAndDerivative: " << deriv_z[dz] << std::endl;
        //             return -1;
        //         }
        //     }
        // }

        MetricType::MeasureType val_test = metric->GetValue(pz);

        if(std::abs(val_z-val_test)>0){
            std::cout << "Metric value of GetValue and GetValueAndDerivative are different." << std::endl;
            std::cout << " - Value (GetValueAndDerivative): " << val_z << std::endl;
            std::cout << " - Value (GetValue): " << val_test << std::endl;
            return -1;
        }
        
        pz[idx] = pz[idx]+delta;
        MetricType::MeasureType val_h1;
        val_h1 = metric->GetValue(pz);


        double df1 = (val_h1-val_test)/delta;
        double error1 = std::abs(df1-deriv_z[idx]);
        std::cout << "Finite diff, Derivative, Error:\t" << df1 << ",\t" << deriv_z[idx] << ",\t" << error1 << std::endl;

        // if((!mse_is_used && i==0 && std::abs(df1)>1e-8) ||
        //    ( mse_is_used && i==0 && std::abs(df1)>1e-4)){
        //     std::cout << "Finite difference of the image with zero transform should be zero." << std::endl;
        //     return -1;
        // }


        errors.push_back(error1);
    }

    double abs_sum = 0;
    for(auto err : errors){
        abs_sum += std::abs(err);
    }

    std::cout << "Average error: " << abs_sum/errors.size() << std::endl;

    return 0;
}