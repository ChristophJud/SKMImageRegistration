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

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkIterationObserver.h"

// itk includes
#include <itkSummationTransform.h>
#include <itkDisplacementFieldTransform.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>

// own includes
#include "IOUtils.h"
#include "CommonTypes.h"
#include "ConfigUtils.h"
#include "itkUtils.h"
#include "AnisotropicTensorUtils.h"

#include "itkTransformAdapter.h"
#include "itkRegularizedImageToImageMetricv4.h"

// other includes
#include "table_printer.h"

// convenience macro
#define CURRENT(option, scale, type) \
    config[option][std::min(scale, static_cast<unsigned int>(config[option].size()-1))].get<type>()

void print_dependencies();
void print_version();

// main registration method
void skmreg(const json& config){
    
    /** read images */
    ImageType::Pointer reference_image;
    ImageType::Pointer target_image;
    ImageType::Pointer weight_image;
    try{
        reference_image = ReadImage<ImageType>(config["reference_filename"]);
        target_image = ReadImage<ImageType>(config["target_filename"]);

        if(config["weight_filename"] != "none"){
            weight_image = ReadImage<ImageType>(config["weight_filename"]);
        }
    }
    catch( itk::ExceptionObject & err){
        std::cout << err << std::endl;
        return;
    }
    PRINT("Application", "total number of image pixels " << reference_image->GetLargestPossibleRegion().GetNumberOfPixels(),1)

    /** read landmarks */
    std::vector<ImageType::PointType> reference_landmarks;
    std::vector<ImageType::PointType> target_landmarks;
    if(config["reference_landmarks"] != "none" &&
       config["target_landmarks"]    != "none"){
            reference_landmarks = ReadLandmarks<ImageType::PointType,SpaceDimensions>( config["reference_landmarks"] );
            target_landmarks    = ReadLandmarks<ImageType::PointType,SpaceDimensions>( config["target_landmarks"]    );
    }

    /** preprocessing of images */
    if(config["intensity_thresholds"].size()==2){
        reference_image = ThresholdImage<ImageType>(reference_image, config["intensity_thresholds"][1], config["intensity_thresholds"][0]);
        target_image    = ThresholdImage<ImageType>(target_image, config["intensity_thresholds"][1], config["intensity_thresholds"][0]);
    }
    reference_image = RescaleImage<ImageType>(reference_image, 0, 1);
    target_image    = RescaleImage<ImageType>(target_image, 0, 1);

    std::string temp_directory = config["temp_directory"];
    WRITE(reference_image, temp_directory + "/reference-preprocessed.vtk", ImageType);
    WRITE(target_image,    temp_directory + "/target-preprocessed.vtk", ImageType);

    /** checking if registration should be resumed at a certain scale level */
    DisplacementFieldType::Pointer past_displacement;
    int restart_at = config["restart_at"];
    if(restart_at > 1 && restart_at <= config["num_scales"]){
        // trying to read preceeding displacement field
        try{
            std::stringstream ss; ss << temp_directory + "/df-full-" << std::max(0,restart_at-2) << ".vtk";
            PRINT("Application", "reading " << ss.str() << " for restarting",2)
            past_displacement = ReadImage<DisplacementFieldType>(ss.str());
            PRINT("Application", "restarting on scale level " << restart_at,1)
        }
        catch(...){ // otherwise registration is started from level 1
            restart_at = 1;
        }
    }
    else{
        restart_at = 1;
    }

    /** declaration of variables, which have to be visible outside the main loop */
    ImageType::Pointer warped_reference;                       
    
    //defining summation transform which holds the last displacement field as transform and the current RKHS transform
    typedef itk::SummationTransform<ScalarType,SpaceDimensions>    SummationTransformType;
    SummationTransformType::Pointer  multiscale_transform  =       SummationTransformType::New();

    /** iteration loop over all scale levels */
    for(unsigned scale=std::max(0,restart_at-1); scale<config["num_scales"]; scale++){
        PRINT("Application", "scale level " << scale+1 << " of " << config["num_scales"],1);

        /** setup new summation transform where the first transformation is a
            displacement field transform containing all displacements of the
            preceeding scale levels  */
        if(scale>0){
            PRINT("Application", "adding past transformation",3)
            typedef itk::DisplacementFieldTransform<ScalarType, SpaceDimensions>  DisplacementFieldTransform;
            DisplacementFieldTransform::Pointer df_transform = DisplacementFieldTransform::New();
            df_transform->SetDisplacementField(past_displacement);

            multiscale_transform  =    SummationTransformType::New();
            multiscale_transform->AddTransform(df_transform);
        }

        /** smooth images */
        double variance = CURRENT("smoothing_variance", scale,double);
        ImageType::Pointer reference_image_smoothed;
        ImageType::Pointer target_image_smoothed;
        if(variance>1e-5){
            reference_image_smoothed = GaussianSmoothing<ImageType>(reference_image, variance);
            target_image_smoothed    = GaussianSmoothing<ImageType>(target_image,    variance);
        }
        else{
            reference_image_smoothed = reference_image;
            target_image_smoothed    = target_image;
        }
        if(config["verbosity"]>=2){
            WRITE(reference_image_smoothed, temp_directory + "/smoothed_reference-" << scale << ".vtk", ImageType);
            WRITE(target_image_smoothed,    temp_directory + "/smoothed_target-"    << scale << ".vtk", ImageType);
        }

        /** setup BSpline interpolator used by the metric */
        typedef itk::BSplineInterpolateImageFunction<ImageType, ScalarType> InterpolatorType;
        InterpolatorType::Pointer interpolator = InterpolatorType::New();
        interpolator->SetInputImage(reference_image_smoothed);
        interpolator->SetSplineOrder(3);

        /** setup regularized image to image metric */
        PRINT("Metric", "setting up metric",2);
        typedef itk::RegularizedImageToImageMetricv4 < ImageType , ImageType, ImageType, ScalarType > MetricType;
        MetricType::Pointer metric = MetricType::New();
        metric->SetFixedImage       (target_image_smoothed);
        metric->SetMovingImage      (reference_image_smoothed);
        metric->SetTransform        (multiscale_transform);

        /** configure regularization terms */
        metric->SetRegularizerL1        (CURRENT("reg_l1",          scale,double));
        metric->SetRegularizerL21       (CURRENT("reg_l21",         scale,double));
        metric->SetRegularizerL2        (CURRENT("reg_l2",          scale,double));
        metric->SetRegularizerRKHS      (CURRENT("reg_rkhs",        scale,double));
        metric->SetRegularizerRD        (CURRENT("reg_rd",          scale,double));
        metric->SetRegularizerRDScaling (CURRENT("reg_rd_scaling",  scale,double));
        metric->SetRegularizerPG        (CURRENT("reg_pg",          scale,double));
        metric->SetRegularizerPGScaling (CURRENT("reg_pg_scaling",  scale,double));

        /** switch loss function mse/ncc */
        if(config["metric"]=="mse"){
            metric->UseNccMetricOff();
            PRINT("Metric", "mse loss is used",2);
        }
        if(config["metric"]=="ncc"){
            metric->UseNccMetricOn();
            PRINT("Metric", "ncc loss is used",2);
        }
        if(config["metric"]=="lcc"){
            metric->UseLccMetricOn();
            PRINT("Metric", "lcc loss is used",2);
        }

        /** set subsampling rate for metric evaluation */
        metric->SetSubSample(CURRENT("sampling_rate",scale,unsigned));
        metric->SetSubSampleNeighborhood(CURRENT("neighborhood_sampling",scale,unsigned));

        metric->DoResamplingOn();

        /** further metric configuration */
        metric->SetOutputFilename(temp_directory+"/output.txt");
        metric->SetVerbosity(0);

        /** set landmarks */
        if(reference_landmarks.size() >  0 && 
           reference_landmarks.size() == target_landmarks.size()){
               metric->SetMovingLandmarks(reference_landmarks);
               metric->SetFixedLandmarks (target_landmarks);
        }

        /** setup TransformAdapter which is compartible with the itk registration framework*/
        PRINT("Transform", "setting up transform",2);
        typedef itk::TransformAdapter<ScalarType, SpaceDimensions> TransformType;
        TransformType::Pointer transform = TransformType::New();

        /** set kernel parameters */
        transform->SetSigmaC0(CURRENT("sigma_C0",scale,float));
        transform->SetSigmaC4(CURRENT("sigma_C4",scale,float));

        // estimate spacing for control point image
        typedef TransformType::ControlPointImageType    ControlPointImageType;
        ControlPointImageType::SpacingType spacing;
        for(unsigned d=0; d<SpaceDimensions; d++){
            if(config["cp_spacing_scaler"]<1){
                spacing[d] = std::min(CURRENT("sigma_C4",scale,float),CURRENT("sigma_C0",scale,float))/1.0;
            }
            else{
                spacing[d] = std::min(CURRENT("sigma_C4",scale,float),CURRENT("sigma_C0",scale,float))/config["cp_spacing_scaler"].get<float>();
            }

        }
        PRINT("Transform", "compute control point image",1);
        transform->ComputeControlPointImage(reference_image_smoothed,spacing);
        PRINT("Transform", "number of control points: " << transform->GetControlPointImage()->GetLargestPossibleRegion().GetNumberOfPixels(),1);
        PRINT("Transform", "number of parameters: " << transform->GetNumberOfParameters(),1);
        WRITE(transform->GetControlPointImage(), temp_directory + "/cp_image-" << scale << ".vtk", ControlPointImageType);

        /** set reference/target images which is needed by the GPU evaluator */
        transform->SetReferenceImage(reference_image_smoothed);
        transform->SetTargetImage(target_image_smoothed);

        /** set the sampling rate for the GPU evaluator */
        transform->SetSubsample(CURRENT("sampling_rate",scale,unsigned));
        transform->SetSubsampleNeighborhood(CURRENT("neighborhood_sampling",scale,unsigned));

        /** provide weight image to transform */
        if(config["weight_filename"]!="none"){
            transform->SetWeightingImage(weight_image);
        }

        /** setting up tensor image
            case 1: no tensor weighting: weight_tensor_filename = "none" and tensor_weight = 0 (all if clauses are evaluate to false)
            case 2: an external tensor weighting is defined: weight_tensor_filename != "none" and tensor_weight = 0
            case 3: tensor weighting is calculated from target: weight_tensor_filename = "none" and tensor_weight > 0
            case 4: tensor weighting is calculated from external guidance image: weight_tensor_filename != "none" and tensor_weight > 0 */
        double current_tensor_weight = CURRENT("tensor_weights",scale,double);
        TensorImageType::Pointer weight_tensor = 0;
        double maximum_tilt = -1;
        if(config["weight_tensor_filename"]!="none" && current_tensor_weight==0){ // case 2
            PRINT("Transform", "load covariance tensor from file",1);
            try{
                weight_tensor = ReadImage<TensorImageType>(config["weight_tensor_filename"]);
            }
            catch( itk::ExceptionObject & err){
                std::cout << err << std::endl;
                return;
            }
        }
        else if(config["weight_tensor_filename"] == "none" && current_tensor_weight > 0){ // case 3
            PRINT("Transform", "calculating covariance tensor based on target image",1);
            weight_tensor = ComputeAnisotropicTensor(target_image_smoothed, CURRENT("sigma_C4",scale,float), current_tensor_weight, 2, maximum_tilt, false, false);
        }
        else if(config["weight_tensor_filename"] != "none" && current_tensor_weight > 0){
            PRINT("Transform", "calculating covariance tensor based on guidance image",1);
            ImageType::Pointer guidance_image;
            try{
                guidance_image = ReadImage<ImageType>(config["weight_tensor_filename"]);
            }
            catch( itk::ExceptionObject & err){
                std::cout << err << std::endl;
                return;
            }
            weight_tensor = ComputeAnisotropicTensor(guidance_image, CURRENT("sigma_C4",scale,float), current_tensor_weight, 2, maximum_tilt, false, false);
        }
        else{
            PRINT("Transform", "no covariance tensor is used",1);
        }
        if(weight_tensor){
            PRINT("Transform", "setting covariance tensor and calculating maximum tilt.",2);
            transform->SetWeightingTensor(weight_tensor, maximum_tilt);
            if(maximum_tilt>0){
                PRINT("Transform", "maximum tilt is " << maximum_tilt,2);
            }
            WRITE(weight_tensor, temp_directory + "/covariance_tensor-" << scale << ".vtk", TensorImageType);
        }
        
        /** initialize transform. Mainly setting up GPU evaluator */
        PRINT("Transform", "initializing GPU evaluator",1);
        transform->Initialize();

        /** synthesize a displacement field with one nonzero coefficient
           such that the shape of a basis function can be analyzed */
        {
            ControlPointImageType::Pointer tmp_cp_image = transform->GetControlPointImage();
            ControlPointImageType::IndexType tmp_idx;
            ControlPointImageType::ValueType tmp_val, zero_val;
            for(unsigned d=0; d<SpaceDimensions; d++){
                tmp_idx[d] = tmp_cp_image->GetLargestPossibleRegion().GetUpperIndex()[d]/2;
                tmp_val[d] = 1;
                zero_val[d] = 0;
            }
            tmp_cp_image->SetPixel(tmp_idx,tmp_val);


            PRINT("Application", "calculate kernel respose",1);
            DisplacementFieldType::Pointer df = GenerateDisplacementFieldGpu<ImageType,DisplacementFieldType>(
                                transform, 
                                reference_image_smoothed);
            WRITE(df, temp_directory + "/df-kernel-" << scale << ".vtk", DisplacementFieldType);
            tmp_cp_image->SetPixel(tmp_idx,zero_val);
        }

        // for testing
        if(false){
                unsigned num_kernel_responces = 20;
                for(unsigned t=0; t<num_kernel_responces; t++){
                    ControlPointImageType::Pointer tmp_cp_image = transform->GetControlPointImage();
                    ControlPointImageType::IndexType tmp_idx;
                    ControlPointImageType::ValueType tmp_val, zero_val;
                    for(unsigned d=0; d<SpaceDimensions; d++){
                        tmp_idx[d] = tmp_cp_image->GetLargestPossibleRegion().GetUpperIndex()[d]/2;
                        tmp_val[d] = 1;
                        zero_val[d] = 0;
                    }

                    tmp_idx[0] = t * tmp_cp_image->GetLargestPossibleRegion().GetUpperIndex()[0]/num_kernel_responces;
                    std::cout << tmp_cp_image->GetLargestPossibleRegion().GetUpperIndex() << std::endl;
                    tmp_idx[1] = 65;

                    tmp_cp_image->SetPixel(tmp_idx,tmp_val);


                    PRINT("Application", "calculate kernel respose",1);
                    DisplacementFieldType::Pointer df = GenerateDisplacementFieldGpu<ImageType,DisplacementFieldType>(
                                        transform, 
                                        reference_image_smoothed);
                    WRITE(df, temp_directory + "/df-kernel-" << t << ".vtk", DisplacementFieldType);
                    tmp_cp_image->SetPixel(tmp_idx,zero_val);
                }
            }

        /** add new transformation to summation transform and set this transform to optimize only */
        multiscale_transform->AddTransform(transform);
        multiscale_transform->SetOnlyMostRecentTransformToOptimizeOn();

        PRINT("Metric", "initialize metric.",1);
        metric->Initialize();

        unsigned numPixelsCount = reference_image_smoothed->GetLargestPossibleRegion().GetNumberOfPixels();
        std::stringstream optimizer_output;

        auto print = [&numPixelsCount, &optimizer_output]
                        (const MetricType* m, const MetricType::TemporaryValuesType v){
            static auto t0 = std::chrono::system_clock::now(); // time measurement
            numPixelsCount = v[0];

            std::cout << optimizer_output.str(); // print out stuff of optimizer observer
            std::cout << "NumPixelsCounted: " << v[0];
            std::cout << "\t value: " << v[1];
            if(m->GetRegularizerL1()   >0) std::cout << "\t l1: "   << m->GetRegularizerL1()   << " * " << v[3] << " = " << m->GetRegularizerL1()   * v[3];
            if(m->GetRegularizerL21()  >0) std::cout << "\t l21: "  << m->GetRegularizerL21()  << " * " << v[4] << " = " << m->GetRegularizerL21()  * v[4];
            if(m->GetRegularizerL2()   >0) std::cout << "\t l2: "   << m->GetRegularizerL2()   << " * " << v[5] << " = " << m->GetRegularizerL2()   * v[5];
            if(m->GetRegularizerRKHS() >0) std::cout << "\t rkhs: " << m->GetRegularizerRKHS() << " * " << v[6] << " = " << m->GetRegularizerRKHS() * v[6];
            if(m->GetRegularizerRD()   >0) std::cout << "\t rd: "   << m->GetRegularizerRD()   << " * " << v[7] << " = " << m->GetRegularizerRD()   * v[7];
            if(m->GetRegularizerPG()   >0) std::cout << "\t pg: "   << m->GetRegularizerPG()   << " * " << v[8] << " = " << m->GetRegularizerPG()   * v[8];

            // not yet supported
            auto tre = m->GetTRE();
            if(tre>=0) std::cout << "\t tre: " << tre;

            // print elapsed seconds since last call
            auto t1 = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = t1-t0;
            std::cout << " \t(" << elapsed_seconds.count() << "s)" << std::flush;
            t0 = t1;
            //std::cout << std::endl;
        };
        metric->SetPrintFunction(std::bind(print,metric.GetPointer(),std::placeholders::_1));

        /** setup optimizer */
        typedef itk::RegularStepGradientDescentOptimizerv4<ScalarType>   OptimizerType;
        OptimizerType::Pointer  optimizer = OptimizerType::New();
        optimizer->SetNumberOfIterations(CURRENT("num_function_evaluations",scale,unsigned));
        optimizer->SetLearningRate(CURRENT("initial_step_size",scale,double));
        optimizer->ReturnBestParametersAndValueOn();
        optimizer->SetMinimumStepLength( optimizer->GetLearningRate()/1000000.0 );
        optimizer->SetGradientMagnitudeTolerance(1e-8);
        optimizer->SetRelaxationFactor( 0.99 );
        optimizer->SetMetric(metric);
        optimizer->SetNumberOfThreads(1);

        /** add observer */
        typedef itk::IterationObserver<OptimizerType>  IterationObserverType;
        IterationObserverType::Pointer observer = IterationObserverType::New();

        observer->SetOptimizer(optimizer);
        observer->PrintValueOff();
        observer->PrintPositionOff();
        
        // if(metric->GetRegularizerL1()    >0 &&
        //    metric->GetRegularizerL21()  <=0 && 
        //    metric->GetRegularizerL2()   <=0 && 
        //    metric->GetRegularizerRKHS() <=0 && 
        //    metric->GetRegularizerRD()   <=0 && 
        //    metric->GetRegularizerPG()   <=0 ){
        //        PRINT("Optimizer", "orthant projection is on",1);
        //        optimizer->SetOrthantProjection(true);
        //    }

        // if(metric->GetRegularizerL1()   <=0 &&
        //    metric->GetRegularizerL21()   >0 && 
        //    metric->GetRegularizerL2()   <=0 && 
        //    metric->GetRegularizerRKHS() <=0 && 
        //    metric->GetRegularizerRD()   <=0 && 
        //    metric->GetRegularizerPG()   <=0 ){
        //        PRINT("Optimizer", "orthant projection is on",1);
        //        optimizer->SetOrthantProjection(true);
        //    }


        // TODO: handle orthant projection

        /** start registration */
        try{
            PRINT("Application","starting optimization",1);
            optimizer->StartOptimization();
        }
        catch( itk::ExceptionObject & err ){
            std::cerr << err << std::endl;
            return;
        }

        PRINT("Optimizer", "Stop condition: " << optimizer->GetStopConditionDescription(),1)

        /** set transform to best position */
        transform->SetParameters(optimizer->GetCurrentPosition());
 
        PRINT("Application", "writing temporal results",1)
        {
        /** synthesize displacement field */
            auto t0 = std::chrono::system_clock::now();
            past_displacement = GenerateDisplacementFieldGpu<ImageType,DisplacementFieldType>(
                                    multiscale_transform, 
                                    reference_image);

            auto t1 = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = t1-t0;
            PRINT("Application", "displacement field generated in " << elapsed_seconds.count() << " seconds",2)
        }
        WRITE(past_displacement, temp_directory + "/df-full-" << scale << ".vtk", DisplacementFieldType);
        WRITE(transform->GetControlPointImage(), temp_directory + "/cp_image-" << scale << ".vtk", ControlPointImageType);

        ImageType::Pointer jacdet = JacobianDeterminant<ControlPointImageType,ImageType>(transform->GetControlPointImage());
        WRITE(jacdet, temp_directory + "/cp_jacdet-" << scale << ".vtk", ImageType);

        /** print TRE */
        auto tre = CalculateTRE<MetricType,DisplacementFieldType>(past_displacement, reference_landmarks, target_landmarks); 
        if(tre>=0){
            PRINT("Application", "TRE is " << tre, 1);
        }

        /** write partial results */
        if(config["verbosity"]>=2){
            DisplacementFieldType::Pointer df = GenerateDisplacementFieldGpu<ImageType,DisplacementFieldType>(
                                transform, 
                                reference_image);
            WRITE(df, temp_directory + "/df-" << scale << ".vtk", DisplacementFieldType);

            warped_reference = WarpImage<ImageType, DisplacementFieldType>(reference_image, past_displacement);
            WRITE(warped_reference, temp_directory + "/warped-" << scale << ".vtk", ImageType);
        }

        /** analyze resulting parameters */
        TransformType::ParametersType parameters = transform->GetParameters();
        unsigned z_counter = 0;
        for(unsigned i=0; i<parameters.GetSize(); i++){
            if(parameters.GetElement(i)==0) z_counter++;
        }
        PRINT("Transform", "number of zero parameters: " << 100.0*z_counter/static_cast<double>(parameters.GetSize()) << "%",1);
    } // loop over scale levels

    PRINT("Application", "writing final results",1)
    /** synthesize final displacement field */
    DisplacementFieldType::Pointer df = GenerateDisplacementFieldGpu<ImageType,DisplacementFieldType>(
                                            multiscale_transform, 
                                            reference_image);
    WRITE(df, temp_directory + "/df.vtk", DisplacementFieldType);

    /** write final registered image */
    typedef itk::ResampleImageFilter<ImageType, ImageType, ScalarType, ScalarType>  ResampleFilterType;
    ResampleFilterType::Pointer resample = ResampleFilterType::New();

    typedef itk::DisplacementFieldTransform<ScalarType, SpaceDimensions>  DisplacementFieldTransform;
    DisplacementFieldTransform::Pointer df_transform = DisplacementFieldTransform::New();
    df_transform->SetDisplacementField(df);

    resample->SetTransform( df_transform );
    resample->SetInput( reference_image);
    resample->SetSize(    target_image->GetLargestPossibleRegion().GetSize() );
    resample->SetOutputOrigin(  target_image->GetOrigin() );
    resample->SetOutputSpacing( target_image->GetSpacing() );
    resample->SetOutputDirection( target_image->GetDirection() );
    resample->SetDefaultPixelValue( 0 );

    WriteImage<ImageType>(resample->GetOutput(), temp_directory + "/warped.vtk");
}

int main (int argc, char * argv[]) {


    // load default config
    json config = load_json_config();

    // parse console arguments
    InputParser input(argc, argv);

    // print help message
    if(input.cmdOptionExists("help") || input.cmdOptionExists("-h") || input.cmdOptionExists("--help")){
        print_version();

        static const char USAGE[] =
        R"(        
    Usage:
        SKMreg
        SKMreg config_file <path-to-config-file>
        SKMreg option <option-value> ...
        SKMreg config_file <path-to-config-file> option <option-value> ...
    )";
        static const char EXAMPLES[] =
    R"(
    If no option is provided, the default options are used. By providing
    a config file, the default options are overwritten by the provided
    options. Additionally, once can overwrite options by providing them
    by the command line.
    Priority order is: command line, config file, default value

    Examples:
        SKMreg temp_directory \"/tmp/experiment/\" sigma_C4 "[32,16,8]" verbosity 3
        SKMreg config_file /tmp/config.json
        SKMreg config_file /tmp/config.json temp_directory \"/tmp/experiment/\"

    Note:   if a string is passed as program option as e.g. a filename
            explicit quotes have to be provided and escaped.
    )";
        std::cout << USAGE << std::endl;
        std::cout << EXAMPLES << std::endl;
        std::cout << "Default options:" << std::endl;
        print_json_config(config);
        return 0;
    }

    // print version message
    if(input.cmdOptionExists("version") || input.cmdOptionExists("--version")){
        print_version();
        print_dependencies();
        return 0;
    }

    try{
        // overwrite default config with config file entries
        const std::string &config_filename = input.getCmdOption("config_file");
        if(!config_filename.empty()){
            json user_config = load_json_config(config_filename);
            for(json::iterator item = config.begin(); item!=config.end(); ++item){
                if (user_config.find(item.key()) != user_config.end()) {
                    config[item.key()] = user_config[item.key()];
                }
            }
        }

        // overwrite json params with console params
        for(json::iterator item = config.begin(); item!=config.end(); ++item){
            if(input.cmdOptionExists(item.key())){
                config[item.key()] = json::parse(input.getCmdOption(item.key()));
            }
        }
    }
    catch(const std::invalid_argument& e){
        std::cerr << "Invalid configuration argument: " << e.what() << std::endl;
        return -1;
    }

    // printout config parameters
    std::cout << "Final config: " << std::endl;
    print_json_config(config);

    // start main application with derived configuration
    auto t0 = std::chrono::system_clock::now();
    skmreg(config);
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-t0;
    std::cout << "Registration done in " << elapsed_seconds.count() << "s" << std::endl;

    return 0;
}

void print_dependencies(){
    std::cout << "Dependencies:" << std::endl; 
    std::vector< std::pair<std::string,std::string> > versions;


    #ifdef ITK_VERSION
    versions.emplace_back(std::make_pair("ITK", std::string(ITK_VERSION)));
    #endif

    #ifdef JSON_VERSION
    versions.emplace_back(std::make_pair("Json", std::string(JSON_VERSION)));
    #endif

    #ifdef BPRINTER_VERSION
    versions.emplace_back(std::make_pair("BPrinter", std::string(BPRINTER_VERSION)));
    #endif

    unsigned max_lib_length = 0;
    unsigned max_ver_length = 0;
    for(const auto& elem : versions){
        if(elem.first.size() > max_lib_length)
            max_lib_length = elem.first.size();
        if(elem.second.size() > max_ver_length)
            max_ver_length = elem.second.size();
    }

    const std::string lib_string("Library");
    const std::string ver_string("Version");

    bprinter::TablePrinter tp(&std::cout);
    tp.set_flush_left();
    tp.AddColumn(lib_string, std::max(static_cast<unsigned>(lib_string.size()),max_lib_length));
    tp.AddColumn(ver_string,    std::max(static_cast<unsigned>(ver_string.size()),max_ver_length));
    tp.PrintHeader();

    for(const auto& elem : versions){
        tp << elem.first << elem.second;
    }
    tp.PrintFooter();
}
void print_version(){
    std::cout << "-------------------------------------------------" << std::endl;
    static const char MESSAGE[] =
    R"(Sparse Kernel Machine for Image Registration )";
    std::cout << MESSAGE;
    #if SpaceDimensions == 3
        std::cout << "(3D)";
    #endif
    #if SpaceDimensions == 2
        std::cout << "(2D)";
    #endif
    std::cout << std::endl;      

    #ifdef SKM_VERSION
        std::cout << "Version " << SKM_VERSION << std::endl;
    #endif

    std::cout << std::endl;
    std::cout << "Main contributors:" << std::endl << std::endl;
    std::cout << " - Christoph Jud (christoph.jud@unibas.ch)" << std::endl;
    std::cout << " - Benedikt Bitterli (benedikt.bitterli@unibas.ch)" << std::endl;
    std::cout << " - Nadia Möri (nadia.moeri@unibas.ch)" << std::endl;
    std::cout << " - Robin Sandkühler (robin.sandkuehler@unibas.ch)" << std::endl;
    std::cout << " - Philippe C. Cattin (philippe.cattin@unibas.ch)" << std::endl;

    std::cout << std::endl;
    std::cout << "Please cite:" << std::endl << std::endl;
    
static const char PUB1[] =
R"(Christoph Jud, Nadia Möri, and Philippe C. Cattin.
Sparse Kernel Machines for Discontinuous Registration and Nonstationary Regularization.
In 7th International Workshop on Biomedical Image Registration, 2016.
)";
static const char PUB2[] =
R"(Christoph Jud, Nadia Möri, Benedikt Bitterli and Philippe C. Cattin.
Bilateral Regularization in Reproducing Kernel Hilbert Spaces for Discontinuity Preserving Image Registration
In 7th International Conference on Machine Learning in Medical Imaging, 2016.
)";
    std::cout << PUB1 << std::endl;
    std::cout << PUB2 << std::endl;

    return;
}
