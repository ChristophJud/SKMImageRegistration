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

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

// Templated itk ReadImage function.
// Attention: if nifti files are readed, the intend_code in the header
// has to be correct (1007) for displacement fields.
template<typename TImageType>
typename TImageType::Pointer
ReadImage(const std::string& filename)
{
    typedef itk::ImageFileReader<TImageType> ReaderType;
    typename ReaderType::Pointer reader = ReaderType::New();

    reader->SetFileName(filename);
    reader->Update();

    return reader->GetOutput();
}

// Templated itk WriteImage function.
// Attention: if nifti files are readed, the intend_code in the header
// has to be correct (1007) for displacement fields.
template<typename TImageType>
void WriteImage(typename TImageType::Pointer image, const std::string& filename)
{
    typedef itk::ImageFileWriter<TImageType> WriterType;
    typename WriterType::Pointer writer = WriterType::New();

    writer->SetFileName(filename);
    writer->SetInput(image);
    writer->Update();
}

// convenience macro to print messages to cout
#define PRINT(component, message, verbosity) \
    if(config["verbosity"]>=verbosity) std::cout << " (" << component << ") \t" << message << std::endl;

// convenience macro to write result
#define WRITE(variable_name, filename, image_type) \
{ \
    std::stringstream ss; \
    ss << filename; \
    WriteImage<image_type>(variable_name, ss.str()); \
}


template<typename TPointType, unsigned TDimensions >
typename std::vector<TPointType> ReadLandmarks(const std::string& input_filename){
    typename std::vector<TPointType> points;
    unsigned num_points = 0;

    std::ifstream infile;
    infile.open(input_filename);
    std::string line;
    const std::string startswith_string = "POINTS";
    while(std::getline(infile, line)){
        // ignoring comments
        if(line[0] == '#') continue;

        // get number of points
        if(line.substr(0,startswith_string.length()) == startswith_string){
            std::istringstream line_stream(line);
            std::string np;
            line_stream >> np; // POINTS
            line_stream >> num_points; // number of points
            continue;
        }
        
        // parse points data
        double p[3];
        std::istringstream line_stream(line);
        while(line_stream >> p[0] && line_stream >> p[1] && line_stream >> p[2]){
            TPointType itk_point;
            for(unsigned d=0; d<TDimensions; d++){
                itk_point.SetElement(d,p[d]);
            }
            points.emplace_back(itk_point);
        }

    }

    if(points.size() != num_points){
        std::cout << "Warning: number of points read does not correspond to the number of points declared in the header" << std::endl;
    }

    return points;
}