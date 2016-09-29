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

#include "CommonConfig.h"
#include <itkImage.h>
#include <itkSymmetricSecondRankTensor.h>

typedef itk::Image<ScalarType, SpaceDimensions>                                     ImageType;
typedef itk::Image< itk::Vector<ScalarType, SpaceDimensions>, SpaceDimensions >     ControlPointImageType;
typedef itk::Image<ScalarType, SpaceDimensions>                                     ControlPointWeightImageType;
typedef itk::Image< itk::Vector<ScalarType, SpaceDimensions>, SpaceDimensions >     DisplacementFieldType;

typedef itk::SymmetricSecondRankTensor< ImageType::PixelType, SpaceDimensions>      SymmetricTensorType;
typedef itk::Image<SymmetricTensorType::EigenVectorsMatrixType, SpaceDimensions>    TensorImageType;
