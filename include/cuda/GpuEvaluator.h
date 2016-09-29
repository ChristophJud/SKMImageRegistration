/*
 * Copyright 2016 University of Basel, Medical Image Analysis Center
 *
 * Author: Benedikt Bitterli (benedikt.bitterli@unibas.ch)
 *         Christoph Jud     (christoph.jud@unibas.ch)
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

#include <iomanip>
#include <chrono>
#include <vector>
#include <cstdio>

#include "CommonConfig.h"
#include "BsplineImage.h"
#include "ImageNearest.h"
#include "UniformRng.h"
#include "ImageBase.h"
#include "Kernel.h"
#include "Vec.h"

#include "CudaUtils.h"

typedef Vec<ScalarType, SpaceDimensions> VecF;
typedef Mat<ScalarType, SpaceDimensions> MatF;
typedef Vec<int, SpaceDimensions> VecI;


struct GpuParams
{
    VecI paramSize;
    VecI subsampledSize;
    int subsample;
    int subsampleNeighborhood;
    const VecI *gridShift;
    Kernel kernel;
    BsplineImage<ScalarType, SpaceDimensions> movingImage;
    ImageNearest<ScalarType, SpaceDimensions> fixedImage;
    ImageNearest<VecF, SpaceDimensions> cpImage;
    ImageNearest<ScalarType, SpaceDimensions> cpwImage;
    ImageNearest<VecF, SpaceDimensions> displacementField;
    bool useDisplacementField;
    ScalarType regRKHS;
    ScalarType regRD;
    ScalarType regPG;
    ScalarType regRDScaling;
    ScalarType regPGScaling;

    VecF *gradients;
    VecF *displacements;
    int *pixelsCounted;
    ScalarType *derivatives;
    ScalarType *bilateralRegularization;
};

class GpuKernelEvaluator
{
private:
    typedef Vec<ScalarType, SpaceDimensions> VecF;
    typedef Mat<ScalarType, SpaceDimensions> MatF;
    typedef Vec<int, SpaceDimensions> VecI;

    UniformRng _rng;

    int _numberOfParameters;
    int _subsample;
    int _subsampleNeighborhood;
    ScalarType _regRKHS;
    ScalarType _regRD;
    ScalarType _regPG;
    ScalarType _regRDScaling;
    ScalarType _regPGScaling;
    VecI _subsampledSize;
    VecI _subsampledNeighborhoodSize;
    const VecF *_displacementFieldPtr;
    std::unique_ptr<ScalarType[]> _diffs;
    std::unique_ptr<ScalarType[]> _fvalues;
    std::unique_ptr<ScalarType[]> _mvalues;
    // std::unique_ptr<ScalarType[]> _ffvalues;
    // std::unique_ptr<ScalarType[]> _mmvalues;
    // std::unique_ptr<ScalarType[]> _fmvalues;
    std::unique_ptr<ScalarType[]> _ccvalues;
    std::unique_ptr<ScalarType[]> _derivativesF;
    std::unique_ptr<ScalarType[]> _derivativesM;
    std::unique_ptr<ScalarType[]> _bilateralRegularization;
    std::unique_ptr<int[]> _numberOfPixelsCounted;
    std::unique_ptr<VecI[]> _cubeOffsets;
    std::unique_ptr<VecI[]> _cubeNeighborhoodOffsets;

    cuda_ptr<ScalarType> _deviceDerivatives;
    cuda_ptr<ScalarType> _deviceDerivativesF;
    cuda_ptr<ScalarType> _deviceDerivativesM;
    cuda_ptr<ScalarType> _deviceBilateralRegularization;
    cuda_ptr<ScalarType> _deviceDiffs;
    cuda_ptr<ScalarType> _deviceFvalues;
    cuda_ptr<ScalarType> _deviceMvalues;
    // cuda_ptr<ScalarType> _deviceFFvalues;
    // cuda_ptr<ScalarType> _deviceMMvalues;
    // cuda_ptr<ScalarType> _deviceFMvalues;
    cuda_ptr<ScalarType> _deviceCCvalues;
    cuda_ptr<int> _deviceNumberOfPixelsCounted;
    cuda_ptr<VecF> _deviceGradients;

    cuda_ptr<ScalarType> _deviceMovingImage;
    cuda_ptr<ScalarType> _deviceFixedImage;
    cuda_ptr<VecF> _deviceCpImage;
    cuda_ptr<ScalarType> _deviceCpwImage;
    cuda_ptr<ScalarType> _deviceWImage;
    cuda_ptr<MatF> _deviceWTensor;
    cuda_ptr<VecF> _deviceDisplacementField;
    cuda_ptr<VecI> _deviceCubeOffsets;
    cuda_ptr<VecI> _deviceCubeNeighborhoodOffsets;

    Kernel _kernel;

    BsplineImage<ScalarType, SpaceDimensions> _movingImage;
    ImageNearest<ScalarType, SpaceDimensions> _fixedImage;
    ImageNearest<VecF, SpaceDimensions> _displacementField;
    ImageNearest<VecF, SpaceDimensions> _cpImage;
    ImageNearest<ScalarType, SpaceDimensions> _cpwImage;

    bool is_evaluated_once;

public:
    struct EvaluationResult
    {
        ScalarType  measure;
        ScalarType  reg_value_rkhs;
        ScalarType  reg_value_rd;
        ScalarType  reg_value_pg;
        unsigned    numPixelsCountedGpu;
    };

    GpuKernelEvaluator(int numParameters, 
                       int subsample, 
                       int subsampleNeighborhood, 
                       Kernel kernel,
                       BsplineImage<ScalarType, SpaceDimensions> movingImage,
                       ImageNearest<ScalarType, SpaceDimensions> fixedImage,
                       ImageNearest<VecF, SpaceDimensions> cpImage,
                       ImageNearest<ScalarType, SpaceDimensions> cpwImage);

    void SetRegularizerRKHS(ScalarType weight);
    void SetRegularizerRD  (ScalarType weight, ScalarType scaling=1.0);
    void SetRegularizerPG  (ScalarType weight, ScalarType scaling=1.0);

    enum class MEASURE {MSE, NCC, LCC};

    EvaluationResult getValue(MEASURE metric, 
                              const VecF *cpData, 
                              const ScalarType *cpwData, 
                              ImageNearest<VecF, SpaceDimensions> displacementField,
                              bool do_resampling=true);

    EvaluationResult getValueAndDerivative(MEASURE metric,
                              const VecF *cpData, 
                              const ScalarType *cpwData, 
                              ImageNearest<VecF, SpaceDimensions> displacementField, 
                              ScalarType *derivatives,
                              bool do_resampling=true);

    void evaluateDisplacementField(const VecF *cpData, 
                              const ScalarType *cpwData, 
                              ImageNearest<VecF, SpaceDimensions> displacementField, 
                              VecF *dst);
};