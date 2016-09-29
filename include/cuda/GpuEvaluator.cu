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

#include "GpuEvaluator.h"
#include "CudaUtils.h"

#include "GpuEvaluatorDisplacementField.cu"
#include "GpuEvaluatorMSE.cu"
#include "GpuEvaluatorNCC.cu"
#include "GpuEvaluatorLCC.cu"
#include "GpuEvaluatorRegularizers.cu"


GpuKernelEvaluator::GpuKernelEvaluator(int numParameters, 
                                       int subsample, 
                                       int subsampleNeighborhood, 
                                       Kernel kernel,
                                       BsplineImage<ScalarType, SpaceDimensions> movingImage,
                                       ImageNearest<ScalarType, SpaceDimensions> fixedImage,
                                       ImageNearest<VecF, SpaceDimensions> cpImage,
                                       ImageNearest<ScalarType, SpaceDimensions> cpwImage)
: _numberOfParameters(numParameters),
  _subsample(subsample),
  _subsampleNeighborhood(subsampleNeighborhood),
  _regRKHS(0),
  _regRD(0),
  _regPG(0),
  _regRDScaling(1.0),
  _regPGScaling(1.0),
  _subsampledSize(fixedImage.size()/_subsample),
  _displacementFieldPtr(nullptr),
  _diffs(new ScalarType[fixedImage.size().product()]),
  _fvalues(new ScalarType[fixedImage.size().product()]),
  _mvalues(new ScalarType[fixedImage.size().product()]),
//   _ffvalues(new ScalarType[fixedImage.size().product()]),
//   _mmvalues(new ScalarType[fixedImage.size().product()]),
//   _fmvalues(new ScalarType[fixedImage.size().product()]),
  _ccvalues(new ScalarType[fixedImage.size().product()]),
  _derivativesF(new ScalarType[numParameters]),
  _derivativesM(new ScalarType[numParameters]),
  _numberOfPixelsCounted(new int[fixedImage.size().product()]),
  _cubeOffsets(new VecI[_subsampledSize.product()]),
  _deviceDerivatives(allocCuda<ScalarType>(numParameters)),
  _deviceDerivativesF(allocCuda<ScalarType>(numParameters)),
  _deviceDerivativesM(allocCuda<ScalarType>(numParameters)),
  _deviceDiffs(allocCuda<ScalarType>(fixedImage.size().product())),
  _deviceFvalues(allocCuda<ScalarType>(fixedImage.size().product())),
  _deviceMvalues(allocCuda<ScalarType>(fixedImage.size().product())),
//   _deviceFFvalues(allocCuda<ScalarType>(fixedImage.size().product())),
//   _deviceMMvalues(allocCuda<ScalarType>(fixedImage.size().product())),
//   _deviceFMvalues(allocCuda<ScalarType>(fixedImage.size().product())),
  _deviceCCvalues(allocCuda<ScalarType>(fixedImage.size().product())),
  _deviceNumberOfPixelsCounted(allocCuda<int>(fixedImage.size().product())),
  _deviceGradients(allocCuda<VecF>(fixedImage.size().product())),
  _deviceMovingImage(allocCuda<ScalarType>(movingImage.size().product(), movingImage.coeffs())),
  _deviceFixedImage(allocCuda<ScalarType>(fixedImage.size().product(), fixedImage.data())),
  _deviceCpImage(allocCuda<VecF>(cpImage.size().product())),
  _deviceCpwImage(allocCuda<ScalarType>(cpwImage.size().product())),
  _deviceWImage(allocCuda<ScalarType>(kernel.wImage().size().product(), kernel.wImage().data())),
  _deviceWTensor(allocCuda<MatF>(kernel.wTensor().size().product(), kernel.wTensor().data())),
  _deviceCubeOffsets(_subsample > 1 ? allocCuda<VecI>(_subsampledSize.product()) : nullptr),
  _kernel(kernel),
  _movingImage(movingImage),
   _fixedImage(fixedImage),
      _cpImage(cpImage),
     _cpwImage(cpwImage),
  is_evaluated_once(false)
{
    _movingImage.assignCoeffs(_deviceMovingImage.get());
    _fixedImage.assignData(_deviceFixedImage.get());
    _cpImage.assignData(_deviceCpImage.get());
    _cpwImage.assignData(_deviceCpwImage.get());
    _kernel.wImage().assignData(_deviceWImage.get());
    _kernel.wTensor().assignData(_deviceWTensor.get());
}


GpuKernelEvaluator::EvaluationResult 
GpuKernelEvaluator::getValue(MEASURE metric,
                             const VecF *cpData, 
                             const ScalarType *cpwData, 
                             ImageNearest<VecF, SpaceDimensions> displacementField,
                             bool do_resampling)
{
    VecI fixedSize = _fixedImage.size();
    VecI paramSize = _cpImage.size();

    cudaCheck(cudaMemcpy( _deviceCpImage.get(),  cpData,  _cpImage.size().product()*sizeof(VecF),       cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cudaCheck(cudaMemcpy(_deviceCpwImage.get(), cpwData, _cpwImage.size().product()*sizeof(ScalarType), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    if (_subsample > 1) {
        int numTexels  = _subsampledSize.product();

        if(!is_evaluated_once || do_resampling){
            for (int i = 0; i < numTexels; ++i)
                _cubeOffsets[i] = VecI(_rng.nextV<SpaceDimensions>()*static_cast<ScalarType>(_subsample));

            is_evaluated_once = true;
        }

        cudaCheck(cudaMemcpy(_deviceCubeOffsets.get(), _cubeOffsets.get(), _subsampledSize.product()*sizeof(VecI), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    }

    if (displacementField.data() != _displacementFieldPtr) {
        _displacementFieldPtr = displacementField.data();
        _displacementField = displacementField;
        _deviceDisplacementField = allocCuda<VecF>(displacementField.size().product());
        cudaCheck(cudaMemcpy(_deviceDisplacementField.get(), _displacementFieldPtr, displacementField.size().product()*sizeof(VecF), cudaMemcpyHostToDevice), __FILE__, __LINE__);
        _displacementField.assignData(_deviceDisplacementField.get());
    }

    GpuParams params {
        paramSize,
        _subsampledSize,
        _subsample,
        _subsampleNeighborhood,
        _deviceCubeOffsets.get(),
        _kernel,
        _movingImage,
        _fixedImage,
        _cpImage,
        _cpwImage,
        _displacementField,
        _displacementFieldPtr != nullptr,
        _regRKHS,
        _regRD,
        _regPG,
        _regRDScaling,
        _regPGScaling,
        _deviceGradients.get(),
        nullptr, // displacements
        _deviceNumberOfPixelsCounted.get(),
        nullptr, // derivatives
        _deviceBilateralRegularization.get()
    };


    //------------------------------------------------------------------------------------------------------
    // calculating loss function values

#if SpaceDimensions == 3
    dim3 threadsPerBlock(4, 4, 4);
#else
    dim3 threadsPerBlock(16, 16);
#endif

    dim3 blocksPerGrid = dim3(
        (_subsampledSize[0] + threadsPerBlock.x - 1)/threadsPerBlock.x,
        (_subsampledSize[1] + threadsPerBlock.y - 1)/threadsPerBlock.y,
#if SpaceDimensions == 3
        (_subsampledSize[2] + threadsPerBlock.z - 1)/threadsPerBlock.z
#else
        1
#endif
    );


    ScalarType measure = 0.0;
    int pixelsCounted = 0;

    if(metric == MEASURE::MSE){
        GpuMSEParams mse_params{
            _deviceDiffs.get(),
        };

        if (_subsample == 1){
            resolveValueMSE<false>(blocksPerGrid, threadsPerBlock, params, mse_params);
        }
        else{
            resolveValueMSE<true> (blocksPerGrid, threadsPerBlock, params, mse_params);
        }
        cudaCheck(cudaMemcpy(_diffs.get(),                 _deviceDiffs.get(), fixedSize.product()*sizeof(                ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaMemcpy(_numberOfPixelsCounted.get(), _deviceNumberOfPixelsCounted.get(), fixedSize.product()*sizeof(       int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        ScalarType mse = 0.0;
        pixelsCounted = 0;
        for (int i = 0; i < _subsampledSize.product(); ++i) {
            mse += _diffs[i]*_diffs[i]; // TODO: could be done with only fvalues and mvalues. diffs is not needed.
            pixelsCounted += _numberOfPixelsCounted[i];
        }

        mse /= pixelsCounted; // pixelsCounted should be checked to zero
        measure = mse;
    }
    else if(metric == MEASURE::NCC){
        GpuNCCParams ncc_params{
            0, // mean_mf;
            0, // mean_mm;
            _deviceFvalues.get(),
            _deviceMvalues.get(),
            // _deviceFFvalues.get(),
            // _deviceMMvalues.get(),
            // _deviceFMvalues.get(),
            nullptr,
            nullptr
        };

        if (_subsample == 1){
            resolveValueNCC<false>(blocksPerGrid, threadsPerBlock, params, ncc_params);
        }
        else{
            resolveValueNCC<true> (blocksPerGrid, threadsPerBlock, params, ncc_params);
        }
        cudaCheck(cudaMemcpy(_fvalues.get(),               _deviceFvalues.get(), fixedSize.product()*sizeof(              ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaMemcpy(_mvalues.get(),               _deviceMvalues.get(), fixedSize.product()*sizeof(              ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        // cudaCheck(cudaMemcpy(_ffvalues.get(),              _deviceFFvalues.get(), fixedSize.product()*sizeof(             ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        // cudaCheck(cudaMemcpy(_mmvalues.get(),              _deviceMMvalues.get(), fixedSize.product()*sizeof(             ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        // cudaCheck(cudaMemcpy(_fmvalues.get(),              _deviceFMvalues.get(), fixedSize.product()*sizeof(             ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);        
        cudaCheck(cudaMemcpy(_numberOfPixelsCounted.get(), _deviceNumberOfPixelsCounted.get(), fixedSize.product()*sizeof(       int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        ScalarType denom = 1.0;
        ScalarType smm = 0.0;
        ScalarType sfm = 0.0;

        ScalarType sf = 0.0;
        ScalarType sm = 0.0;
        ScalarType sff = 0.0;
        smm = 0.0;
        sfm = 0.0;
        pixelsCounted = 0;
        for (int i = 0; i < _subsampledSize.product(); ++i) {
            // sff += _ffvalues[i];
            // smm += _mmvalues[i];
            // sfm += _fmvalues[i];
            sff += _fvalues[i]*_fvalues[i];
            smm += _mvalues[i]*_mvalues[i];
            sfm += _fvalues[i]*_mvalues[i];
            sf  += _fvalues[i];
            sm  += _mvalues[i];
            pixelsCounted += _numberOfPixelsCounted[i];
        }

        // subtract mean
        sff -= (sf*sf/pixelsCounted);
        smm -= (sm*sm/pixelsCounted);
        sfm -= (sf*sm/pixelsCounted);

        denom = -1.0 * std::sqrt(sff*smm);
        ScalarType ncc = 0.0;
        if(denom!=0)
            ncc = sfm/denom;

        measure = ncc;
    }
    else if(metric == MEASURE::LCC){
        GpuLCCParams lcc_params{
            _deviceCCvalues.get(),
        };

        if (_subsample == 1){
            #ifndef EXEC_SINGLE_THREAD
            resolveValueLCC<false>(blocksPerGrid, threadsPerBlock, params, lcc_params);
            #else
            resolveValueLCC<false>(1, 1, params, lcc_params);
            #endif
        }
        else{
            #ifndef EXEC_SINGLE_THREAD
            resolveValueLCC<true> (blocksPerGrid, threadsPerBlock, params, lcc_params);
            #else
            resolveValueLCC<true> (1, 1, params, lcc_params);
            #endif
        }

        cudaCheck(cudaMemcpy(             _ccvalues.get(), _deviceCCvalues             .get(), fixedSize.product()*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaMemcpy(_numberOfPixelsCounted.get(), _deviceNumberOfPixelsCounted.get(), fixedSize.product()*sizeof(       int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        ScalarType lcc = 0.0;
        pixelsCounted = 0;
        for (int i = 0; i < _subsampledSize.product(); ++i) {
            // ScalarType v = _ccvalues[i];
            // if(std::isfinite(v))
            //     lcc += v;  
            lcc += _ccvalues[i];  
            pixelsCounted += _numberOfPixelsCounted[i];
        }
        lcc /= pixelsCounted; // pixelsCounted should be checked to zero
        measure = lcc;
    }


    //------------------------------------------------------------------------------------------------------
    // calculating bilateral regularizer

    // copy current derivatives to gpu
    ScalarType reg_value_rkhs = 0;
    ScalarType reg_value_rd   = 0;
    ScalarType reg_value_pg   = 0;

    /** RKHS norm: c'Kc */
    if(_regRKHS>0){
        if (params.kernel.useWeightImage())
            regularizerRKHS<true, false, false><<<blocksPerGrid, threadsPerBlock>>>(params);
        else if (params.kernel.useWeightTensor())
            regularizerRKHS<false, true, false><<<blocksPerGrid, threadsPerBlock>>>(params);
        else
            regularizerRKHS<false, false, false><<<blocksPerGrid, threadsPerBlock>>>(params);

        cudaCheck(cudaMemcpy(_bilateralRegularization.get(), _deviceBilateralRegularization.get(), _numberOfParameters/SpaceDimensions*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        for (int i = 0; i < _numberOfParameters/SpaceDimensions; ++i)
            reg_value_rkhs += _bilateralRegularization[i];
    }

    if(_regRD>0){
        if (params.kernel.useWeightImage())
            regularizerRD<true, false, false><<<blocksPerGrid, threadsPerBlock>>>(params);
        else if (params.kernel.useWeightTensor())
            regularizerRD<false, true, false><<<blocksPerGrid, threadsPerBlock>>>(params);
        else
            regularizerRD<false, false, false><<<blocksPerGrid, threadsPerBlock>>>(params);

        cudaCheck(cudaMemcpy(_bilateralRegularization.get(), _deviceBilateralRegularization.get(), _numberOfParameters/SpaceDimensions*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        for (int i = 0; i < _numberOfParameters/SpaceDimensions; ++i)
            reg_value_rd += _bilateralRegularization[i];
    }

    if(_regPG){
        if (params.kernel.useWeightImage())
            regularizerPG<true, false, false><<<blocksPerGrid, threadsPerBlock>>>(params);
        else if (params.kernel.useWeightTensor())
            regularizerPG<false, true, false><<<blocksPerGrid, threadsPerBlock>>>(params);
        else
            regularizerPG<false, false, false><<<blocksPerGrid, threadsPerBlock>>>(params);

        cudaCheck(cudaMemcpy(_bilateralRegularization.get(), _deviceBilateralRegularization.get(), _numberOfParameters/SpaceDimensions*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        for (int i = 0; i < _numberOfParameters/SpaceDimensions; ++i)
            reg_value_pg += _bilateralRegularization[i];
    }


    return EvaluationResult{measure, reg_value_rkhs, reg_value_rd, reg_value_pg, pixelsCounted};
}


// TODO
GpuKernelEvaluator::EvaluationResult 
GpuKernelEvaluator::getValueAndDerivative(MEASURE metric,
                             const VecF *cpData, 
                             const ScalarType *cpwData, 
                             ImageNearest<VecF, SpaceDimensions> displacementField, 
                             ScalarType *derivatives, 
                             bool do_resampling)
{
    VecI fixedSize = _fixedImage.size();
    VecI paramSize = _cpImage.size();

    cudaCheck(cudaMemcpy( _deviceCpImage.get(),  cpData,  _cpImage.size().product()*sizeof(VecF),       cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cudaCheck(cudaMemcpy(_deviceCpwImage.get(), cpwData, _cpwImage.size().product()*sizeof(ScalarType), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    if (_subsample > 1) {
        int numTexels  = _subsampledSize.product();

        if(!is_evaluated_once || do_resampling){
            for (int i = 0; i < numTexels; ++i)
                _cubeOffsets[i] = VecI(_rng.nextV<SpaceDimensions>()*static_cast<ScalarType>(_subsample));

            is_evaluated_once = true;
        }
        cudaCheck(cudaMemcpy(_deviceCubeOffsets.get(), _cubeOffsets.get(), _subsampledSize.product()*sizeof(VecI), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    }

    if (displacementField.data() != _displacementFieldPtr) {
        _displacementFieldPtr = displacementField.data();
        _displacementField = displacementField;
        _deviceDisplacementField = allocCuda<VecF>(displacementField.size().product());
        cudaCheck(cudaMemcpy(_deviceDisplacementField.get(), _displacementFieldPtr, displacementField.size().product()*sizeof(VecF), cudaMemcpyHostToDevice), __FILE__, __LINE__);
        _displacementField.assignData(_deviceDisplacementField.get());
    }

    GpuParams params {
        paramSize,
        _subsampledSize,
        _subsample,
        _subsampleNeighborhood,
        _deviceCubeOffsets.get(),
        _kernel,
        _movingImage,
        _fixedImage,
        _cpImage,
        _cpwImage,
        _displacementField,
        _displacementFieldPtr != nullptr,
        _regRKHS,
        _regRD,
        _regPG,
        _regRDScaling,
        _regPGScaling,
        _deviceGradients.get(),
        nullptr,
        _deviceNumberOfPixelsCounted.get(),
        _deviceDerivatives.get(),
        _deviceBilateralRegularization.get()
    };


    //------------------------------------------------------------------------------------------------------
    // calculating loss function values

#if SpaceDimensions == 3
    dim3 threadsPerBlock(4, 4, 4);
#else
    dim3 threadsPerBlock(16, 16);
#endif

    dim3 blocksPerGrid = dim3(
        (_subsampledSize[0] + threadsPerBlock.x - 1)/threadsPerBlock.x,
        (_subsampledSize[1] + threadsPerBlock.y - 1)/threadsPerBlock.y,
#if SpaceDimensions == 3
        (_subsampledSize[2] + threadsPerBlock.z - 1)/threadsPerBlock.z
#else
        1
#endif
    );

    ScalarType measure = 0.0;
    int pixelsCounted  = 0;

    ScalarType denom   = 1.0;
    ScalarType smm     = 0.0;
    ScalarType sfm     = 0.0;
    ScalarType mean_mf = 0.0;
    ScalarType mean_mm = 0.0;

    if(metric == MEASURE::MSE){
        GpuMSEParams mse_params{
            _deviceDiffs.get(),
        };

        if (_subsample == 1){
            resolveValueMSE<false>(blocksPerGrid, threadsPerBlock, params, mse_params);
        }
        else{
            resolveValueMSE<true> (blocksPerGrid, threadsPerBlock, params, mse_params);
        }
        cudaCheck(cudaMemcpy(_diffs.get(),                 _deviceDiffs.get(), fixedSize.product()*sizeof(                ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaMemcpy(_numberOfPixelsCounted.get(), _deviceNumberOfPixelsCounted.get(), fixedSize.product()*sizeof(       int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        ScalarType mse = 0.0;
        pixelsCounted = 0;
        for (int i = 0; i < _subsampledSize.product(); ++i) {
            mse += _diffs[i]*_diffs[i]; // TODO: could be done with only fvalues and mvalues. diffs is not needed.
            pixelsCounted += _numberOfPixelsCounted[i];
        }

        mse /= pixelsCounted; // pixelsCounted should be checked to zero
        measure = mse;
    }
    else if(metric == MEASURE::NCC){
        GpuNCCParams ncc_params{
            0, // mean_mf;
            0, // mean_mm;
            _deviceFvalues.get(),
            _deviceMvalues.get(),
            // _deviceFFvalues.get(),
            // _deviceMMvalues.get(),
            // _deviceFMvalues.get(),
            _deviceDerivativesF.get(),
            _deviceDerivativesM.get()
        };

        if (_subsample == 1){
            resolveValueNCC<false>(blocksPerGrid, threadsPerBlock, params, ncc_params);
        }
        else{
            resolveValueNCC<true>(blocksPerGrid, threadsPerBlock, params, ncc_params);
        }
        cudaCheck(cudaMemcpy(_fvalues.get(),               _deviceFvalues.get(), fixedSize.product()*sizeof(              ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaMemcpy(_mvalues.get(),               _deviceMvalues.get(), fixedSize.product()*sizeof(              ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        // cudaCheck(cudaMemcpy(_ffvalues.get(),              _deviceFFvalues.get(), fixedSize.product()*sizeof(             ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        // cudaCheck(cudaMemcpy(_mmvalues.get(),              _deviceMMvalues.get(), fixedSize.product()*sizeof(             ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        // cudaCheck(cudaMemcpy(_fmvalues.get(),              _deviceFMvalues.get(), fixedSize.product()*sizeof(             ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);        
        cudaCheck(cudaMemcpy(_numberOfPixelsCounted.get(), _deviceNumberOfPixelsCounted.get(), fixedSize.product()*sizeof(       int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        ScalarType sf = 0.0;
        ScalarType sm = 0.0;
        ScalarType sff = 0.0;
        smm = 0.0;
        sfm = 0.0;
        pixelsCounted = 0;
        for (int i = 0; i < _subsampledSize.product(); ++i) {
            // sff += _ffvalues[i];
            // smm += _mmvalues[i];
            // sfm += _fmvalues[i];
            sff += _fvalues[i]*_fvalues[i];
            smm += _mvalues[i]*_mvalues[i];
            sfm += _fvalues[i]*_mvalues[i];
            sf  += _fvalues[i];
            sm  += _mvalues[i];
            pixelsCounted += _numberOfPixelsCounted[i];
        }

        // subtract mean
        sff -= (sf*sf/pixelsCounted);
        smm -= (sm*sm/pixelsCounted);
        sfm -= (sf*sm/pixelsCounted);

        denom = -1.0 * std::sqrt(sff*smm);
        ScalarType ncc = 0.0;
        if(denom!=0)
            ncc = sfm/denom;

        measure = ncc;

        // save mean f and m values in GPU parameter struct
        // since they are needed in calculating the derivative
        mean_mf = sf/pixelsCounted;
        mean_mm = sm/pixelsCounted;
    }
    else if(metric == MEASURE::LCC){
        GpuLCCParams lcc_params{
            _deviceCCvalues.get(),
        };

        if (_subsample == 1){
            #ifndef EXEC_SINGLE_THREAD
            resolveValueLCC<false>(blocksPerGrid, threadsPerBlock, params, lcc_params);
            #else
            resolveValueLCC<false>(1, 1, params, lcc_params);
            #endif
        }
        else{
            #ifndef EXEC_SINGLE_THREAD
            resolveValueLCC<true>(blocksPerGrid, threadsPerBlock, params, lcc_params);
            #else
            resolveValueLCC<true>(1, 1, params, lcc_params);
            #endif
        }

        cudaCheck(cudaMemcpy(             _ccvalues.get(), _deviceCCvalues             .get(), fixedSize.product()*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaMemcpy(_numberOfPixelsCounted.get(), _deviceNumberOfPixelsCounted.get(), fixedSize.product()*sizeof(       int), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        ScalarType lcc = 0.0;
        pixelsCounted = 0;
        for (int i = 0; i < _subsampledSize.product(); ++i) {
            // ScalarType v = _ccvalues[i];
            // if(std::isfinite(v))
            //     lcc += v;  
            lcc += _ccvalues[i];  
            pixelsCounted += _numberOfPixelsCounted[i];
        }
        lcc /= pixelsCounted; // pixelsCounted should be checked to zero
        measure = lcc;
    }


    //------------------------------------------------------------------------------------------------------
    // calculating derivatives
    cudaMemset(_deviceDerivatives.get(), 0, _numberOfParameters*sizeof(ScalarType));

    blocksPerGrid = dim3(
        (paramSize[0] + threadsPerBlock.x - 1)/threadsPerBlock.x,
        (paramSize[1] + threadsPerBlock.y - 1)/threadsPerBlock.y,
#if SpaceDimensions == 3
        (paramSize[2] + threadsPerBlock.z - 1)/threadsPerBlock.z
#else
        1
#endif
    );

    if(metric == MEASURE::MSE){
        GpuMSEParams mse_params{
            _deviceDiffs.get(),
        };

        if (_subsample == 1){
            resolveDerivativeMSE<false>(blocksPerGrid, threadsPerBlock, params, mse_params);
        }
        else{
            resolveDerivativeMSE<true> (blocksPerGrid, threadsPerBlock, params, mse_params);
        }
        cudaCheck(cudaMemcpy(derivatives, _deviceDerivatives.get(), _numberOfParameters*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        for (int i = 0; i < _numberOfParameters; ++i)
            derivatives[i] /= pixelsCounted;
    }
    else if(metric == MEASURE::NCC){
        GpuNCCParams ncc_params{
            mean_mf,
            mean_mm,
            _deviceFvalues.get(),
            _deviceMvalues.get(),
            // _deviceFFvalues.get(),
            // _deviceMMvalues.get(),
            // _deviceFMvalues.get(),
            _deviceDerivativesF.get(),
            _deviceDerivativesM.get()
        };

        if (_subsample == 1){
            resolveDerivativeNCC<false>(blocksPerGrid, threadsPerBlock, params, ncc_params);
        }
        else{
            resolveDerivativeNCC<true> (blocksPerGrid, threadsPerBlock, params, ncc_params);
        }
        cudaCheck(cudaMemcpy(_derivativesF.get(), _deviceDerivativesF.get(), _numberOfParameters*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaMemcpy(_derivativesM.get(), _deviceDerivativesM.get(), _numberOfParameters*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        for (int i = 0; i < _numberOfParameters; ++i){
            if(denom!=0)
                derivatives[i] = ( _derivativesF[i] - (sfm/smm)*_derivativesM[i] ) / denom;
            else
                derivatives[i] = 0.0;
        }
    }
    else if(metric == MEASURE::LCC){
        if (_subsample == 1){
            resolveDerivativeLCC<false>(blocksPerGrid, threadsPerBlock, params);
        }
        else{
            resolveDerivativeLCC<true> (blocksPerGrid, threadsPerBlock, params);
        }
        cudaCheck(cudaMemcpy(derivatives, _deviceDerivatives.get(), _numberOfParameters*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        for (int i = 0; i < _numberOfParameters; ++i)
            derivatives[i] /= pixelsCounted;
    }

    //------------------------------------------------------------------------------------------------------
    // calculating bilateral regularizer

    // copy current derivatives to gpu
    if(_regRKHS>0 || _regRD>0 || _regPG>0){
        cudaCheck(cudaMemcpy(_deviceDerivatives.get(), derivatives, _numberOfParameters*sizeof(ScalarType), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    }

    ScalarType reg_value_rkhs = 0;
    ScalarType reg_value_rd   = 0;
    ScalarType reg_value_pg   = 0;

    /** RKHS norm: c'Kc */
    if(_regRKHS>0){
        if (params.kernel.useWeightImage())
            regularizerRKHS<true, false,  true><<<blocksPerGrid, threadsPerBlock>>>(params);
        else if (params.kernel.useWeightTensor())
            regularizerRKHS<false, true,  true><<<blocksPerGrid, threadsPerBlock>>>(params);
        else
            regularizerRKHS<false, false, true><<<blocksPerGrid, threadsPerBlock>>>(params);

        cudaCheck(cudaMemcpy(_bilateralRegularization.get(), _deviceBilateralRegularization.get(), _numberOfParameters/SpaceDimensions*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        for (int i = 0; i < _numberOfParameters/SpaceDimensions; ++i)
            reg_value_rkhs += _bilateralRegularization[i];
    }

    if(_regRD>0){
        if (params.kernel.useWeightImage())
            regularizerRD<true, false,  true><<<blocksPerGrid, threadsPerBlock>>>(params);
        else if (params.kernel.useWeightTensor())
            regularizerRD<false, true,  true><<<blocksPerGrid, threadsPerBlock>>>(params);
        else
            regularizerRD<false, false, true><<<blocksPerGrid, threadsPerBlock>>>(params);

        cudaCheck(cudaMemcpy(_bilateralRegularization.get(), _deviceBilateralRegularization.get(), _numberOfParameters/SpaceDimensions*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        for (int i = 0; i < _numberOfParameters/SpaceDimensions; ++i)
            reg_value_rd += _bilateralRegularization[i];
    }

    if(_regPG){
        if (params.kernel.useWeightImage())
            regularizerPG<true, false,  true><<<blocksPerGrid, threadsPerBlock>>>(params);
        else if (params.kernel.useWeightTensor())
            regularizerPG<false, true,  true><<<blocksPerGrid, threadsPerBlock>>>(params);
        else
            regularizerPG<false, false, true><<<blocksPerGrid, threadsPerBlock>>>(params);

        cudaCheck(cudaMemcpy(_bilateralRegularization.get(), _deviceBilateralRegularization.get(), _numberOfParameters/SpaceDimensions*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
        cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);

        for (int i = 0; i < _numberOfParameters/SpaceDimensions; ++i)
            reg_value_pg += _bilateralRegularization[i];
    }


    // fetch current derivative which has been updated by the bilateral regularizers
    if(_regRKHS>0 || _regRD>0 || _regPG>0){
        cudaCheck(cudaMemcpy(derivatives, _deviceDerivatives.get(), _numberOfParameters*sizeof(ScalarType), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    }

    return EvaluationResult{measure, reg_value_rkhs, reg_value_rd, reg_value_pg, pixelsCounted};
}

void GpuKernelEvaluator::evaluateDisplacementField(const VecF *cpData, const ScalarType *cpwData, ImageNearest<VecF, SpaceDimensions> displacementField, VecF *dst)
{
    cudaCheck(cudaMemcpy( _deviceCpImage.get(),  cpData,  _cpImage.size().product()*sizeof(VecF),       cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cudaCheck(cudaMemcpy(_deviceCpwImage.get(), cpwData, _cpwImage.size().product()*sizeof(ScalarType), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    if (displacementField.data() != _displacementFieldPtr) {
        _displacementFieldPtr = displacementField.data();
        _displacementField = displacementField;
        _deviceDisplacementField.reset();
        if (_displacementFieldPtr) {
            _deviceDisplacementField = allocCuda<VecF>(displacementField.size().product());
            cudaCheck(cudaMemcpy(_deviceDisplacementField.get(), _displacementFieldPtr, displacementField.size().product()*sizeof(VecF), cudaMemcpyHostToDevice), __FILE__, __LINE__);
        }
        _displacementField.assignData(_deviceDisplacementField.get());
    }

    GpuParams params {
        _cpImage.size(),
        _fixedImage.size(),
        1,
        1,
        nullptr,
        _kernel,
        _movingImage,
        _fixedImage,
        _cpImage,
        _cpwImage,
        _displacementField,
        _displacementField.data() != nullptr,
        _regRKHS,
        _regRD,
        _regPG,
        _regRDScaling,
        _regPGScaling,
        nullptr,
        _deviceGradients.get(),
        nullptr,
        nullptr,
        nullptr
    };

#if SpaceDimensions == 3
    dim3 threadsPerBlock(4, 4, 4);
#else
    dim3 threadsPerBlock(16, 16);
#endif

    dim3 blocksPerGrid = dim3(
        (_fixedImage.size()[0] + threadsPerBlock.x - 1)/threadsPerBlock.x,
        (_fixedImage.size()[1] + threadsPerBlock.y - 1)/threadsPerBlock.y,
#if SpaceDimensions == 3
        (_fixedImage.size()[2] + threadsPerBlock.z - 1)/threadsPerBlock.z
#else
        1
#endif
    );

    if (params.kernel.useWeightImage())
        evaluateDisplacement<true, false><<<blocksPerGrid, threadsPerBlock>>>(params);
    else if (params.kernel.useWeightTensor())
        evaluateDisplacement<false, true><<<blocksPerGrid, threadsPerBlock>>>(params);
    else
        evaluateDisplacement<false, false><<<blocksPerGrid, threadsPerBlock>>>(params);

    cudaCheck(cudaPeekAtLastError(), __FILE__, __LINE__);
    cudaCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    cudaCheck(cudaMemcpy(dst, _deviceGradients.get(), _fixedImage.size().product()*sizeof(VecF), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
}

void GpuKernelEvaluator::SetRegularizerRKHS(ScalarType weight){
    if (weight != _regRKHS) {
        _regRKHS = weight;
        _bilateralRegularization.reset();
        _deviceBilateralRegularization.reset();

        if (_regRKHS > 0) {
            _bilateralRegularization.reset(new ScalarType[_numberOfParameters/SpaceDimensions]);
            _deviceBilateralRegularization = allocCuda<ScalarType>(_numberOfParameters/SpaceDimensions);
        }
    }
}

void GpuKernelEvaluator::SetRegularizerRD(ScalarType weight, ScalarType scaling){
    if (weight != _regRD) {
        _regRD = weight;
        _bilateralRegularization.reset();
        _deviceBilateralRegularization.reset();

        if (_regRD > 0) {
            _bilateralRegularization.reset(new ScalarType[_numberOfParameters/SpaceDimensions]);
            _deviceBilateralRegularization = allocCuda<ScalarType>(_numberOfParameters/SpaceDimensions);
        }
    }
    if(scaling<=0.0){
        std::cout << "Attention: scaling of regularizer must be strictly positive!" << std::endl;
    }
    _regRDScaling = scaling;
}

void GpuKernelEvaluator::SetRegularizerPG(ScalarType weight, ScalarType scaling){
    if (weight != _regPG) {
        _regPG = weight;
        _bilateralRegularization.reset();
        _deviceBilateralRegularization.reset();

        if (_regPG > 0) {
            _bilateralRegularization.reset(new ScalarType[_numberOfParameters/SpaceDimensions]);
            _deviceBilateralRegularization = allocCuda<ScalarType>(_numberOfParameters/SpaceDimensions);
        }
    }
    if(scaling<=0.0){
        std::cout << "Attention: scaling of regularizer must be strictly positive!" << std::endl;
    }
    _regPGScaling = scaling;
}