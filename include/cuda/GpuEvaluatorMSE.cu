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

struct GpuMSEParams
{
    ScalarType *diffs;
};

#if SpaceDimensions == 3

template<bool DoSubsample, bool UseWeightImage, bool UseWeightTensor>
__global__ void evaluateValueMSE(GpuParams params, GpuMSEParams mse_params)
{
    Vec3i size = params.subsampledSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    if (x >= size[0] || y >= size[1] || z >= size[2])
        return;

    Vec3i samplePos = Vec3i(x, y, z);
    if (DoSubsample)
        samplePos = samplePos*params.subsample + params.gridShift[x + y*size[0] + z*size[0]*size[1]];
    Vec3f  fixedImagePoint = params.fixedImage.toGlobal(Vec3f(samplePos));
    Vec3f centerRegionPoint = fixedImagePoint;
    if (params.useDisplacementField)
        centerRegionPoint += params.displacementField.atGlobal(fixedImagePoint);

    ScalarType support;
    ScalarType sigmaP;
    Mat3f covP;
    if (UseWeightImage) {
        sigmaP  = params.kernel.getSigmaAtPoint(fixedImagePoint);
        support = params.kernel.getRegionSupport(sigmaP);
    } else if (UseWeightTensor) {
        covP = params.kernel.getCovarianceAtPoint(fixedImagePoint);
        support = params.kernel.getRegionSupport(covP);
    } else {
        support = params.kernel.getRegionSupport();
    }

    Vec3f indexPoint = params.cpImage.toLocal(fixedImagePoint);
    Vec3i imgLower = std::min(std::max(Vec3i(indexPoint - support/params.cpImage.scale() + 1.0), Vec3i(0)), params.cpImage.size() - 1);
    Vec3i imgUpper = std::min(std::max(Vec3i(indexPoint + support/params.cpImage.scale()      ), Vec3i(0)), params.cpImage.size() - 1);

    for (int zi = imgLower[2]; zi <= imgUpper[2]; ++zi) {
        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
                Vec3f point_i = params.cpImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)));
                ScalarType k;
                if (UseWeightImage)
                    k = params.kernel.evaluate(fixedImagePoint, point_i, sigmaP);
                else if (UseWeightTensor)
                    k = params.kernel.evaluate(fixedImagePoint, point_i, covP);
                else
                    k = params.kernel.evaluate(fixedImagePoint, point_i);

                centerRegionPoint += params.cpwImage.at(Vec3i(xi, yi, zi))*k*params.cpImage.at(Vec3i(xi, yi, zi));
            }
        }
    }

    ScalarType diff = 0.0;
    Vec3f movingImageGradient(0.0);
    int inside = 0;
    if (params.movingImage.insideGlobal(centerRegionPoint)) {
        ScalarType movingImageValue;
        params.movingImage.derivsGlobal(centerRegionPoint, movingImageValue, movingImageGradient);
        diff = movingImageValue - params.fixedImage.at(samplePos);
        inside = 1;
    }

    mse_params.diffs    [x + y*size[0] + z*size[0]*size[1]] = diff;
    params.gradients    [x + y*size[0] + z*size[0]*size[1]] = movingImageGradient;
    params.pixelsCounted[x + y*size[0] + z*size[0]*size[1]] = inside;
}

template<bool DoSubsample, bool UseWeightImage, bool UseWeightTensor>
__global__ void evaluateDerivativeMSE(GpuParams params, GpuMSEParams mse_params)
{
    Vec3i size = params.paramSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    if (x >= size[0] || y >= size[1] || z >= size[2])
        return;

    Vec3f globalPoint = params.cpImage.toGlobal(Vec3f(Vec3i(x, y, z)));
    ScalarType w_i = params.cpwImage.at(Vec3i(x, y, z));

    ScalarType support;
    ScalarType sigmaP;
    Mat3f covP;
    if (UseWeightImage) {
        sigmaP  = params.kernel.getSigmaAtPoint(globalPoint);
        support = params.kernel.getRegionSupport(sigmaP);
    } else if (UseWeightTensor) {
        covP = params.kernel.getCovarianceAtPoint(globalPoint);
        support = params.kernel.getRegionSupport(covP);
    } else {
        support = params.kernel.getRegionSupport();
    }

    Vec3f indexPoint = params.fixedImage.toLocal(globalPoint);
    Vec3i imgLower, imgUpper;
    if (DoSubsample) {
        imgLower = std::min(std::max(Vec3i((indexPoint - support/params.fixedImage.scale())/float(params.subsample)      ), Vec3i(0)), params.subsampledSize - 1);
        imgUpper = std::min(std::max(Vec3i((indexPoint + support/params.fixedImage.scale())/float(params.subsample) + 1.0), Vec3i(0)), params.subsampledSize - 1);
    } else {
        imgLower = std::min(std::max(Vec3i(indexPoint - support/params.fixedImage.scale()      ), Vec3i(0)), params.fixedImage.size() - 1);
        imgUpper = std::min(std::max(Vec3i(indexPoint + support/params.fixedImage.scale() + 1.0), Vec3i(0)), params.fixedImage.size() - 1);
    }

    Vec3f derivative(0.0);
    for (int zi = imgLower[2]; zi <= imgUpper[2]; ++zi) {
        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
                int idx;
                if (DoSubsample)
                    idx = xi + params.subsampledSize[0]*(yi + params.subsampledSize[1]*zi);
                else
                    idx = params.fixedImage.toIndex(Vec3i(xi, yi, zi));
                ScalarType diff   = mse_params.diffs[idx];
                Vec3f gradient    = params.gradients[idx];

                Vec3f point_i;
                if (DoSubsample)
                    point_i = params.fixedImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)*params.subsample + params.gridShift[idx]));
                else
                    point_i = params.fixedImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)));

                ScalarType k;
                if (UseWeightImage)
                    k = params.kernel.evaluate(globalPoint, point_i, sigmaP);
                else if (UseWeightTensor)
                    k = params.kernel.evaluate(globalPoint, point_i, covP);
                else
                    k = params.kernel.evaluate(globalPoint, point_i);

                derivative += static_cast<ScalarType>(2.0)*diff*w_i*k*gradient;
            }
        }
    }

    const int dimensionStride = params.cpImage.size().product();
    int pixelIndex = params.cpImage.toIndex(Vec3i(x, y, z));
    for (unsigned d = 0; d < SpaceDimensions; d++){
        params.derivatives[pixelIndex + d*dimensionStride] = derivative[d];
    }
}

#else //////////// 2D ///////////

template<bool DoSubsample, bool UseWeightImage, bool UseWeightTensor>
__global__ void evaluateValueMSE(GpuParams params, GpuMSEParams mse_params)
{
    Vec2i size = params.subsampledSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= size[0] || y >= size[1])
        return;

    Vec2i samplePos = Vec2i(x, y);
    if (DoSubsample)
        samplePos = samplePos*params.subsample + params.gridShift[x + y*size[0]];
    Vec2f  fixedImagePoint = params.fixedImage.toGlobal(Vec2f(samplePos));
    Vec2f movingImagePoint = fixedImagePoint;
    if (params.useDisplacementField)
        movingImagePoint += params.displacementField.atGlobal(fixedImagePoint);

    ScalarType support;
    ScalarType sigmaP;
    Mat2f covP;
    if (UseWeightImage) {
        sigmaP  = params.kernel.getSigmaAtPoint(fixedImagePoint);
        support = params.kernel.getRegionSupport(sigmaP);
    } else if (UseWeightTensor) {
        covP = params.kernel.getCovarianceAtPoint(fixedImagePoint);
        support = params.kernel.getRegionSupport(covP);
    } else {
        support = params.kernel.getRegionSupport();
    }

    Vec2f indexPoint = params.cpImage.toLocal(fixedImagePoint);
    Vec2i imgLower = std::min(std::max(Vec2i(indexPoint - support/params.cpImage.scale() + 1.0), Vec2i(0)), params.cpImage.size() - 1);
    Vec2i imgUpper = std::min(std::max(Vec2i(indexPoint + support/params.cpImage.scale()      ), Vec2i(0)), params.cpImage.size() - 1);

    for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
        for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
            Vec2f point_i = params.cpImage.toGlobal(Vec2f(xi, yi));
            ScalarType k;
            if (UseWeightImage)
                k = params.kernel.evaluate(fixedImagePoint, point_i, sigmaP);
            else if (UseWeightTensor)
                k = params.kernel.evaluate(fixedImagePoint, point_i, covP);
            else
                k = params.kernel.evaluate(fixedImagePoint, point_i);

            movingImagePoint += params.cpwImage.at(Vec2i(xi, yi))*k*params.cpImage.at(Vec2i(xi, yi));
        }
    }

    ScalarType diff = 0.0;
    Vec2f movingImageGradient(0.0);
    int inside = 0;
    if (params.movingImage.insideGlobal(movingImagePoint)) {
        ScalarType movingImageValue;
        params.movingImage.derivsGlobal(movingImagePoint, movingImageValue, movingImageGradient);
        diff = movingImageValue - params.fixedImage.at(samplePos);
        inside = 1;
    }

    mse_params.diffs    [x + y*size[0]] = diff;
    params.gradients    [x + y*size[0]] = movingImageGradient;
    params.pixelsCounted[x + y*size[0]] = inside;
}

template<bool DoSubsample, bool UseWeightImage, bool UseWeightTensor>
__global__ void evaluateDerivativeMSE(GpuParams params, GpuMSEParams mse_params)
{
    Vec2i size = params.paramSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= size[0] || y >= size[1])
        return;

    Vec2f globalPoint = params.cpImage.toGlobal(Vec2f(static_cast<ScalarType>(x), static_cast<ScalarType>(y)));
    ScalarType w_i = params.cpwImage.at(Vec2i(x, y));

    ScalarType support;
    ScalarType sigmaP;
    Mat2f covP;
    if (UseWeightImage) {
        sigmaP  = params.kernel.getSigmaAtPoint(globalPoint);
        support = params.kernel.getRegionSupport(sigmaP);
    } else if (UseWeightTensor) {
        covP = params.kernel.getCovarianceAtPoint(globalPoint);
        support = params.kernel.getRegionSupport(covP);
    } else {
        support = params.kernel.getRegionSupport();
    }

    Vec2f indexPoint = params.fixedImage.toLocal(globalPoint);
    Vec2i imgLower, imgUpper;
    if (DoSubsample) {
        imgLower = std::min(std::max(Vec2i((indexPoint - support/params.fixedImage.scale())/float(params.subsample) + 1.0), Vec2i(0)), params.subsampledSize - 1);
        imgUpper = std::min(std::max(Vec2i((indexPoint + support/params.fixedImage.scale())/float(params.subsample)      ), Vec2i(0)), params.subsampledSize - 1);
    } else {
        imgLower = std::min(std::max(Vec2i(indexPoint - support/params.fixedImage.scale() + 1.0), Vec2i(0)), params.fixedImage.size() - 1);
        imgUpper = std::min(std::max(Vec2i(indexPoint + support/params.fixedImage.scale()      ), Vec2i(0)), params.fixedImage.size() - 1);
    }

    Vec2f derivative(0.0);
    for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
        for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
            int idx;
            if (DoSubsample)
                idx = xi + params.subsampledSize[0]*yi;
            else
                idx = params.fixedImage.toIndex(Vec2i(xi, yi));
            ScalarType diff = mse_params.diffs[idx];
            Vec2f gradient = params.gradients[idx];

            Vec2f point_i;
            if (DoSubsample)
                point_i = params.fixedImage.toGlobal(Vec2f(Vec2i(xi, yi)*params.subsample + params.gridShift[idx]));
            else
                point_i = params.fixedImage.toGlobal(Vec2f(Vec2i(xi, yi)));

            ScalarType k;
            if (UseWeightImage)
                k = params.kernel.evaluate(globalPoint, point_i, sigmaP);
            else if (UseWeightTensor)
                k = params.kernel.evaluate(globalPoint, point_i, covP);
            else
                k = params.kernel.evaluate(globalPoint, point_i);

            derivative += static_cast<ScalarType>(2.0)*diff*w_i*k*gradient;
        }
    }

    const int dimensionStride = params.cpImage.size().product();
    int pixelIndex = params.cpImage.toIndex(Vec2i(x, y));
    for (unsigned d = 0; d < SpaceDimensions; d++){
        params.derivatives[pixelIndex + d*dimensionStride] = derivative[d];
    }
}

#endif


template<bool DoSubsample>
void resolveValueMSE(dim3 gridDim, dim3 blockDim, GpuParams params, GpuMSEParams mse_params)
{
    if (params.kernel.useWeightImage())
        evaluateValueMSE<DoSubsample, true,  false><<<gridDim, blockDim>>>(params, mse_params);
    else if (params.kernel.useWeightTensor())
        evaluateValueMSE<DoSubsample, false, true> <<<gridDim, blockDim>>>(params, mse_params);
    else
        evaluateValueMSE<DoSubsample, false, false><<<gridDim, blockDim>>>(params, mse_params);
}

template<bool DoSubsample>
void resolveDerivativeMSE(dim3 gridDim, dim3 blockDim, GpuParams params, GpuMSEParams mse_params)
{
    if (params.kernel.useWeightImage())
        evaluateDerivativeMSE<DoSubsample, true,  false><<<gridDim, blockDim>>>(params, mse_params);
    else if (params.kernel.useWeightTensor())
        evaluateDerivativeMSE<DoSubsample, false, true> <<<gridDim, blockDim>>>(params, mse_params);
    else
        evaluateDerivativeMSE<DoSubsample, false, false><<<gridDim, blockDim>>>(params, mse_params);
}
