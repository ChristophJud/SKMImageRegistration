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

#if SpaceDimensions == 3

template<bool UseWeightImage, bool UseWeightTensor, bool CalculateDerivative>
__global__ void regularizerRKHS(GpuParams params)
{
    Vec3i size = params.paramSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    if (x >= size[0] || y >= size[1] || z >= size[2])
        return;

    Vec3f globalPoint = params.cpImage.toGlobal(Vec3f(Vec3i(x, y, z)));
    ScalarType w_j = params.cpwImage.at(Vec3i(x, y, z));
    Vec3f c_j = params.cpImage.at(Vec3i(x, y, z));

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

    Vec3f indexPoint = Vec3f(Vec3i(x, y, z));
    Vec3i imgLower = std::min(std::max(Vec3i(indexPoint - support/params.cpImage.scale() + 1.0), Vec3i(0)), params.cpImage.size() - 1);
    Vec3i imgUpper = std::min(std::max(Vec3i(indexPoint + support/params.cpImage.scale()      ), Vec3i(0)), params.cpImage.size() - 1);

    ScalarType value = 0.0;
    Vec3f derivative(0.0);

    for (int zi = imgLower[2]; zi <= imgUpper[2]; ++zi) {
        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
                Vec3f point_i = params.cpImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)));
                Vec3f c_i = params.cpImage.at(Vec3i(xi, yi, zi));

                ScalarType k;
                if (UseWeightImage)
                    k = params.kernel.evaluate(globalPoint, point_i, sigmaP);
                else if (UseWeightTensor)
                    k = params.kernel.evaluate(globalPoint, point_i, covP);
                else
                    k = params.kernel.evaluate(globalPoint, point_i);

                value += (k*w_j*c_j).dot(c_i);

                if(CalculateDerivative)
                    derivative += static_cast<ScalarType>(2.0)*c_i*k*w_j;
            }
        }
    }

    // Only multiply derivative with weight, not the value.
    // The value will be scaled later in RegularizeImageToImageMetric
    if(CalculateDerivative)
        derivative *= params.regRKHS;

    const int dimensionStride = params.cpImage.size().product();
    int pixelIndex = params.cpImage.toIndex(Vec3i(x, y, z));
    params.bilateralRegularization[pixelIndex] = value;

    if(CalculateDerivative){
        for (unsigned d = 0; d < SpaceDimensions; d++)
            params.derivatives[pixelIndex + d*dimensionStride] += derivative[d];
    }
}

template<bool UseWeightImage, bool UseWeightTensor, bool CalculateDerivative>
__global__ void regularizerRD(GpuParams params)
{
    Vec3i size = params.paramSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    if (x >= size[0] || y >= size[1] || z >= size[2])
        return;

    Vec3f globalPoint = params.cpImage.toGlobal(Vec3f(Vec3i(x, y, z)));
    ScalarType w_j = params.cpwImage.at(Vec3i(x, y, z));
    Vec3f c_j = params.cpImage.at(Vec3i(x, y, z));

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

    Vec3f indexPoint = Vec3f(Vec3i(x, y, z));
    Vec3i imgLower = std::min(std::max(Vec3i(indexPoint - support/params.cpImage.scale() + 1.0), Vec3i(0)), params.cpImage.size() - 1);
    Vec3i imgUpper = std::min(std::max(Vec3i(indexPoint + support/params.cpImage.scale()      ), Vec3i(0)), params.cpImage.size() - 1);

    ScalarType value = 0.0;
    Vec3f derivative(0.0);

    for (int zi = imgLower[2]; zi <= imgUpper[2]; ++zi) {
        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
                Vec3f point_i = params.cpImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)));
                Vec3f c_i = params.cpImage.at(Vec3i(xi, yi, zi));

                ScalarType k;
                if (UseWeightImage)
                    k = params.kernel.evaluate(globalPoint, point_i, sigmaP);
                else if (UseWeightTensor)
                    k = params.kernel.evaluate(globalPoint, point_i, covP);
                else
                    k = params.kernel.evaluate(globalPoint, point_i);

                value += (c_j - c_i).lengthSq()*k*w_j;

                if(CalculateDerivative)
                    derivative += static_cast<ScalarType>(4.0)*(c_j - c_i)*k*w_j;
            }
        }
    }

    // Only multiply derivative with weight, not the value.
    // The value will be scaled later in RegularizedImageToImageMetric
    if(CalculateDerivative)
        derivative *= params.regRD;

    const int dimensionStride = params.cpImage.size().product();
    int pixelIndex = params.cpImage.toIndex(Vec3i(x, y, z));
    params.bilateralRegularization[pixelIndex] = value;

    if(CalculateDerivative){
        for (unsigned d = 0; d < SpaceDimensions; d++)
            params.derivatives[pixelIndex + d*dimensionStride] += derivative[d];
    }
}

template<bool UseWeightImage, bool UseWeightTensor, bool CalculateDerivative>
__global__ void regularizerPG(GpuParams params)
{
    Vec3i size = params.paramSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    if (x >= size[0] || y >= size[1] || z >= size[2])
        return;

    Vec3f globalPoint = params.cpImage.toGlobal(Vec3f(Vec3i(x, y, z)));
    ScalarType w_j = params.cpwImage.at(Vec3i(x, y, z));
    Vec3f c_j = params.cpImage.at(Vec3i(x, y, z));

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

    Vec3f indexPoint = Vec3f(Vec3i(x, y, z));
    Vec3i imgLower = std::min(std::max(Vec3i(indexPoint - support/params.cpImage.scale() + 1.0), Vec3i(0)), params.cpImage.size() - 1);
    Vec3i imgUpper = std::min(std::max(Vec3i(indexPoint + support/params.cpImage.scale()      ), Vec3i(0)), params.cpImage.size() - 1);

    ScalarType value = 0.0;
    Vec3f derivative(0.0);
    ScalarType scale = params.regPGScaling;
    for (int zi = imgLower[2]; zi <= imgUpper[2]; ++zi) {
        for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
            for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
                Vec3f point_i = params.cpImage.toGlobal(Vec3f(Vec3i(xi, yi, zi)));
                Vec3f c_i = params.cpImage.at(Vec3i(xi, yi, zi));

                ScalarType k;
                if (UseWeightImage)
                    k = params.kernel.evaluate(globalPoint, point_i, sigmaP);
                else if (UseWeightTensor)
                    k = params.kernel.evaluate(globalPoint, point_i, covP);
                else
                    k = params.kernel.evaluate(globalPoint, point_i);

                Vec3f cross_product(0.0);
                cross_product[0] = c_i[1]*c_j[2] - c_i[2]*c_j[1];
                cross_product[1] = c_i[2]*c_j[0] - c_i[0]*c_j[2];
                cross_product[2] = c_i[0]*c_j[1] - c_i[1]*c_j[0];
                ScalarType cp_norm2 = cross_product[0]*cross_product[0]+cross_product[1]*cross_product[1]+cross_product[2]*cross_product[2];
                value += k*w_j * scale*scale/2.0 * std::log(1.0+cp_norm2/scale);

                if(CalculateDerivative){
                    Vec3f dev(0.0);
                    ScalarType dp = 0;
                    dp = 2.0*c_i[2]*(c_i[2]*c_j[0]-c_i[0]*c_j[2]) - 2.0*c_i[1]*(c_i[0]*c_j[1]-c_i[1]*c_j[0]);
                    dev[0] = scale*dp/(2.0*(cp_norm2/scale+1));
                    dp = 2.0*c_i[0]*(c_i[0]*c_j[1]-c_i[1]*c_j[0]) - 2.0*c_i[2]*(c_i[1]*c_j[2]-c_i[2]*c_j[1]);
                    dev[1] = scale*dp/(2.0*(cp_norm2/scale+1));
                    dp = 2.0*c_i[1]*(c_i[1]*c_j[2]-c_i[2]*c_j[1]) - 2.0*c_i[0]*(c_i[2]*c_j[0]-c_i[0]*c_j[2]);
                    dev[2] = scale*dp/(2.0*(cp_norm2/scale+1));
                    derivative += 2.0 * k*w_j * dev;
                }
            }
        }
    }

    // Only multiply derivative with weight, not the value.
    // The value will be scaled later in RegularizedImageToImageMetric
    if(CalculateDerivative)
        derivative *= params.regPG;

    const int dimensionStride = params.cpImage.size().product();
    int pixelIndex = params.cpImage.toIndex(Vec3i(x, y, z));
    params.bilateralRegularization[pixelIndex] = value;

    if(CalculateDerivative){
        for (unsigned d = 0; d < SpaceDimensions; d++)
            params.derivatives[pixelIndex + d*dimensionStride] += derivative[d];
    }
}

#else ///////////// 2D ////////////

template<bool UseWeightImage, bool UseWeightTensor, bool CalculateDerivative>
__global__ void regularizerRKHS(GpuParams params)
{
    Vec2i size = params.paramSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= size[0] || y >= size[1])
        return;

    Vec2f globalPoint = params.cpImage.toGlobal(Vec2f(static_cast<ScalarType>(x), static_cast<ScalarType>(y)));
    ScalarType w_j = params.cpwImage.at(Vec2i(x, y));
    Vec2f c_j = params.cpImage.at(Vec2i(x, y));

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

    Vec2f indexPoint = Vec2f(static_cast<ScalarType>(x), static_cast<ScalarType>(y));
    Vec2i imgLower = std::min(std::max(Vec2i(indexPoint - support/params.cpImage.scale() + 1.0), Vec2i(0)), params.cpImage.size() - 1);
    Vec2i imgUpper = std::min(std::max(Vec2i(indexPoint + support/params.cpImage.scale()      ), Vec2i(0)), params.cpImage.size() - 1);

    ScalarType value = 0.0;
    Vec2f derivative(0.0);

    for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
        for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
            Vec2f point_i = params.cpImage.toGlobal(Vec2f(xi, yi));
            Vec2f c_i = params.cpImage.at(Vec2i(xi, yi));

            ScalarType k;
            if (UseWeightImage)
                k = params.kernel.evaluate(globalPoint, point_i, sigmaP);
            else if (UseWeightTensor)
                k = params.kernel.evaluate(globalPoint, point_i, covP);
            else
                k = params.kernel.evaluate(globalPoint, point_i);

            value += (k*w_j*c_j).dot(c_i);

            if(CalculateDerivative)
                derivative += static_cast<ScalarType>(2.0)*c_i*k*w_j;
        }
    }

    // Only multiply derivative with weight, not the value.
    // The value will be scaled later in RegularizedImageToImageMetric
    if(CalculateDerivative)
        derivative *= params.regRKHS;

    const int dimensionStride = params.cpImage.size().product();
    int pixelIndex = params.cpImage.toIndex(Vec2i(x, y));
    params.bilateralRegularization[pixelIndex] = value;

    if(CalculateDerivative){
        for (unsigned d = 0; d < SpaceDimensions; d++)
            params.derivatives[pixelIndex + d*dimensionStride] += derivative[d];
    }
}

template<bool UseWeightImage, bool UseWeightTensor, bool CalculateDerivative>
__global__ void regularizerRD(GpuParams params)
{
    Vec2i size = params.paramSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= size[0] || y >= size[1])
        return;

    Vec2f globalPoint = params.cpImage.toGlobal(Vec2f(static_cast<ScalarType>(x), static_cast<ScalarType>(y)));
    ScalarType w_j = params.cpwImage.at(Vec2i(x, y));
    Vec2f c_j = params.cpImage.at(Vec2i(x, y));

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

    Vec2f indexPoint = Vec2f(static_cast<ScalarType>(x), static_cast<ScalarType>(y));
    Vec2i imgLower = std::min(std::max(Vec2i(indexPoint - support/params.cpImage.scale() + 1.0), Vec2i(0)), params.cpImage.size() - 1);
    Vec2i imgUpper = std::min(std::max(Vec2i(indexPoint + support/params.cpImage.scale()      ), Vec2i(0)), params.cpImage.size() - 1);

    ScalarType value = 0.0;
    Vec2f derivative(0.0);

    for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
        for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
            Vec2f point_i = params.cpImage.toGlobal(Vec2f(xi, yi));
            Vec2f c_i = params.cpImage.at(Vec2i(xi, yi));

            ScalarType k;
            if (UseWeightImage)
                k = params.kernel.evaluate(globalPoint, point_i, sigmaP);
            else if (UseWeightTensor)
                k = params.kernel.evaluate(globalPoint, point_i, covP);
            else
                k = params.kernel.evaluate(globalPoint, point_i);

            value += (c_j - c_i).lengthSq()*k*w_j;

            if(CalculateDerivative)
                derivative += static_cast<ScalarType>(4.0)*(c_j - c_i)*k*w_j;
        }
    }

    // Only multiply derivative with weight, not the value.
    // The value will be scaled later in RegularizedImageToImageMetric
    if(CalculateDerivative)
        derivative *= params.regRD;

    const int dimensionStride = params.cpImage.size().product();
    int pixelIndex = params.cpImage.toIndex(Vec2i(x, y));
    params.bilateralRegularization[pixelIndex] = value;

    if(CalculateDerivative){
        for (unsigned d = 0; d < SpaceDimensions; d++)
            params.derivatives[pixelIndex + d*dimensionStride] += derivative[d];
    }
}

template<bool UseWeightImage, bool UseWeightTensor, bool CalculateDerivative>
__global__ void regularizerPG(GpuParams params)
{
    Vec2i size = params.paramSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= size[0] || y >= size[1])
        return;

    Vec2f globalPoint = params.cpImage.toGlobal(Vec2f(static_cast<ScalarType>(x), static_cast<ScalarType>(y)));
    ScalarType w_j = params.cpwImage.at(Vec2i(x, y));
    Vec2f c_j = params.cpImage.at(Vec2i(x, y));

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

    Vec2f indexPoint = Vec2f(static_cast<ScalarType>(x), static_cast<ScalarType>(y));
    Vec2i imgLower = std::min(std::max(Vec2i(indexPoint - support/params.cpImage.scale() + 1.0), Vec2i(0)), params.cpImage.size() - 1);
    Vec2i imgUpper = std::min(std::max(Vec2i(indexPoint + support/params.cpImage.scale()      ), Vec2i(0)), params.cpImage.size() - 1);

    ScalarType value = 0.0;
    Vec2f derivative(0.0);
    ScalarType scale = params.regPGScaling;
    ScalarType eps = 1e-10;
    for (int yi = imgLower[1]; yi <= imgUpper[1]; ++yi) {
        for (int xi = imgLower[0]; xi <= imgUpper[0]; ++xi) {
            Vec2f point_i = params.cpImage.toGlobal(Vec2f(xi, yi));
            Vec2f c_i = params.cpImage.at(Vec2i(xi, yi));

            ScalarType k;
            if (UseWeightImage)
                k = params.kernel.evaluate(globalPoint, point_i, sigmaP);
            else if (UseWeightTensor)
                k = params.kernel.evaluate(globalPoint, point_i, covP);
            else
                k = params.kernel.evaluate(globalPoint, point_i);

            // get the parallelogram regularizer working in 2d
            ScalarType val = c_i[0]*c_j[1] - c_i[1]*c_j[0];
            ScalarType val2 = val*val;
            value += k*w_j * scale*scale/2.0 * std::log(1.0+val2/scale);

            if(CalculateDerivative){
                Vec2f dev(0.0);
                ScalarType factor = val2/scale + 1;
                dev[0] = -scale*c_i[1]*val / factor;
                dev[1] =  scale*c_i[0]*val / factor;
                derivative += 2.0*k*w_j * dev;
            }
        }
    }

    // Only multiply derivative with weight, not the value.
    // The value will be scaled later in RegularizedImageToImageMetric
    if(CalculateDerivative)
        derivative *= params.regPG;

    const int dimensionStride = params.cpImage.size().product();
    int pixelIndex = params.cpImage.toIndex(Vec2i(x, y));
    params.bilateralRegularization[pixelIndex] = value;

    if(CalculateDerivative){
        for (unsigned d = 0; d < SpaceDimensions; d++)
            params.derivatives[pixelIndex + d*dimensionStride] += derivative[d];
    }
}

#endif

