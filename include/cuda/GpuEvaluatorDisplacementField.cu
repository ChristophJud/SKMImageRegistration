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

template<bool UseWeightImage, bool UseWeightTensor>
__global__ void evaluateDisplacement(GpuParams params)
{
    Vec3i size = params.subsampledSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int z = blockDim.z*blockIdx.z + threadIdx.z;
    if (x >= size[0] || y >= size[1] || z >= size[2])
        return;

    Vec3f fixedImagePoint = params.fixedImage.toGlobal(Vec3f(Vec3i(x, y, z)));
    Vec3f displacement(0.0);
    if (params.useDisplacementField)
        displacement = params.displacementField.atGlobal(fixedImagePoint);

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

                displacement += params.cpwImage.at(Vec3i(xi, yi, zi))*k*params.cpImage.at(Vec3i(xi, yi, zi));
            }
        }
    }

    params.displacements[x + y*size[0] + z*size[0]*size[1]] = displacement;
}

#else

template<bool UseWeightImage, bool UseWeightTensor>
__global__ void evaluateDisplacement(GpuParams params)
{
    Vec2i size = params.subsampledSize;

    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    if (x >= size[0] || y >= size[1])
        return;

    Vec2f  fixedImagePoint = params.fixedImage.toGlobal(Vec2f(Vec2i(x, y)));
    Vec2f displacement(0.0);
    if (params.useDisplacementField)
        displacement = params.displacementField.atGlobal(fixedImagePoint);

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

            displacement += params.cpwImage.at(Vec2i(xi, yi))*k*params.cpImage.at(Vec2i(xi, yi));
        }
    }

    params.displacements[x + y*size[0]] = displacement;
}

#endif


