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

#include "CommonConfig.h"
#include "ImageBase.h"
#include "Vec.h"
#include "Mat.h"

#include <cmath>

template<typename InternalType>
class GKernel
{
    typedef Vec<ScalarType, SpaceDimensions> VecFEx;
    typedef Mat<ScalarType, SpaceDimensions> MatFEx;
    typedef Vec<InternalType, SpaceDimensions> VecF;
    typedef Mat<InternalType, SpaceDimensions> MatF;
    typedef Vec<int, SpaceDimensions> VecI;

    bool _useWeightImage;
    bool _useWeightTensor;
    InternalType _sigmaC0;
    InternalType _sigmaC4;
    InternalType _maximumWeight;
    MatF _maximumEigenvalueCovariance;

    ImageNearest<ScalarType, SpaceDimensions> _wImage;
    ImageNearest<MatFEx, SpaceDimensions> _wTensor;

    __host__ __device__ inline InternalType evaluateWendlandStationaryC0(InternalType radius) const
    {
        InternalType r = radius/_sigmaC0;
        InternalType f = max(InternalType(0.0), InternalType(1.0) - r);
        return f*f;
    }

    __host__ __device__ inline InternalType evaluateWendlandNonStationaryC0(VecF p, VecF q, InternalType sigmaP, InternalType sigmaQ, InternalType F) const
    {
        InternalType sqrtQ = std::sqrt(InternalType(2.0)/(sigmaP + sigmaQ))*(p - q).length();
        InternalType r = sqrtQ/_sigmaC0;
        InternalType f = max(InternalType(0.0), InternalType(1.0) - r);
        return F*f*f;
    }

    __host__ __device__ inline InternalType evaluateWendlandAnisotropicC0(VecF p, VecF q, const MatF &covP, const MatF &covQ, InternalType F) const
    {
        VecF v = p - q;
        InternalType sqrt_Q = std::sqrt((v*invert((covP + covQ)*InternalType(0.5))).dot(v));
        InternalType r = sqrt_Q/_sigmaC0;
        InternalType f = max(InternalType(0.0), InternalType(1.0) - r);
        return F*f*f;
    }

    __host__ __device__ inline InternalType evaluateWendlandStationaryC4(InternalType radius) const
    {
        InternalType r = radius/_sigmaC4;
        InternalType f = max(InternalType(0.0), InternalType(1.0) - r);
        return f*f*f*f*f*f*(InternalType(3.0) + InternalType(18.0)*r + InternalType(35.0)*r*r)*InternalType(560.0/1680.0);
    }

    __host__ __device__ inline InternalType evaluateWendlandNonStationaryC4(VecF p, VecF q, InternalType sigmaP, InternalType sigmaQ, InternalType F) const
    {
        InternalType sqrtQ = std::sqrt(ScalarType(2.0)/(sigmaP + sigmaQ))*(p - q).length();
        InternalType r = sqrtQ/_sigmaC4;
        InternalType f = max(InternalType(0.0), InternalType(1.0) - r);
        return F*f*f*f*f*f*f*(InternalType(3.0) + InternalType(18.0)*r + InternalType(35.0)*r*r)*InternalType(560.0/1680.0);
    }

    __host__ __device__ inline InternalType evaluateWendlandAnisotropicC4(VecF p, VecF q, const MatF &covP, const MatF &covQ, InternalType F) const
    {
        VecF v = p - q;
        InternalType sqrt_Q = std::sqrt((v*invert((covP + covQ)*InternalType(0.5))).dot(v));
        InternalType r = sqrt_Q/_sigmaC4;
        InternalType f = max(InternalType(0.0), InternalType(1.0) - r);
        return F*f*f*f*f*f*f*(InternalType(3.0) + InternalType(18.0)*r + InternalType(35.0)*r*r)*InternalType(560.0/1680.0);
    }

    __host__ __device__ inline InternalType evaluateStationary(InternalType u) const
    {
#ifdef USE_WENDLAND_C0
        return evaluateWendlandStationaryC4(u)*evaluateWendlandStationaryC0(u);
#else
        return evaluateWendlandStationaryC4(u);
#endif
    }

    __host__ __device__ inline InternalType evaluateNonStationary(VecF p, VecF q, InternalType sigmaP, InternalType sigmaQ) const
    {
        InternalType F = std::pow(std::abs(sigmaP), InternalType(0.25))
                        *std::pow(std::abs(sigmaQ), InternalType(0.25))
                        *std::pow(std::abs((sigmaP+sigmaQ)/InternalType(2.0)), -InternalType(0.5));

#ifdef USE_WENDLAND_C0
        return evaluateWendlandNonStationaryC4(p, q, sigmaP, sigmaQ, F)
              *evaluateWendlandNonStationaryC0(p, q, sigmaP, sigmaQ, F);
#else
        return evaluateWendlandNonStationaryC4(p, q, sigmaP, sigmaQ, F);
#endif
    }

    __host__ __device__ inline InternalType evaluateAnisotropic(VecF p, VecF q, const MatF &covP, const MatF &covQ) const
    {
        InternalType F = std::pow(determinant(covP), InternalType(0.25))
                        *std::pow(determinant(covQ), InternalType(0.25))
                        *std::pow(determinant(((covP + covQ)*InternalType(0.5))), -InternalType(0.5));

#ifdef USE_WENDLAND_C0
        return evaluateWendlandAnisotropicC4(p, q, covP, covQ, F)
              *evaluateWendlandAnisotropicC0(p, q, covP, covQ, F);
#else
        return evaluateWendlandAnisotropicC4(p, q, covP, covQ, F);
#endif
    }

public:
    GKernel(ScalarType sigmaC0, ScalarType sigmaC4,
            bool useWeightImage,  ScalarType maximumWeight,               ImageNearest<ScalarType, SpaceDimensions> wImage,
            bool useWeightTensor, MatFEx     maximumEigenvalueCovariance, ImageNearest<MatFEx, SpaceDimensions> wTensor)
    : _useWeightImage(useWeightImage),
      _useWeightTensor(useWeightTensor),
      _sigmaC0(sigmaC0),
      _sigmaC4(sigmaC4),
      _maximumWeight(maximumWeight),
      _maximumEigenvalueCovariance(maximumEigenvalueCovariance),
      _wImage(wImage),
      _wTensor(wTensor)
    {
    }

    __host__ __device__ inline ScalarType getSigmaAtPoint(VecFEx fixedImagePoint) const
    {
        return _wImage.atGlobal(fixedImagePoint);
    }

    __host__ __device__ inline MatFEx getCovarianceAtPoint(VecFEx p) const
    {
        return _wTensor.atGlobal(p);
    }

    __host__ __device__ inline ScalarType getRegionSupport() const
    {
        return min(_sigmaC4, _sigmaC0);
    }

    __host__ __device__ inline ScalarType getRegionSupport(ScalarType sigmaP) const
    {
        return min(_sigmaC4, _sigmaC0)/std::sqrt(2.0/(_maximumWeight + sigmaP));
    }

    __host__ __device__ inline ScalarType getRegionSupport(const MatFEx &covP) const
    {
        return min(_sigmaC4, _sigmaC0)/std::sqrt(InternalType(2.0)/maxEigenValue((_maximumEigenvalueCovariance + covP)));
    }

    __host__ __device__ inline ScalarType evaluate(VecFEx p, VecFEx q) const
    {
        return evaluateStationary((VecF(p) - VecF(q)).length());
    }

    __host__ __device__ inline ScalarType evaluate(VecFEx p, VecFEx q, ScalarType sigmaP) const
    {
        return evaluateNonStationary(VecF(p), VecF(q), sigmaP, getSigmaAtPoint(q));
    }

    __host__ __device__ inline ScalarType evaluate(VecFEx p, VecFEx q, const MatFEx &covP) const
    {
        return evaluateAnisotropic(VecF(p), VecF(q), MatF(covP), MatF(getCovarianceAtPoint(q)));
    }

    __host__ __device__ bool useWeightImage() const
    {
        return _useWeightImage;
    }

    __host__ __device__ bool useWeightTensor() const
    {
        return _useWeightTensor;
    }

    ImageNearest<ScalarType, SpaceDimensions> &wImage()
    {
        return _wImage;
    }

    ImageNearest<MatFEx, SpaceDimensions> &wTensor()
    {
        return _wTensor;
    }
};

typedef GKernel<float> Kernel;
