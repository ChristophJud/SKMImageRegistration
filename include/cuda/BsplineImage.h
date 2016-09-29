 /*
 * Copyright 2016 University of Basel, Medical Image Analysis Center
 *
 * Author: Benedikt Bitterli (benedikt.bitterli@unibas.ch)
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

#include "ImageBase.h"
#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <cstring>
#include <cmath>

template<typename Texel, int Dimensions>
class BsplineImage : public ImageBase<Dimensions>
{
    typedef Vec<Texel, Dimensions> TexelDeriv;
    typedef Vec<ScalarType, Dimensions> VecF;
    typedef Vec<int, Dimensions> VecI;

    const Texel *_coeffs;

    Texel initialAntiCausalCoefficient(Texel *c, int length, int stride, Texel z) const
    {
        return (z/(z*z - 1.0))*(z*c[(length - 2)*stride] + c[(length - 1)*stride]);
    }

    Texel initialCausalCoefficient(Texel *c, int length, int stride, Texel z) const
    {
        Texel zn = z;
        Texel iz = 1.0/z;
        Texel z2n = std::pow(z, (Texel)(length - 1));
        Texel sum = c[0] + z2n*c[(length - 1)*stride];
        z2n *= z2n*iz;
        for (int n = 1; n < length - 1; n++) {
            sum += (zn + z2n)*c[n*stride];
            zn *= z;
            z2n *= iz;
        }
        return sum/(1.0 - zn*zn);
    }

    void convertToInterpolationCoefficients(Texel *c, int length, int stride, Texel z) const
    {
        Texel lambda = 2.0 - z - 1.0/z;
        for (int n = 0; n < length; n++)
            c[n*stride] *= lambda;

        c[0] = initialCausalCoefficient(c, length, stride, z);
        for (int n = 1; n < length; n++)
            c[n*stride] += z*c[(n - 1)*stride];

        c[(length - 1)*stride] = initialAntiCausalCoefficient(c, length, stride, z);
        for (int n = length - 2; n >= 0; n--)
            c[n*stride] = z*(c[(n + 1)*stride] - c[n*stride]);
    }

    std::unique_ptr<Texel[]> mirrorCoeffBorders(const Texel *srcCopy, Vec2i origSize) const
    {
        int mirrorW = origSize[0] - 1;
        int mirrorH = origSize[1] - 1;

        std::unique_ptr<Texel[]> coeffs(new Texel[this->_size.product()]);
        for (int y = 0; y < this->_size[1]; ++y) {
            for (int x = 0; x < this->_size[0]; ++x) {
                int xi = mirrorW - std::abs(mirrorW - std::abs(x - 1));
                int yi = mirrorH - std::abs(mirrorH - std::abs(y - 1));
                coeffs[x + y*this->_size[0]] = srcCopy[xi + yi*origSize[0]];
            }
        }

        return std::move(coeffs);
    }

    std::unique_ptr<Texel[]> mirrorCoeffBorders(const Texel *srcCopy, Vec3i origSize) const
    {
        int mirrorW = origSize[0] - 1;
        int mirrorH = origSize[1] - 1;
        int mirrorD = origSize[2] - 1;

        std::unique_ptr<Texel[]> coeffs(new Texel[this->_size.product()]);
        for (int z = 0; z < this->_size[2]; ++z) {
            for (int y = 0; y < this->_size[1]; ++y) {
                for (int x = 0; x < this->_size[0]; ++x) {
                    int xi = mirrorW - std::abs(mirrorW - std::abs(x - 1));
                    int yi = mirrorH - std::abs(mirrorH - std::abs(y - 1));
                    int zi = mirrorD - std::abs(mirrorD - std::abs(z - 1));
                    coeffs[x + this->_size[0]*(y + z*this->_size[1])] = srcCopy[xi + origSize[0]*(yi + zi*origSize[1])];
                }
            }
        }

        return std::move(coeffs);
    }

    __host__ __device__ inline void reduceWithDeriv(Vec2i idx, Vec2f weights[4], Vec2f weightDerivs[4], Texel &result, TexelDeriv &resultDeriv) const
    {
        typedef Vec<Texel, Dimensions + 1> CombinedType;
        CombinedType combined(Texel(0.0));

        for (int y = 0; y < 4; ++y) {
            CombinedType wy(Texel(0.0));
            for (int x = 0; x < 4; ++x)
                wy += CombinedType(weightDerivs[x][0], weights[x][0], weights[x][0])*_coeffs[(idx[0] + x) + (idx[1] + y)*this->_strides[1]];
            combined += CombinedType(weights[y][1], weightDerivs[y][1], weights[y][1])*wy;
        }

        result = combined[2];
        resultDeriv = TexelDeriv(combined[0], combined[1])*this->_invScale;
    }

    __host__ __device__ inline void reduceWithDeriv(Vec3i idx, Vec3f weights[4], Vec3f weightDerivs[4], Texel &result, TexelDeriv &resultDeriv) const
    {
        typedef Vec<Texel, Dimensions + 1> CombinedType;
        CombinedType combined(Texel(0.0));

        for (int z = 0; z < 4; ++z) {
            CombinedType wz(Texel(0.0));
            for (int y = 0; y < 4; ++y) {
                CombinedType wy(Texel(0.0));
                for (int x = 0; x < 4; ++x)
                    wy += CombinedType(weightDerivs[x][0], weights[x][0], weights[x][0], weights[x][0])*_coeffs[(idx[0] + x) + (idx[1] + y)*this->_strides[1] + (idx[2] + z)*this->_strides[2]];
                wz += CombinedType(weights[y][1], weightDerivs[y][1], weights[y][1], weights[y][1])*wy;
            }
            combined += CombinedType(weights[z][2], weights[z][2], weightDerivs[z][2], weights[z][2])*wz;
        }

        result = combined[3];
        resultDeriv = TexelDeriv(combined[0], combined[1], combined[2])*this->_invScale;
    }

    __host__ __device__ inline Texel reduce(Vec2i idx, Vec2f weights[4]) const
    {
        Texel result(0.0);
        for (int y = 0; y < 4; ++y) {
            Texel wy(0.0);
            for (int x = 0; x < 4; ++x)
                wy += weights[x][0]*_coeffs[(idx[0] + x) + (idx[1] + y)*this->_strides[1]];
            result += weights[y][1]*wy;
        }
        return result;
    }

    __host__ __device__ inline Texel reduce(Vec3i idx, Vec3f weights[4]) const
    {
        Texel result(0.0);
        for (int z = 0; z < 4; ++z) {
            Texel wz(0.0);
            for (int y = 0; y < 4; ++y) {
                Texel wy(0.0);
                for (int x = 0; x < 4; ++x)
                    wy += weights[x][0]*_coeffs[(idx[0] + x) + (idx[1] + y)*this->_strides[1] + (idx[2] + z)*this->_strides[2]];
                wz += weights[y][1]*wy;
            }
            result += weights[z][2]*wz;
        }
        return result;
    }

    void computeCoefficients(Texel *srcCopy, Texel pole, Vec2i size) const
    {
        for (int y = 0; y < size[1]; y++)
            convertToInterpolationCoefficients(srcCopy + y*size[0], size[0],       1, pole);
        for (int x = 0; x < size[0]; x++)
            convertToInterpolationCoefficients(srcCopy + x,         size[1], size[0], pole);
    }

    void computeCoefficients(Texel *srcCopy, Texel pole, Vec3i size) const
    {
        for (int y = 0; y < size[1]; ++y)
            for (int x = 0; x < size[0]; x++)
                convertToInterpolationCoefficients(srcCopy + x + y*size[0], size[2], size[0]*size[1], pole);

        for (int z = 0; z < size[2]; ++z)
            for (int x = 0; x < size[0]; x++)
                convertToInterpolationCoefficients(srcCopy + x + z*size[0]*size[1], size[1], size[0], pole);

        for (int z = 0; z < size[2]; ++z)
            for (int y = 0; y < size[1]; ++y)
                convertToInterpolationCoefficients(srcCopy + size[0]*(y + z*size[1]), size[0], 1, pole);
    }

public:
    BsplineImage()
    : _coeffs(nullptr)
    {
    }

    BsplineImage(const Texel *coeffs, VecF offset, VecF scale, VecI size)
    : ImageBase<Dimensions>(offset, scale, size + 3),
      _coeffs(coeffs)
    {
    }

    void computeCoeffs(const Texel *src, std::unique_ptr<Texel[]> &coeffs)
    {
        VecI size = this->_size - 3;

        std::unique_ptr<Texel> srcCopy(new Texel[size.product()]);
        std::memcpy(srcCopy.get(), src, size.product()*sizeof(Texel));

        const Texel pole = std::sqrt(3.0) - 2.0;

        computeCoefficients(srcCopy.get(), pole, size);

        coeffs = mirrorCoeffBorders(srcCopy.get(), size);
    }

    void assignCoeffs(const Texel *coeffs)
    {
        _coeffs = coeffs;
    }

    const Texel *coeffs() const
    {
        return _coeffs;
    }

    __host__ __device__ inline Texel atLocal(VecF idx) const
    {
        VecI idxI = VecI(idx);
        VecF w = idx - VecF(idxI);

        VecF weights[4];
        weights[3] = (1.0/6.0)*w*w*w;
        weights[0] = (1.0/6.0) + 0.5*w*(w - 1.0) - weights[3];
        weights[2] = w + weights[0] - 2.0*weights[3];
        weights[1] = 1.0 - weights[0] - weights[2] - weights[3];

        return reduce(idxI, weights);
    }

    __host__ __device__ inline void derivsLocal(VecF idx, Texel &result, TexelDeriv &resultDeriv) const
    {
        VecI idxI = VecI(idx);
        VecF w = idx - VecF(idxI);

        VecF weights[4];
        weights[3] = static_cast<ScalarType>(1.0/6.0)*w*w*w;
        weights[0] = static_cast<ScalarType>(1.0/6.0) + static_cast<ScalarType>(0.5)*w*(w - static_cast<ScalarType>(1.0)) - weights[3];
        weights[2] = w + weights[0] - static_cast<ScalarType>(2.0)*weights[3];
        weights[1] = static_cast<ScalarType>(1.0) - weights[0] - weights[2] - weights[3];

        VecF weightDerivs[4];
        weightDerivs[3] = static_cast<ScalarType>(3.0/6.0)*w*w;
        weightDerivs[0] = w - static_cast<ScalarType>(0.5) - weightDerivs[3];
        weightDerivs[2] = static_cast<ScalarType>(1.0) + weightDerivs[0] - static_cast<ScalarType>(2.0)*weightDerivs[3];
        weightDerivs[1] = -weightDerivs[0] - weightDerivs[2] - weightDerivs[3];

        reduceWithDeriv(idxI, weights, weightDerivs, result, resultDeriv);
    }

    __host__ __device__ inline Texel atGlobal(VecF p) const
    {
        return atLocal(this->toLocal(p));
    }

    __host__ __device__ inline void derivsGlobal(VecF p, Texel &result, TexelDeriv &resultDeriv) const
    {
        derivsLocal(this->toLocal(p), result, resultDeriv);
    }

    __host__ __device__ inline bool inside(VecI idx) const
    {
        for (int i = 0; i < Dimensions; ++i)
            if (idx[i] < 0 || idx[i] >= this->_size[i] - 3)
                return false;
        return true;
    }

    __host__ __device__ inline bool insideLocal(VecF idx) const
    {
        for (int i = 0; i < Dimensions; ++i)
            if (idx[i] < -0.5f || idx[i] + 3.5f >= this->_size[i])
                return false;
        return true;
    }

    __host__ __device__ inline bool insideGlobal(VecF idx) const
    {
        return insideLocal(toLocal(idx));
    }
};
