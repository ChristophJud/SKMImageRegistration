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

#include "Vec.h"

#include <cuda_runtime.h>

template<int Dimensions>
class ImageBase
{
    typedef Vec<ScalarType, Dimensions> VecF;
    typedef Vec<int, Dimensions> VecI;

protected:
    VecF _offset;
    VecF _scale, _invScale;

    VecI _size;
    VecI _strides;

public:
    ImageBase() = default;
    ImageBase(VecF offset, VecF scale, VecI size)
    : _offset(offset),
      _scale(scale),
      _invScale(ScalarType(1.0)/_scale),
      _size(size)
    {
        _strides[0] = 1;
        for (int i = 1; i < Dimensions; ++i)
            _strides[i] = _strides[i - 1]*_size[i - 1];
    }

    __host__ __device__ inline VecF toGlobal(VecF idx) const
    {
        return _offset + _scale*idx;
    }

    __host__ __device__ inline VecF toLocal(VecF point) const
    {
        return _invScale*(point - _offset);
    }

    __host__ __device__ inline VecI toNearest(VecF idx) const
    {
        return VecI(idx + ScalarType(0.5));
    }

    __host__ __device__ inline int toIndex(VecI idx) const
    {
        return _strides.dot(idx);
    }

    __host__ __device__ inline VecF offset() const
    {
        return _offset;
    }

    __host__ __device__ inline VecF scale() const
    {
        return _scale;
    }

    __host__ __device__ inline VecI size() const
    {
        return _size;
    }

    __host__ __device__ inline bool inside(VecI idx) const
    {
        for (int i = 0; i < Dimensions; ++i)
            if (idx[i] < 0 || idx[i] >= _size[i])
                return false;
        return true;
    }

    __host__ __device__ inline bool insideLocal(VecF idx) const
    {
        for (int i = 0; i < Dimensions; ++i)
            if (idx[i] < -0.5f || idx[i] + 0.5f >= _size[i])
                return false;
        return true;
    }

    __host__ __device__ inline bool insideGlobal(VecF idx) const
    {
        return insideLocal(toLocal(idx));
    }
};
