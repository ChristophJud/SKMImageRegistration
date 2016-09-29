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
#include "Mat.h"

#include <memory>

template<typename Texel, int Dimensions>
class ImageNearest : public ImageBase<Dimensions>
{
    typedef Vec<ScalarType, Dimensions> VecF;
    typedef Vec<int, Dimensions> VecI;

    const Texel *_data;

    template<typename T>
    __host__ __device__ inline T defaultValue(T) const
    {
        return T(0.0);
    }

    template<typename T, int Size>
    __host__ __device__ inline Mat<T, Size> defaultValue(Mat<T, Size>) const
    {
        return Mat<T, Size>::identity();
    }

public:
    ImageNearest()
    : _data(nullptr)
    {
    }

    ImageNearest(const Texel *data, VecF offset, VecF scale, VecI size)
    : ImageBase<Dimensions>(offset, scale, size),
      _data(data)
    {
    }

    void assignData(const Texel *data)
    {
        _data = data;
    }

    const Texel *data() const
    {
        return _data;
    }

    __host__ __device__ inline Texel at(VecI idx) const
    {
        return _data[this->toIndex(idx)];
    }

    __host__ __device__ inline Texel atLocal(VecF idx) const
    {
        VecI idxI = this->toNearest(idx);
        return this->inside(idxI) ? _data[this->toIndex(idxI)] : defaultValue(Texel());
    }

    __host__ __device__ inline Texel atGlobal(VecF idx) const
    {
        VecI idxI = this->toNearest(this->toLocal(idx));
        return this->inside(idxI) ? _data[this->toIndex(idxI)] : defaultValue(Texel());
    }
};
