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

#include <cuda_runtime.h>
#include <ostream>

template<typename Type, int Dimensions>
class Vec
{
    Type _data[Dimensions];

public:
    __host__ __device__ Vec() = default;

    __host__ __device__ inline explicit Vec(Type a)
    {
        for (unsigned i = 0; i < Dimensions; ++i)
            _data[i] = a;
    }

    template<typename... Ts>
    __host__ __device__ inline Vec(Type a, Type b, Ts... ts)
    : _data{a, b, ts...}
    {
    }

    template<typename OtherType>
    __host__ __device__ inline explicit Vec(const Vec<OtherType, Dimensions> &other)
    {
        for (unsigned i = 0; i < Dimensions; ++i)
            _data[i] = Type(other[i]);
    }

    __host__ __device__ inline Type  operator[](int i) const { return _data[i]; }
    __host__ __device__ inline Type &operator[](int i)       { return _data[i]; }

    __host__ __device__ inline Type max() const
    {
        Type result(_data[0]);
        for (int i = 1; i < Dimensions; ++i)
            if (_data[i] > result)
                result = _data[i];
        return result;
    }

    __host__ __device__ inline Type dot(const Vec &o) const
    {
        Type result = _data[0]*o[0];
        for (int i = 1; i < Dimensions; ++i)
            result += _data[i]*o[i];
        return result;
    }

    __host__ __device__ inline Type product() const
    {
        Type result = _data[0];
        for (int i = 1; i < Dimensions; ++i)
            result *= _data[i];
        return result;
    }

    __host__ __device__ inline Type sum() const
    {
        Type result = _data[0];
        for (int i = 1; i < Dimensions; ++i)
            result += _data[i];
        return result;
    }

    __host__ __device__ inline Type lengthSq() const
    {
        Type result = _data[0]*_data[0];
        for (int i = 1; i < Dimensions; ++i)
            result += _data[i]*_data[i];
        return result;
    }

    __host__ __device__ inline Type length() const
    {
        return std::sqrt(lengthSq());
    }

    __host__ __device__ inline Vec operator-() const
    {
        Vec result;
        for (int i = 0; i < Dimensions; ++i)
            result[i] = -_data[i];
        return result;
    }

    __host__ __device__ inline Vec operator+(Type o) const
    {
        Vec result;
        for (int i = 0; i < Dimensions; ++i)
            result[i] = _data[i] + o;
        return result;
    }

    __host__ __device__ inline Vec operator-(Type o) const
    {
        Vec result;
        for (int i = 0; i < Dimensions; ++i)
            result[i] = _data[i] - o;
        return result;
    }

    __host__ __device__ inline Vec operator*(Type o) const
    {
        Vec result;
        for (int i = 0; i < Dimensions; ++i)
            result[i] = _data[i]*o;
        return result;
    }

    __host__ __device__ inline Vec operator/(Type o) const
    {
        Vec result;
        for (int i = 0; i < Dimensions; ++i)
            result[i] = _data[i]/o;
        return result;
    }

    __host__ __device__ inline Vec operator+(const Vec &o) const
    {
        Vec result;
        for (int i = 0; i < Dimensions; ++i)
            result[i] = _data[i] + o[i];
        return result;
    }

    __host__ __device__ inline Vec operator-(const Vec &o) const
    {
        Vec result;
        for (int i = 0; i < Dimensions; ++i)
            result[i] = _data[i] - o[i];
        return result;
    }

    __host__ __device__ inline Vec operator*(const Vec &o) const
    {
        Vec result;
        for (int i = 0; i < Dimensions; ++i)
            result[i] = _data[i]*o[i];
        return result;
    }

    __host__ __device__ inline Vec operator/(const Vec &o) const
    {
        Vec result;
        for (int i = 0; i < Dimensions; ++i)
            result[i] = _data[i]/o[i];
        return result;
    }

    __host__ __device__ inline Vec &operator+=(Type o)
    {
        for (int i = 0; i < Dimensions; ++i)
            _data[i] += o;
        return *this;
    }

    __host__ __device__ inline Vec &operator-=(Type o)
    {
        for (int i = 0; i < Dimensions; ++i)
            _data[i] -= o;
        return *this;
    }

    __host__ __device__ inline Vec &operator*=(Type o)
    {
        for (int i = 0; i < Dimensions; ++i)
            _data[i] *= o;
        return *this;
    }

    __host__ __device__ inline Vec &operator/=(Type o)
    {
        for (int i = 0; i < Dimensions; ++i)
            _data[i] /= o;
        return *this;
    }

    __host__ __device__ inline Vec &operator+=(const Vec &o)
    {
        for (int i = 0; i < Dimensions; ++i)
            _data[i] += o[i];
        return *this;
    }

    __host__ __device__ inline Vec &operator-=(const Vec &o)
    {
        for (int i = 0; i < Dimensions; ++i)
            _data[i] -= o[i];
        return *this;
    }

    __host__ __device__ inline Vec &operator*=(const Vec &o)
    {
        for (int i = 0; i < Dimensions; ++i)
            _data[i] *= o[i];
        return *this;
    }

    __host__ __device__ inline Vec &operator/=(const Vec &o)
    {
        for (int i = 0; i < Dimensions; ++i)
            _data[i] /= o[i];
        return *this;
    }

    friend std::ostream &operator<< (std::ostream &stream, const Vec &v) {
        stream << '(';
        for (int i = 0; i < Dimensions; ++i)
            stream << v[i] << (i == Dimensions - 1 ? ')' : ',');
        return stream;
    }
};

template<typename Type, int Dimensions>
__host__ __device__ inline Vec<Type, Dimensions> operator+(Type s, const Vec<Type, Dimensions> &v)
{
    Vec<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = s + v[i];
    return result;
}

template<typename Type, int Dimensions>
__host__ __device__ inline Vec<Type, Dimensions> operator-(Type s, const Vec<Type, Dimensions> &v)
{
    Vec<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = s - v[i];
    return result;
}

template<typename Type, int Dimensions>
__host__ __device__ inline Vec<Type, Dimensions> operator*(Type s, const Vec<Type, Dimensions> &v)
{
    Vec<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = s*v[i];
    return result;
}

template<typename Type, int Dimensions>
__host__ __device__ inline Vec<Type, Dimensions> operator/(Type s, const Vec<Type, Dimensions> &v)
{
    Vec<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = s/v[i];
    return result;
}

typedef Vec<ScalarType, 2> Vec2f;
typedef Vec<ScalarType, 3> Vec3f;
typedef Vec<int, 2> Vec2i;
typedef Vec<int, 3> Vec3i;

template<typename T>
__host__ __device__ inline T max(T a, T b)
{
    return a > b ? a : b;
}
template<typename T>
__host__ __device__ inline T min(T a, T b)
{
    return a < b ? a : b;
}

namespace std {

template<typename Type, int Dimensions>
__host__ __device__ inline Vec<Type, Dimensions> abs(const Vec<Type, Dimensions> &v)
{
    Vec<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = std::abs(v[i]);
    return result;
}

template<typename Type, int Dimensions>
__host__ __device__ inline Vec<Type, Dimensions> max(const Vec<Type, Dimensions> &a, const Vec<Type, Dimensions> &b)
{
    Vec<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = ::max(a[i], b[i]);
    return result;
}

template<typename Type, int Dimensions>
__host__ __device__ inline Vec<Type, Dimensions> min(const Vec<Type, Dimensions> &a, const Vec<Type, Dimensions> &b)
{
    Vec<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = ::min(a[i], b[i]);
    return result;
}

}
