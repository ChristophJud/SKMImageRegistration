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

template<typename Type, int Size>
class Mat
{
    typedef Vec<Type, Size> SVec;

    Type _data[Size*Size];

public:
    __host__ __device__ Mat() = default;

    __host__ __device__ Mat(const Type *data)
    {
        for (int i = 0; i < Size*Size; ++i)
            _data[i] = data[i];
    }

    __host__ __device__ Mat(SVec v)
    {
        for (int i = 0; i < Size; ++i)
            for (int j = 0; j < Size; ++j)
                _data[i*Size + j] = (i == j) ? v[i] : Type(0);
    }

    template<typename OtherType>
    __host__ __device__ inline Mat(const Mat<OtherType, Size> &o)
    {
        for (int i = 0; i < Size*Size; ++i)
            _data[i] = Type(o[i]);
    }

    template<typename... Ts>
    __host__ __device__ inline Mat(Type a, Ts... ts)
    : _data{a, ts...}
    {
    }

    __host__ __device__ inline Type  operator[](int i) const { return _data[i]; }
    __host__ __device__ inline Type &operator[](int i)       { return _data[i]; }

    __host__ __device__ inline Type  operator()(int i, int j) const { return _data[i*Size + j]; }
    __host__ __device__ inline Type &operator()(int i, int j)       { return _data[i*Size + j]; }

    __host__ __device__ inline const Type *data() const { return _data; };
    __host__ __device__ inline       Type *data()       { return _data; };

    __host__ __device__ inline Mat operator-() const
    {
        Mat result;
        for (int i = 0; i < Size*Size; ++i)
            result[i] = -_data[i];
        return result;
    }

    __host__ __device__ inline Mat operator-(Type o) const
    {
        Mat result(*this);
        for (int i = 0; i < Size; ++i)
            result[i*Size + i] -= o;
        return result;
    }

    __host__ __device__ inline Mat operator+(Type o) const
    {
        Mat result(*this);
        for (int i = 0; i < Size; ++i)
            result[i*Size + i] += o;
        return result;
    }

    __host__ __device__ inline Mat operator*(Type o) const
    {
        Mat result;
        for (int i = 0; i < Size*Size; ++i)
            result[i] = _data[i]*o;
        return result;
    }

    __host__ __device__ inline Mat operator/(Type o) const
    {
        Mat result;
        for (int i = 0; i < Size*Size; ++i)
            result[i] = _data[i]/o;
        return result;
    }

    __host__ __device__ inline Mat operator+(const Mat &o) const
    {
        Mat result;
        for (int i = 0; i < Size*Size; ++i)
            result[i] = _data[i] + o[i];
        return result;
    }

    __host__ __device__ inline Mat operator-(const Mat &o) const
    {
        Mat result;
        for (int i = 0; i < Size*Size; ++i)
            result[i] = _data[i] - o[i];
        return result;
    }

    __host__ __device__ inline Mat operator*(const Mat &o) const
    {
        Mat result;
        for (int i = 0; i < Size; ++i) {
            for (int j = 0; j < Size; ++j) {
                Type dot(0);
                for (int k = 0; k < Size; ++k)
                    dot += _data[i*Size + k]*o[k*Size + j];
                result[i*Size + j] = dot;
            }
        }
        return result;
    }

    __host__ __device__ inline SVec operator*(const SVec &o) const
    {
        SVec result;
        for (int i = 0; i < Size; ++i) {
            Type dot(0);
            for (int k = 0; k < Size; ++k)
                dot += _data[i*Size + k]*o[k];
            result[i] = dot;
        }
        return result;
    }

    __host__ __device__ inline Mat &operator*=(Type o)
    {
        for (int i = 0; i < Size*Size; ++i)
            _data[i] *= o;
        return *this;
    }

    __host__ __device__ inline Mat &operator/=(Type o)
    {
        for (int i = 0; i < Size*Size; ++i)
            _data[i] /= o;
        return *this;
    }

    __host__ __device__ inline Mat &operator+=(const Mat &o)
    {
        for (int i = 0; i < Size*Size; ++i)
            _data[i] += o[i];
        return *this;
    }

    __host__ __device__ inline Mat &operator-=(const Mat &o)
    {
        for (int i = 0; i < Size*Size; ++i)
            _data[i] -= o[i];
        return *this;
    }

    __host__ __device__ inline Mat &operator*=(const Mat &o)
    {
        *this = *this*o;
        return *this;
    }

    friend std::ostream &operator<< (std::ostream &stream, const Mat &m) {
        for (int i = 0; i < Size; ++i) {
            stream << '[';
            for (int j = 0; j < Size; ++j)
                stream << m[i*Size + j] << (j == Size - 1 ? ']' : ',');
            if (i < Size - 1)
                std::cout << "\n";
        }
        return stream;
    }

    __host__ __device__ static Mat identity()
    {
        Mat result;
        for (int i = 0; i < Size; ++i)
            for (int j = 0; j < Size; ++j)
                result(i, j) = i == j ? Type(1) : Type(0);
        return result;
    }
};

template<typename Type, int Size>
__host__ __device__ Vec<Type, Size> operator*(const Vec<Type, Size> &v, const Mat<Type, Size> &m)
{
    Vec<Type, Size> result;
    for (int i = 0; i < Size; ++i) {
        Type dot(0);
        for (int k = 0; k < Size; ++k)
            dot += v[k]*m[k*Size + i];
        result[i] = dot;
    }
    return result;
}

typedef Mat<ScalarType, 2> Mat2f;
typedef Mat<ScalarType, 3> Mat3f;

template<typename Type>
__host__ __device__ Type sqr(Type a)
{
    return a*a;
}

template<typename Type>
__host__ __device__ Type determinant(const Mat<Type, 2> &m)
{
    return m(0, 0)*m(1, 1) - m(1, 0)*m(0, 1);
}

template<typename Type>
__host__ __device__ Type determinant(const Mat<Type, 3> &m)
{
    return m(0, 0)*m(1, 1)*m(2, 2) + m(1, 0)*m(2, 1)*m(0, 2) + m(2, 0)*m(0, 1)*m(1, 2)
          -m(0, 0)*m(2, 1)*m(1, 2) - m(2, 0)*m(1, 1)*m(0, 2) - m(1, 0)*m(0, 1)*m(2, 2);
}

template<typename Type>
__host__ __device__ Mat<Type, 2> invert(const Mat<Type, 2> &m)
{
    return Mat<Type, 2>(
         m(1, 1), -m(0, 1),
        -m(1, 0),  m(0, 0)
    )/determinant(m);
}

template<typename Type>
__host__ __device__ Mat<Type, 3> invert(const Mat<Type, 3> &m)
{
    return Mat<Type, 3>(
        m(1, 1)*m(2, 2) - m(1, 2)*m(2, 1), m(0, 2)*m(2, 1) - m(0, 1)*m(2, 2), m(0, 1)*m(1, 2) - m(0, 2)*m(1, 1),
        m(1, 2)*m(2, 0) - m(1, 0)*m(2, 2), m(0, 0)*m(2, 2) - m(0, 2)*m(2, 0), m(0, 2)*m(1, 0) - m(0, 0)*m(1, 2),
        m(1, 0)*m(2, 1) - m(1, 1)*m(2, 0), m(0, 1)*m(2, 0) - m(0, 0)*m(2, 1), m(0, 0)*m(1, 1) - m(0, 1)*m(1, 0)
    )/determinant(m);
}

template<typename Type>
__host__ __device__ Type maxEigenValue(const Mat<Type, 2> & m)
{
    Type trace = m(0, 0) + m(1, 1);
    Type det = determinant(m);
    return Type(0.5)*(trace + std::sqrt(max(trace*trace - Type(4.0)*det, Type(0.0))));
}

template<typename Type>
__host__ __device__ Type maxEigenValue(const Mat<Type, 3> & m)
{
    Type triSq = sqr(m(0, 1)) + sqr(m(0, 2)) + sqr(m(1, 2));
    if (triSq == Type(0)) {
        return max(m(0, 0), max(m(1, 1), m(2, 2)));
    } else {
        Type q = Type(1.0/3.0)*(m(0, 0) + m(1, 1) + m(2, 2));
        Type pSq = Type(2.0)*triSq + sqr(m(0, 0) - q) + sqr(m(1, 1) - q) + sqr(m(2, 2) - q);
        Type p = std::sqrt(pSq*Type(1.0/6.0));
        Type r = determinant(m - q)*(Type(0.5)/(p*p*p));
        Type phi = Type(1.0/3.0)*std::acos(min(Type(1.0), max(Type(-1.0), r)));
        return q + Type(2.0)*p*std::cos(phi);
    }
}
