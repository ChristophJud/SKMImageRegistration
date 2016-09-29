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

template<typename Type, unsigned int Dimensions>
Vec<Type, Dimensions> fromItk(const itk::Vector<Type, Dimensions> &v)
{
    Vec<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = v[i];
    return result;
}

template<typename Type, unsigned int Dimensions>
Vec<Type, Dimensions> fromItk(const itk::Point<Type, Dimensions> &v)
{
    Vec<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = v[i];
    return result;
}

template<typename Type, unsigned int Dimensions>
Vec<Type, Dimensions> fromItk(const itk::CovariantVector<Type, Dimensions> &v)
{
    Vec<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = v[i];
    return result;
}

template<unsigned int Dimensions>
Vec<int, Dimensions> fromItk(const itk::Index<Dimensions> &v)
{
    Vec<int, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = v[i];
    return result;
}

template<typename ImageType, unsigned int Dimensions>
ImageType fromItk(const itk::ImageBase<Dimensions> &img)
{
    return ImageType(nullptr, fromItk(img.GetOrigin()), fromItk(img.GetSpacing()), fromItk(img.GetBufferedRegion().GetUpperIndex()) + 1);
}

template<typename Type, unsigned int Dimensions>
itk::Vector<Type, Dimensions> toItkVector(const Vec<Type, Dimensions> &v)
{
    itk::Vector<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = v[i];
    return result;
}

template<typename Type, unsigned int Dimensions>
itk::Point<Type, Dimensions> toItkPoint(const Vec<Type, Dimensions> &p)
{
    itk::Point<Type, Dimensions> result;
    for (int i = 0; i < Dimensions; ++i)
        result[i] = p[i];
    return result;
}