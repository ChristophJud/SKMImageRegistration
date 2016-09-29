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

#include <limits.h>
#include <stdint.h>

class UniformRng
{
    uint64_t _state;
    uint64_t _sequence;

public:
    UniformRng() : _state(0xBA5EBA11DEADBEEFULL), _sequence(0) {}

    inline uint32_t nextI()
    {
        uint64_t oldState = _state;
        _state = oldState*6364136223846793005ULL + (_sequence | 1);
        uint32_t xorShifted = uint32_t(((oldState >> 18u) ^ oldState) >> 27u);
        uint32_t rot = oldState >> 59u;
        return (xorShifted >> rot) | (xorShifted << (uint32_t(-int32_t(rot)) & 31));
    }

    inline ScalarType nextF()
    {
        return nextI()/ScalarType(UINT_MAX);
    }

    template<int Dimensions>
    inline Vec<ScalarType, Dimensions> nextV()
    {
        Vec<ScalarType, Dimensions> result;
        for (int i = 0; i < Dimensions; ++i)
            result[i] = nextF();
        return result;
    }
};
