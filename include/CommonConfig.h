/*
 * Copyright 2016 University of Basel, Medical Image Analysis Center
 *
 * Author: Christoph Jud     (christoph.jud@unibas.ch)
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

#ifndef SpaceDimensions
#define SpaceDimensions 3
#endif

#if !(SpaceDimensions == 3) && !(SpaceDimensions == 2) 
#error Only space dimensions of 2 or 3 are supported
#endif

#if SinglePrecisionType==0
typedef double ScalarType;
#elif SinglePrecisionType==1
typedef float ScalarType;
#else
#error Scalar type has to be defined as float or double
#endif

#ifndef USE_WENDLAND_C0
#define USE_WENDLAND_C0 1
#endif

#define NOAPPROX
#define NUM_INTERPOLANTS 9 // 1, 5 or 9
//#define REDUCE_REGISTER_SPILLING
//#define EXEC_SINGLE_THREAD
