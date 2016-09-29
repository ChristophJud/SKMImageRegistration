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
#include <iostream>
#include <memory>

static inline void cudaCheck(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        std::cout << file << ":" << line << " Cuda error! " << cudaGetErrorString(err) << std::endl;
        std::exit(-1);
    }
}

struct CudaDeleter { void operator()(void *p) { cudaCheck(cudaFree(p), __FILE__, __LINE__); } };

template<typename T>
using cuda_ptr = std::unique_ptr<T, CudaDeleter>;

template<typename T>
inline cuda_ptr<T> allocCuda(size_t size, const T *src = nullptr)
{
    T *data;
    cudaCheck(cudaMalloc(&data, size*sizeof(T)), __FILE__, __LINE__);

    if (src)
        cudaCheck(cudaMemcpy(data, src, size*sizeof(T), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    return cuda_ptr<T>(data);
}
