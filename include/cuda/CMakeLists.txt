cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

# configure application
if(SKM_NUM_SPACE_DIMENSIONS EQUAL 3)
    message(STATUS "Building cuda kernels with 3 space dimensions")
elseif(SKM_NUM_SPACE_DIMENSIONS EQUAL 2)
    message(STATUS "Building cuda kernels with 2 space dimensions")
else()
    message(SEND_ERROR "Only 2 or 3 space dimensions supported for cuda kernels")
endif()
add_definitions(-DSpaceDimensions=${SKM_NUM_SPACE_DIMENSIONS})

if(${USE_SINGLE_PRECISION})
    add_definitions(-DSinglePrecisionType=1)
    message(STATUS "Setting precision type to float")
else()
    add_definitions(-DSinglePrecisionType=0)
    message(STATUS "Setting precision type to double")
endif()

include_directories(
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/cuda"
    ${CUDA_INCLUDE_DIRS}    
)

#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(BUILD_SHARED_LIBS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CUDA_CUSTOM_NVCC_FLAGS -use_fast_math CACHE STRING "Further custum nvcc flags")
set(CUDA_NVCC_FLAGS -Xptxas -v "-std=c++11" ${CUDA_CUSTOM_NVCC_FLAGS})

message(STATUS "Cuda version found: ${CUDA_VERSION}")

if(CUDA_VERSION LESS 7.0)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
endif()
if(CUDA_VERSION LESS 8.0)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52) # GTX-970
endif()
if(CUDA_VERSION EQUAL 8.0)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61) # GTX-1080
endif()

message(STATUS "nvcc flags: ${CUDA_NVCC_FLAGS}")

FILE(GLOB_RECURSE HFiles  "${CMAKE_SOURCE_DIR}/include/cuda/*.h")
FILE(GLOB_RECURSE CUFiles "${CMAKE_SOURCE_DIR}/include/cuda/*.cu")

set(_CudaLibraries CACHE INTERNAL "")

foreach( _kernel ${CUFiles})
    get_filename_component(_kernel_name ${_kernel} NAME_WE)
    cuda_add_library(${_kernel_name} ${_kernel})
    set(_CudaLibraries ${_CudaLibraries} ${_kernel_name} CACHE INTERNAL "")
    INSTALL(TARGETS ${_kernel_name} DESTINATION ${SKM_LIBRARY_DESTINATION} PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
endforeach()

INSTALL(FILES 
    ${HFiles}
DESTINATION
    ${SKM_INCLUDE_DESTINATION}/cuda
PERMISSIONS
	OWNER_READ GROUP_READ WORLD_READ
)