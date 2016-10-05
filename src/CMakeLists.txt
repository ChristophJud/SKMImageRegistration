FIND_PACKAGE(ITK REQUIRED)
find_package(bprinter REQUIRED)
find_package(nlohmann_json REQUIRED)

# configure application
if(DEFINED SKM_NUM_SPACE_DIMENSIONS)
if(SKM_NUM_SPACE_DIMENSIONS EQUAL 3)
    message(STATUS "Building application with 3 space dimensions")
elseif(SKM_NUM_SPACE_DIMENSIONS EQUAL 2)
    message(STATUS "Building application with 2 space dimensions")
else()
    message(SEND_ERROR "Only 2 or 3 space dimensions supported")
endif()
else()
set(SKM_NUM_SPACE_DIMENSIONS 3 "Set space dimensions")
endif()
add_definitions(-DSpaceDimensions=${SKM_NUM_SPACE_DIMENSIONS})

if(${USE_SINGLE_PRECISION})
    add_definitions(-DSinglePrecisionType=1)
    message(STATUS "Setting precision type to float")
else()
    add_definitions(-DSinglePrecisionType=0)
    message(STATUS "Setting precision type to double")
endif()

if(SKM_USE_WENDLAND_C0)
    add_definitions(-DUSE_WENDLAND_C0=1)
else()
    add_definitions(-DUSE_WENDLAND_C0=0)
endif()

# dependency versions
add_definitions(-DITK_VERSION="${ITK_VERSION}")
add_definitions(-DJSON_VERSION="${nlohmann_json_VERSION}")
add_definitions(-DBPRINTER_VERSION="${bprinter_VERSION}")

FILE(GLOB_RECURSE HFiles "${CMAKE_SOURCE_DIR}/include/*.h")
FILE(GLOB_RECURSE HXXFiles "${CMAKE_SOURCE_DIR}/include/*.hxx")

find_package(CUDA REQUIRED)
if(DEFINED ${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA includes: ${CUDA_INCLUDE_DIRS}")
else()
message(STATUS "CUDA includes: no specific cuda include dirs found")
endif()

set(_IncludeDirs
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/include/cuda"
    ${JSON_INCLUDE_DIR}
    ${BPRINTER_INCLUDE_DIR}
    ${ITK_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS} # in TransformAdapter, Vec.h is used which depends on cuda_runtime.h
)
set(_LinkDirs
    ${ITK_LIBRARY_DIRS}
    ${BPRINTER_LIBRARY_DIR}
)

add_library(bprinter STATIC IMPORTED)
set_property(TARGET bprinter PROPERTY IMPORTED_LOCATION ${BPRINTER_LIBRARY_DIR}/libbprinter.a)

set(_Headers ${HFiles} ${HXXFiles})
set(_Libraries 
    ${ITK_LIBRARIES}
    bprinter
    GpuEvaluator
)
set( _Apps
	SKMReg
    SKMRegv4
)

include_directories(${_IncludeDirs})
link_directories(${_LinekDirs})

FIND_PACKAGE( OpenMP )
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include( ${ITK_USE_FILE} ) # needed to register itk IO factories

foreach( _app ${_Apps})
    add_executable( ${_app} ${_app}.cxx ${_Headers})
	target_link_libraries( ${_app} ${_Libraries})
	set_target_properties( ${_app} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
endforeach()

INSTALL(TARGETS ${_Apps} RUNTIME DESTINATION ${SKM_BINARY_DESTINATION})
