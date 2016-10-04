cmake_minimum_required(VERSION 3.3)
INCLUDE(ExternalProject)

project(SKM-Superbuild)

# set build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)
set(CMAKE_CXX_STANDARD 11)
add_compile_options(-std=c++11)
SET ( CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "compile flags" FORCE)

option( BUILD_SHARED_LIBS "Build shared libraries" ON )
set(SKM_NUM_SPACE_DIMENSIONS 3 CACHE INTEGRAL "Number of space dimensions (2 or 3)")
option( USE_SINGLE_PRECISION "Build with single precision" OFF)

include( ExternalProject )
set( BUILD_TYPE ${CMAKE_BUILD_TYPE} )
if( NOT INSTALL_DEPENDENCIES_DIR )
    set( INSTALL_DEPENDENCIES_DIR ${CMAKE_BINARY_DIR}/thirdparty CACHE STRING "Install directory for dependencies")
endif()
set( Patches_DIR ${CMAKE_CURRENT_SOURCE_DIR}/patches )
set( SKM_DEPENDENCIES )

option( USE_SYSTEM_ITK "Use system libraries for ITK" OFF )
if( ${USE_SYSTEM_ITK} MATCHES "OFF" )
  include( External-ITK.cmake )
  set( SKM_DEPENDENCIES ${SKM_DEPENDENCIES} ITK )
else()
  find_package( ITK REQUIRED )
  include( ${ITK_USE_FILE} )
endif()

option( USE_SYSTEM_BPrinter "Use system libraries for BPrinter" OFF )
if( ${USE_SYSTEM_BPrinter} MATCHES "OFF" )
  include( External-BPrinter.cmake )
  set( SKM_DEPENDENCIES ${SKM_DEPENDENCIES} BPrinter )
else()
  find_package( BPrinter REQUIRED )
  include( ${BPrinter_USE_FILE} )
endif()

option( USE_SYSTEM_Json "Use system libraries for json" OFF )
if( ${USE_SYSTEM_Json} MATCHES "OFF" )
  include( External-Json.cmake )
  set( SKM_DEPENDENCIES ${SKM_DEPENDENCIES} Json )
else()
  find_package( Json REQUIRED )
  include( ${Json_USE_FILE} )
endif()

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${INSTALL_DEPENDENCIES_DIR}:${INSTALL_DEPENDENCIES_DIR}/lib")

ExternalProject_Add( SKM
  DEPENDS ${SKM_DEPENDENCIES}
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..
  BINARY_DIR SKM-build
  CMAKE_GENERATOR ${EP_CMAKE_GENERATOR}
  CMAKE_ARGS
    ${ep_common_args}
    -DCMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}
    -DITK_DIR:PATH=${ITK_DIR}
    -Dbprinter_DIR:PATH=${bprinter_DIR}
    -Dnlohmann_json_DIR:PATH=${Json_DIR}
    -DCMAKE_CXX_FLAGS:STRING=-std=c++11
    -DSKM_NUM_SPACE_DIMENSIONS:INTEGRAL=${SKM_NUM_SPACE_DIMENSIONS}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_INSTALL_RPATH:PATH=${CMAKE_INSTALL_RPATH}
    -DUSE_SINGLE_PRECISION:BOOL=${USE_SINGLE_PRECISION}
)
