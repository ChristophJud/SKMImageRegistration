message( "External project - ITK" )

find_package(Git)
if(NOT GIT_FOUND)
  message(ERROR "Cannot find git. git is required for Superbuild")
endif()

option( USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." ON)

set(git_protocol "git")
if(NOT USE_GIT_PROTOCOL)
  set(git_protocol "http")
endif()

set( ITK_DEPENDENCIES )

if( ${USE_SYSTEM_Eigen} MATCHES "OFF" )
  set( ITK_DEPENDENCIES Eigen3 ${ITK_DEPENDENCIES} )
endif()

ExternalProject_Add(ITK
  DEPENDS ${ITK_DEPENDENCIES}
  GIT_REPOSITORY ${git_protocol}://github.com/InsightSoftwareConsortium/ITK.git
  GIT_TAG v4.10.0
  SOURCE_DIR ITK
  BINARY_DIR ITK-build
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CMAKE_GENERATOR ${EP_CMAKE_GENERATOR}
  CMAKE_ARGS
    ${ep_common_args}
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
    -DBUILD_TESTING:BOOL=OFF
    -DCMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}
    -DITK_BUILD_DEFAULT_MODULES:BOOL=ON
    -DModule_ITKReview:BOOL=ON
    -DITK_LEGACY_REMOVE:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DEPENDENCIES_DIR}
    -DITK_USE_SYSTEM_HDF5:BOOL=OFF
    -DITK_USE_64BITS_IDS:BOOL=ON
    -DVCL_INCLUDE_CXX_0X:BOOL=ON
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    -DITK_USE_FLOAT_SPACE_PRECISION:BOOL=${USE_SINGLE_PRECISION}
)

set( ITK_DIR ${INSTALL_DEPENDENCIES_DIR}/lib/cmake/ITK-4.10/ )