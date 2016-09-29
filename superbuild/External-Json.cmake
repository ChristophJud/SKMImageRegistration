message( "External project - Json" )

find_package(Git)
if(NOT GIT_FOUND)
  message(ERROR "Cannot find git. git is required for Superbuild")
endif()

option( USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." ON)

set(git_protocol "git")
if(NOT USE_GIT_PROTOCOL)
  set(git_protocol "http")
endif()

set( Json_DEPENDENCIES)

ExternalProject_Add(Json
  DEPENDS ${Json_DEPENDENCIES}
  GIT_REPOSITORY ${git_protocol}://github.com/ChristophJud/json.git
  GIT_TAG include_dir_relocation
  SOURCE_DIR Json
  BINARY_DIR Json-build
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CMAKE_GENERATOR ${EP_CMAKE_GENERATOR}
  CMAKE_ARGS
    ${ep_common_args}
    -DCMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    -DBuildTests:BOOL=OFF
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DEPENDENCIES_DIR}
)

set( Json_DIR ${INSTALL_DEPENDENCIES_DIR}/cmake/ )