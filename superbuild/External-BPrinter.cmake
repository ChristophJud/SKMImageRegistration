message( "External project - BPrinter" )

find_package(Git)
if(NOT GIT_FOUND)
  message(ERROR "Cannot find git. git is required for Superbuild")
endif()

option( USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." ON)

set(git_protocol "git")
if(NOT USE_GIT_PROTOCOL)
  set(git_protocol "http")
endif()

set( BPrinter_DEPENDENCIES)

ExternalProject_Add(BPrinter
  DEPENDS ${BPrinter_DEPENDENCIES}
  GIT_REPOSITORY ${git_protocol}://github.com/dattanchu/bprinter.git
  GIT_TAG master
  SOURCE_DIR BPrinter
  BINARY_DIR BPrinter-build
  UPDATE_COMMAND ""
  PATCH_COMMAND "" 
  CMAKE_GENERATOR ${EP_CMAKE_GENERATOR}
  CMAKE_ARGS
    ${ep_common_args}
    -DCMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    -DUSE_BOOST_KARMA:BOOL=OFF
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_DEPENDENCIES_DIR}
)

set( bprinter_DIR ${INSTALL_DEPENDENCIES_DIR}/cmake/ )