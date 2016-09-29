cmake_minimum_required(VERSION 3.3)

project(SKM VERSION 0.0.1 LANGUAGES CXX)

# set build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 11)
add_compile_options(-std=c++11)
SET ( CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "compile flags" FORCE)

set(BUILD_ITK ON) # to get itk adapter of PGD library

set(SKM_NUM_SPACE_DIMENSIONS 3 CACHE INTEGRAL "Number of space dimensions (2 or 3)")
option(SKM_USE_WENDLAND_C0 "Use the combined kernel including wendland C0" ON )


# define project variables
set(SKM_TARGET_NAME ${PROJECT_NAME})
set(SKM_PACKAGE_NAME ${SKM_TARGET_NAME})
set(SKM_TARGETS_FILENAME "${SKM_PACKAGE_NAME}Targets.cmake")
set(SKM_CONFIG_IN_FILENAME "config.cmake.in")
set(SKM_CONFIG_FILENAME "${SKM_PACKAGE_NAME}Config.cmake")
set(SKM_CONFIGVERSION_FILENAME "${SKM_PACKAGE_NAME}ConfigVersion.cmake")
set(SKM_CONFIG_DESTINATION "cmake")
set(SKM_INCLUDE_DESTINATION "include")
set(SKM_LIBRARY_DESTINATION "lib")
set(SKM_BINARY_DESTINATION "bin")

add_definitions(-DSKM_VERSION="${PROJECT_VERSION}")

ADD_SUBDIRECTORY(include)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)

INCLUDE(CTest)

# generate a config and config version file for the package
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${SKM_CONFIG_IN_FILENAME}
    "${CMAKE_CURRENT_BINARY_DIR}/${SKM_CONFIG_FILENAME}"
    INSTALL_DESTINATION ${SKM_CONFIG_DESTINATION}
    PATH_VARS SKM_INCLUDE_DESTINATION SKM_LIBRARY_DESTINATION SKM_BINARY_DESTINATION
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${SKM_CONFIGVERSION_FILENAME}"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES 
    "${CMAKE_CURRENT_BINARY_DIR}/${SKM_CONFIG_FILENAME}"
	"${CMAKE_CURRENT_BINARY_DIR}/${SKM_CONFIGVERSION_FILENAME}"
	DESTINATION ${SKM_CONFIG_DESTINATION}
)


INCLUDE(CPack)
