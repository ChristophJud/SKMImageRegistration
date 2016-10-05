file(GLOB test_files "*.cpp")

enable_testing()

set(target_tests "")

# configure application
if(SKM_NUM_SPACE_DIMENSIONS EQUAL 3)
    message(STATUS "Building tests with 3 space dimensions")
elseif(SKM_NUM_SPACE_DIMENSIONS EQUAL 2)
    message(STATUS "Building tests with 2 space dimensions")
else()
    message(SEND_ERROR "Only 2 or 3 space dimensions supported")
endif()
add_definitions(-DSpaceDimensions=${SKM_NUM_SPACE_DIMENSIONS})

if(${USE_SINGLE_PRECISION})
    add_definitions(-DSinglePrecisionType=1)
    message(STATUS "Setting precision type to float")
else()
    add_definitions(-DSinglePrecisionType=0)
    message(STATUS "Setting precision type to double")
endif()

set(_IncludeDirs
	"${CMAKE_SOURCE_DIR}/include"
	"${CMAKE_SOURCE_DIR}/include/cuda"
)
set(_LinkDirs)
set(_Libraries
	${_CudaLibraries}
)

# ITK stuff
if(BUILD_ITK)
find_package(ITK REQUIRED)

set(_IncludeDirs ${_IncludeDirs} 
	${ITK_INCLUDE_DIRS}
)
set(_LinkDirs ${_LinkDirs} ${ITK_LIBRARY_DIRS})
set(_Libraries ${_Libraries} ${ITK_LIBRARIES} ${Glue})

else()

# remove itk tests:
# - check if test files start with "itk"
#   and remove them from test_files
set(new_test_files)
foreach(_test ${test_files})
get_filename_component(_test_file ${_test} NAME)
if(${_test_file} MATCHES "itk*")
else()
set(new_test_files ${new_test_files} ${_test})
endif()
endforeach()
set(test_files ${new_test_files})
endif()


include_directories(${_IncludeDirs})
link_directories(${_LinekDirs})

include( ${ITK_USE_FILE} ) # needed to register itk IO factories

foreach( _filename ${test_files})
	get_filename_component(_test ${_filename} NAME_WE)
	add_executable(${_test} ${_filename})
	target_link_libraries(${_test} ${_Libraries})
	set_target_properties(${_test} PROPERTIES OUTPUT_NAME ${_test})
	add_test(${_test} ${_test})
	set(target_tests ${target_tests} ${_test})
endforeach()

INSTALL(TARGETS ${target_tests} RUNTIME DESTINATION tests)



