cmake_minimum_required(VERSION 3.18)

project(
	tiny-cuda-mem-pool
	VERSION 1.0
	DESCRIPTION "Tiny C++/CUDA GPU memory pool"
	LANGUAGES CXX CUDA
)

option(TCMP_BUILD_TESTS "Build tests for tiny-cuda-mem-pool?" ON)

# Set a default configuration if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Only include the necessary source files and headers
set(TCMP_SOURCES
	src/common.cu
	src/gpu_memory.cu
)

list(APPEND TCMP_LIBRARIES cuda)
list(APPEND TCMP_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
add_library(tiny-cuda-mem-pool STATIC ${TCMP_SOURCES})
target_include_directories(tiny-cuda-mem-pool PUBLIC ${TCMP_INCLUDE_DIRECTORIES})
target_link_libraries(tiny-cuda-mem-pool PUBLIC ${TCMP_LIBRARIES})

if(TCMP_BUILD_TESTS)
	add_subdirectory(tests)
endif()