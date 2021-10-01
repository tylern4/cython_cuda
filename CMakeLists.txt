cmake_minimum_required(VERSION 3.20)
project(saxpy LANGUAGES C CXX)

# Set build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Build Options
option(TARGET_GPU "Enable OpenACC GPU       Support" OFF)
option(TARGET_CPU "Enable OpenACC Multicore support" OFF)

if(TARGET_CPU OR TARGET_GPU)
    find_package(OpenACC REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenACC_C_FLAGS} -Minfo=accel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS} -Minfo=accel")
endif()

add_executable(testbin src/test.cxx)


if(TARGET_CPU)
    set(TARGET_ACCEL "-ta=multicore")
    set(TARGET_STDPAR "-stdpar=multicore")
elseif(TARGET_GPU)
    set(TARGET_STDPAR "-acc=gpu")
endif()

target_compile_options(testbin PUBLIC ${TARGET_ACCEL})
