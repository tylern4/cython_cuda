cmake_minimum_required(VERSION 3.20)
project(saxpy LANGUAGES C CXX)

# Set build type
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -fopenmp ${CMAKE_EXE_LINKER_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp ${CMAKE_EXE_LINKER_FLAGS}")

option(TARGET_GPU "Enable OpenACC GPU       Support" OFF)
if(TARGET_GPU)
    find_package(OpenACC REQUIRED)
    set(CMAKE_C_FLAGS "${OpenACC_C_FLAGS} -Minfo=accel")
    set(CMAKE_CXX_FLAGS "${OpenACC_CXX_FLAGS} -Minfo=accel")
    set(TARGET_STDPAR "-acc=gpu")
endif()


add_executable(testbin src/test.cxx)
target_compile_options(testbin PUBLIC ${TARGET_ACCEL})
