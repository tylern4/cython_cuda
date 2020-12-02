cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES C Fortran CXX CUDA)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g -Ofast")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

set(CMAKE_MODULE_PATH
   ${CMAKE_MODULE_PATH}
   ${PROJECT_SOURCE_DIR}/cmake)
 
add_library(compute_lib STATIC
    src/compute.cu
    src/jacobi.cu
)
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(compute_lib PUBLIC cxx_std_14)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(compute_lib
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
add_executable(test src/test.cxx)
 
set_property(TARGET compute_lib
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(test PRIVATE compute_lib)

find_package(Cython)
IF(Cython_FOUND)
  find_package(PythonInterp REQUIRED)
  find_package(PythonLibs REQUIRED)
  find_package(PythonExtensions REQUIRED)
  find_package(NumPy REQUIRED)

  include_directories(${PROJECT_SOURCE_DIR}/src ${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR} ${NumPy_INCLUDE_DIR})

  file(GLOB CYTHON_TARGETS src/*.pyx )
  foreach(pyx ${CYTHON_TARGETS} )
        get_filename_component(pyxname ${pyx} NAME)
        string( REPLACE ".pyx" "" pyxname ${pyxname} )
        add_cython_target(${pyxname} ${pyx} CXX PY3 OUTPUT_VAR _pyxname)
        add_library(${pyxname} MODULE ${_pyxname})
        python_extension_module(${pyxname})
        target_link_libraries(${pyxname} compute_lib ${PYTHON_LIBRARIES})
  endforeach( pyx ${CYTHON_TARGETS} )
ENDIF()

