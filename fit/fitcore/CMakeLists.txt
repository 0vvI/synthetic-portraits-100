cmake_minimum_required(VERSION 3.18)
project(fitcore LANGUAGES CXX)

option(FITCORE_PYTHON "Also build the Python module fitcore (needs pybind11)" OFF)
option(FITCORE_SANITIZE "Build the library and the tests with AddressSanitizer and UBSan" OFF)
option(FITCORE_BLAS "Let Eigen hand its large matrix products to a BLAS library (Accelerate on macOS)" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

# Warnings for the targets built from this folder; Eigen and pybind11 come in as system headers.
function(fitcore_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

add_library(fitcore STATIC src/fitcore.cpp)
target_include_directories(fitcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fitcore PUBLIC Eigen3::Eigen PRIVATE Threads::Threads)
# Position-independent, because the Python module links this static library.
set_target_properties(fitcore PROPERTIES POSITION_INDEPENDENT_CODE ON)
fitcore_warnings(fitcore)
if(FITCORE_BLAS)
  # The solver's time goes into one product, J^T J. A tuned BLAS computes it several times faster than
  # Eigen's own kernel; the results differ only in the last digits.
  find_package(BLAS REQUIRED)
  target_compile_definitions(fitcore PUBLIC EIGEN_USE_BLAS)
  target_link_libraries(fitcore PUBLIC BLAS::BLAS)
endif()
if(FITCORE_SANITIZE)
  # PUBLIC, so the tests are compiled and linked with the same checks.
  target_compile_options(fitcore PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(fitcore PUBLIC -fsanitize=address,undefined)
endif()

add_executable(fitcore_tests tests/test_fitcore.cpp)
target_link_libraries(fitcore_tests PRIVATE fitcore)
fitcore_warnings(fitcore_tests)

enable_testing()
foreach(id IN ITEMS T1 T2 T3 T4 T5 K0)
  add_test(NAME ${id} COMMAND fitcore_tests ${id})
endforeach()

if(FITCORE_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  # The target needs a name of its own; OUTPUT_NAME makes the file importable as "fitcore".
  pybind11_add_module(fitcore_python python/bindings.cpp)
  target_link_libraries(fitcore_python PRIVATE fitcore)
  set_target_properties(fitcore_python PROPERTIES
    OUTPUT_NAME fitcore
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  fitcore_warnings(fitcore_python)
endif()
