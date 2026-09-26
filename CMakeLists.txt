cmake_minimum_required(VERSION 3.24)
project(warpdecode LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WARPDECODE_BUILD_ENGINE "Build the CUDA inference engine" ON)

add_executable(warpdecode-tui tools/warpdecode_tui.cpp)
target_include_directories(warpdecode-tui PRIVATE src)
target_compile_options(warpdecode-tui PRIVATE -O3 -Wall -Wextra -Wpedantic)

if(WARPDECODE_BUILD_ENGINE)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler(nvcc)がありません。TUIだけなら -DWARPDECODE_BUILD_ENGINE=OFF")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "GPUを固定してください: -DCMAKE_CUDA_ARCHITECTURES=70 など")
  endif()

  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch LESS 50)
      message(FATAL_ERROR "warpdecodeはSM50以上を対象にします")
    endif()
  endforeach()

  add_executable(warpdecode src/warpdecode.cu)
  target_include_directories(warpdecode PRIVATE src)
  target_compile_options(warpdecode PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--extra-device-vectorization;-lineinfo;-Xptxas=-warn-spills>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
  set_target_properties(warpdecode PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()
