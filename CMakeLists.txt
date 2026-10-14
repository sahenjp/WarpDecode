cmake_minimum_required(VERSION 3.24)
project(warodecode LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WARODECODE_BUILD_ENGINE "Build the CUDA inference engine" ON)

add_executable(warodecode-tui tools/warodecode_tui.cpp)
target_include_directories(warodecode-tui PRIVATE src)
target_compile_options(warodecode-tui PRIVATE -O3 -Wall -Wextra -Wpedantic)

add_executable(warodecode-qwen38-trace-test tests/qwen38_trace_test.cpp)
target_include_directories(warodecode-qwen38-trace-test PRIVATE src)
target_compile_options(warodecode-qwen38-trace-test PRIVATE -O2 -Wall -Wextra -Wpedantic)

# Issue #1 residual-lifetime regression: F32 residual must beat the F16
# round-trip lifetime over a 48-layer accumulation. Host-only, no GPU.
add_executable(warodecode-issue1-residual-accum-test tests/issue1_residual_accum_test.cpp)
target_include_directories(warodecode-issue1-residual-accum-test PRIVATE src)
target_compile_options(warodecode-issue1-residual-accum-test PRIVATE -O2 -Wall -Wextra -Wpedantic)

add_executable(warodecode-issue1-ffn-precision-test tests/issue1_ffn_precision_test.cpp)
target_include_directories(warodecode-issue1-ffn-precision-test PRIVATE src)
target_compile_options(warodecode-issue1-ffn-precision-test PRIVATE -O2 -Wall -Wextra -Wpedantic)

add_executable(warodecode-issue1-ffn-ablation-test tests/issue1_ffn_ablation_test.cpp)
target_include_directories(warodecode-issue1-ffn-ablation-test PRIVATE src)
target_compile_options(warodecode-issue1-ffn-ablation-test PRIVATE -O2 -Wall -Wextra -Wpedantic)

if(WARODECODE_BUILD_ENGINE)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler(nvcc)がありません。TUIだけなら -DWARODECODE_BUILD_ENGINE=OFF")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "GPUを固定してください: -DCMAKE_CUDA_ARCHITECTURES=70 など")
  endif()

  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch LESS 50)
      message(FATAL_ERROR "warodecodeはSM50以上を対象にします")
    endif()
  endforeach()

  add_executable(warodecode src/warodecode.cu)
  target_include_directories(warodecode PRIVATE src)
  target_compile_options(warodecode PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--extra-device-vectorization;-lineinfo;-Xptxas=-warn-spills>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
  set_target_properties(warodecode PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  add_executable(warodecode-lowbit-test tests/lowbit_cuda_test.cu)
  target_include_directories(warodecode-lowbit-test PRIVATE src)
  target_compile_options(warodecode-lowbit-test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math>
  )
  set_target_properties(warodecode-lowbit-test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  add_executable(warodecode-qwen38-layer0-probe tools/qwen38_layer0_probe.cu)
  target_include_directories(warodecode-qwen38-layer0-probe PRIVATE src)
  target_compile_options(warodecode-qwen38-layer0-probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
  set_target_properties(warodecode-qwen38-layer0-probe PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  add_executable(warodecode-qwen38-moe-cuda-test tests/qwen38_moe_cuda_test.cu)
  target_include_directories(warodecode-qwen38-moe-cuda-test PRIVATE src)
  target_compile_options(warodecode-qwen38-moe-cuda-test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
  set_target_properties(warodecode-qwen38-moe-cuda-test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  add_executable(warodecode-issue1-moe-precision-cuda-test tests/issue1_moe_precision_cuda_test.cu)
  target_include_directories(warodecode-issue1-moe-precision-cuda-test PRIVATE src)
  target_compile_options(warodecode-issue1-moe-precision-cuda-test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
  set_target_properties(warodecode-issue1-moe-precision-cuda-test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  # GDN / HC kernel tests were previously orphaned (present in tests/ but not
  # registered here), which let the F32 residual API change break the HC test
  # unnoticed. Keep them in the normal build so regressions surface.
  add_executable(warodecode-qwen38-gdn-cuda-test tests/qwen38_gdn_cuda_test.cu)
  target_include_directories(warodecode-qwen38-gdn-cuda-test PRIVATE src)
  target_compile_options(warodecode-qwen38-gdn-cuda-test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
  set_target_properties(warodecode-qwen38-gdn-cuda-test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  add_executable(warodecode-qwen38-hc-cuda-test tests/qwen38_hc_cuda_test.cu)
  target_include_directories(warodecode-qwen38-hc-cuda-test PRIVATE src)
  target_compile_options(warodecode-qwen38-hc-cuda-test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
  set_target_properties(warodecode-qwen38-hc-cuda-test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

  add_executable(warodecode-qwen38-qsa-cuda-test tests/qwen38_qsa_cuda_test.cu)
  target_include_directories(warodecode-qwen38-qsa-cuda-test PRIVATE src)
  target_compile_options(warodecode-qwen38-qsa-cuda-test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  )
  set_target_properties(warodecode-qwen38-qsa-cuda-test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()
