#pragma once

#include <string>

namespace wd {

struct GpuKernelPlan {
    int sm = 0;
    const char* family = "unsupported";
    const char* decode = "unsupported";
    const char* verify = "unsupported";
    const char* attention = "online-fp16";
    bool legacy_toolkit = false;
    bool tensor_cores = false;
};

inline GpuKernelPlan kernel_plan_for_sm(int sm) {
    if (sm >= 120) return {sm, "blackwell-geforce", "int4/fp4-tensorcore", "fp4/int4-small-gemm", "native-fa", false, true};
    if (sm >= 100) return {sm, "blackwell-datacenter", "int4/fp4-tensorcore", "fp4/int4-small-gemm", "native-fa", false, true};
    if (sm >= 90)  return {sm, "hopper", "int4/fp8-tensorcore", "wgmma-small-gemm", "native-fa", false, true};
    if (sm >= 89)  return {sm, "ada", "int4-tensorcore", "int4-small-gemm", "native-fa", false, true};
    if (sm >= 80)  return {sm, "ampere", "int4/int8-tensorcore", "marlin-style-small-gemm", "native-fa", false, true};
    if (sm >= 75)  return {sm, "turing", "int8-mma/q4", "int8-small-gemm", "turing-fa", false, true};
    if (sm >= 70)  return {sm, "volta", "q4-fp16-mma", "m8n8k4-small-gemm", "volta-fa", true, true};
    if (sm >= 61)  return {sm, "pascal-dp4a", "q4-q8-dp4a", "dp4a-small-gemm", "online-fp16", true, false};
    if (sm >= 60)  return {sm, "pascal-hbm", "q4-half2", "fp16-small-gemm", "online-fp16", true, false};
    if (sm >= 50)  return {sm, "maxwell", "q4-simt", "q4-simt", "online-fp16", true, false};
    return {sm, "unsupported", "unsupported", "unsupported", "unsupported", true, false};
}

inline std::string sm_name(int sm) {
    return "sm_" + std::to_string(sm);
}

} // namespace wd
