#include "onnxruntime_c_api.h"
#include <stdint.h>
#include <stdio.h>

#if _WIN32
#define FFI_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FFI_PLUGIN_EXPORT __attribute__((visibility("default"))) __attribute__((used))
#endif

FFI_PLUGIN_EXPORT int32_t check_ort_status() {
  printf("[ORT_PLUGIN] Starting ORT check...\n");

    const OrtApiBase* api_base = OrtGetApiBase();
    if (api_base == NULL) {
        printf("[ORT_PLUGIN] ERROR: OrtGetApiBase() returned NULL.\n");
        return 0;
    }
    printf("[ORT_PLUGIN] Successfully retrieved OrtApiBase.\n");

    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    if (api == NULL) {
        printf("[ORT_PLUGIN] ERROR: Failed to get API for version %d.\n", ORT_API_VERSION);
        return 0;
    }

    printf("[ORT_PLUGIN] Successfully initialized ONNX Runtime API.\n");
    return 1;
}