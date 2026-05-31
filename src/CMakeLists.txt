# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(stat_ort_plugin_library VERSION 0.0.1 LANGUAGES C)

add_library(stat_ort_plugin SHARED
  "stat_ort_plugin.c"
  "mel.c"
  "compute_fft.c"
  "normalize_white_space.c"
  "load_vocab.c"
  "vaani_core.c"
  "vaani_decode.c"
  "vaani_vad.c"
  "vaani_speaker.c"
  "vaani_stream.c"
  "vaani_utils.c"
)

set_target_properties(stat_ort_plugin PROPERTIES
  PUBLIC_HEADER stat_ort_plugin.h
  OUTPUT_NAME "stat_ort_plugin"

)

target_compile_definitions(stat_ort_plugin PUBLIC DART_SHARED_LIB)

target_compile_options(stat_ort_plugin PRIVATE -O3 -ffast-math)

# Verbose debug logging (LOGD). Off by default — release builds carry no
# per-segment logging overhead. Enable with: -DVAANI_DEBUG=ON
option(VAANI_DEBUG "Enable verbose Vaani debug logging" OFF)
if (VAANI_DEBUG)
  target_compile_definitions(stat_ort_plugin PRIVATE VAANI_DEBUG)
endif()

if (ANDROID)
  # Support Android 15 16k page size
  target_link_options(stat_ort_plugin PRIVATE "-Wl,-z,max-page-size=16384")

  target_include_directories(stat_ort_plugin PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/../android/src/include"
  )

  # libonnxruntime.so is NOT committed to the repo. android/build.gradle
  # extracts it from the onnxruntime-android Gradle dependency at build time
  # and passes its location in ONNXRUNTIME_LIB_DIR. We link against it (so this
  # library gets a DT_NEEDED entry); at runtime the same .so is packaged into
  # the APK by AGP and resolves the reference.
  if (NOT DEFINED ONNXRUNTIME_LIB_DIR)
    message(FATAL_ERROR
        "ONNXRUNTIME_LIB_DIR is not set. The Android build must run through "
        "Gradle (android/build.gradle), which extracts libonnxruntime.so from "
        "the onnxruntime-android AAR and passes this path to CMake.")
  endif()

  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION
          "${ONNXRUNTIME_LIB_DIR}/${ANDROID_ABI}/libonnxruntime.so"
  )

  target_link_libraries(stat_ort_plugin onnxruntime log)
endif()


