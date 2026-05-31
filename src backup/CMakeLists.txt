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
  "vaani_stream.c"
  "vaani_utils.c"
)

set_target_properties(stat_ort_plugin PROPERTIES
  PUBLIC_HEADER stat_ort_plugin.h
  OUTPUT_NAME "stat_ort_plugin"

)

target_compile_definitions(stat_ort_plugin PUBLIC DART_SHARED_LIB)

target_compile_options(stat_ort_plugin PRIVATE -O3 -ffast-math)

if (ANDROID)
  # Support Android 15 16k page size
  target_link_options(stat_ort_plugin PRIVATE "-Wl,-z,max-page-size=16384")

  #manual linking after download
  target_include_directories(stat_ort_plugin PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/../android/src/include"
  )

  add_library(onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION
          "${CMAKE_CURRENT_SOURCE_DIR}/../android/src/main/jniLibs/${ANDROID_ABI}/libonnxruntime.so"
  )

  target_link_libraries(stat_ort_plugin onnxruntime log)
endif()


