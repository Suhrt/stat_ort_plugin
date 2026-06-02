## 0.0.4

* Switched the FFI layer to the `ffigen`-generated bindings (`StatOrtPluginBindings`)
  as the single source of truth; removed the duplicate hand-written typedefs.
* Reorganized `lib/`: native library loading now lives in `src/native_library.dart`,
  and the generated bindings moved to `src/` (internal, no longer importable from the
  package root).
* Added `NativeFinalizer` backstops to `Vaani` and `VaaniStream`: native resources are
  freed on garbage collection if `dispose()`/`close()` is not called. Explicit
  disposal still works and is preferred.
* `Vaani.transcribe` and `createStream` now throw `StateError` if called after
  `dispose()` instead of using a freed pointer.
* Documentation and formatting: documented `VaaniStream`'s public API and ran
  `dart format`. Fixed broken model-download and example links in the README.
* No public API changes (`Vaani`, `VaaniStream`).

## 0.0.3

Updated read me

## 0.0.2

Updated repo & issue links

## 0.0.1

Initial release.

* On-device Vaani ASR for Flutter via `dart:ffi`, powered by ONNX Runtime.
* File transcription API (`Vaani.transcribe`) for 16 kHz mono PCM WAV.
* Real-time streaming API (`VaaniStream`) with Silero VAD segmentation and
  optional speaker diarization.
* Android and iOS support. ONNX Runtime is provided by the
  `onnxruntime-android` Gradle dependency and the `onnxruntime-c` CocoaPod;
  no native binaries are committed to the repository.
* Heavy work (model init, file transcription) runs on a background isolate.
