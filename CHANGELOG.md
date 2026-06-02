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
