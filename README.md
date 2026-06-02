# stat_ort_plugin

On-device speech recognition (**Vaani ASR**) for Flutter via `dart:ffi`, powered by
[ONNX Runtime](https://onnxruntime.ai/). It provides both **file transcription** and
**real-time streaming** transcription, with [Silero VAD](https://github.com/snakers4/silero-vad)
based segmentation and optional speaker diarization.

> Status: early release (0.0.1). The native pipeline works; the API may still change.

## Features

- 🎙️ Streaming transcription from a microphone PCM stream, segmented by voice activity.
- 📁 File transcription for 16 kHz mono `int16` WAV files.
- 🧑‍🤝‍🧑 Optional speaker diarization (speaker-labelled segments).
- ⚡ Native C pipeline; heavy work runs off the UI isolate.

## Supported platforms

| Platform | Status | ONNX Runtime source |
|----------|--------|---------------------|
| Android  | ✅      | `com.microsoft.onnxruntime:onnxruntime-android` (Gradle) |
| iOS      | ✅      | `onnxruntime-c` (CocoaPods) |

macOS / Windows / Linux are not currently supported.

## Models

This package does **not** ship any models. You supply ONNX model files and a vocabulary
at runtime:

* ASR encoder ([`encoder-vaani.onnx`](https://www.google.com/search?q=%5Bhttps://stathealth.in/vaani/encoder_vaani.onnx%5D(https://stathealth.in/vaani/encoder_vaani.onnx)))
* ASR decoder/joint ([`decoder_joint-vaani.onnx`](https://www.google.com/search?q=%5Bhttps://stathealth.in/vaani/decoder_joint-vaani.onnx%5D(https://stathealth.in/vaani/decoder_joint-vaani.onnx)))
* Vocabulary ([`tokens.txt`](https://www.google.com/search?q=%5Bhttps://stathealth.in/vaani/tokens.txt%5D(https://stathealth.in/vaani/tokens.txt)))
* *(optional)* Silero VAD ([`silero_vad.onnx`](https://www.google.com/search?q=%5Bhttps://stathealth.in/vaani/silero_vad.onnx%5D(https://stathealth.in/vaani/silero_vad.onnx))) — required for streaming segmentation
* *(optional)* Speaker embedding ([`voxblink2_samresnet34_ft.onnx`](https://www.google.com/search?q=%5Bhttps://stathealth.in/vaani/voxblink2_samresnet34_ft.onnx%5D(https://stathealth.in/vaani/voxblink2_samresnet34_ft.onnx))) — required for diarization

Audio must be **16 kHz, mono, 16-bit PCM**. See [NOTICE.md](NOTICE.md) for model licensing.
The example app loads models from its assets and copies them to a temp directory; for a real
app, download them on first launch rather than bundling hundreds of MB into your binary.

## Installation

Add the dependency (from a git ref or, once published, from pub.dev):

```yaml
dependencies:
  stat_ort_plugin:
    git:
      url: https://github.com/your-org/stat_ort_plugin.git
```

No extra setup is needed: ONNX Runtime is pulled in automatically by the Android Gradle
dependency and the iOS CocoaPod.

## Usage

### File transcription

```dart
import 'package:stat_ort_plugin/stat_ort_plugin.dart';

final vaani = await Vaani.create(
  encoderPath, decoderPath, vocabPath, 4, // encoderThreads
  vadPath: vadPath,
  speakerPath: speakerPath,
);

final transcript = await vaani.transcribe(wavPath);
print(transcript);

vaani.dispose();
```

### Streaming transcription

Feed raw 16 kHz mono `int16` PCM as it arrives (e.g. from the `record` package). The
plugin buffers internally into the 512-sample frames Silero VAD requires.

```dart
final stream = vaani.createStream();

await for (final Uint8List chunk in micPcmStream) {
  final segment = stream.pushChunk(chunk.buffer.asInt16List());
  if (segment != null) print(segment); // a finalised "[mm:ss - mm:ss] [Speaker N]: ..." line
}

// IMPORTANT: flush before closing to get the trailing segment.
final tail = stream.finish();
if (tail != null) print(tail);
stream.close();
```

A complete example (mic streaming + file transcription) is in [`example/`](example/).

## Threading notes

- `Vaani.create` and `transcribe` run the native work on a background isolate.
- A single `Vaani` (pipeline) may back multiple `VaaniStream`s; VAD state is held
  per-stream, so independent streams don't interfere. Do not, however, drive the
  *same* pipeline's inference from multiple threads concurrently without your own
  serialization.
- `vaani_pipeline_init` resolves the ONNX Runtime API once and is safe to call from
  multiple isolates.

## Regenerating the FFI bindings

`lib/stat_ort_plugin_bindings_generated.dart` is generated from `src/stat_ort_plugin.h`
with [`package:ffigen`](https://pub.dev/packages/ffigen). It is committed so the package
builds without a generation step. Regenerate after changing the header:

```sh
dart run ffigen --config ffigen.yaml
```

## Debug logging

The native code logs errors via `LOGE` always, and verbose progress via `LOGD` only when
built with `VAANI_DEBUG` defined (CMake: `-DVAANI_DEBUG=ON`). Release builds carry no
verbose-logging overhead.

## License

MIT — see [LICENSE](LICENSE). Third-party dependencies and models are listed in
[NOTICE.md](NOTICE.md); verify model licenses for your use case before shipping.
