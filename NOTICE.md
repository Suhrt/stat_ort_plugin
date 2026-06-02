# Third-party notices

`stat_ort_plugin` itself is released under the MIT License (see [LICENSE](LICENSE)).
It depends on, and is designed to run, the following third-party software and
models, each under its own license. None of these are bundled in the published
package; they are fetched as dependencies or supplied by the user at runtime.

## Runtime libraries

- **ONNX Runtime** — MIT License. © Microsoft Corporation.
  - Android: `com.microsoft.onnxruntime:onnxruntime-android` (Gradle).
  - iOS: `onnxruntime-c` (CocoaPods).
  - https://github.com/microsoft/onnxruntime

## Models

The example app and any real deployment require ONNX model files that are **not**
distributed with this repository. You must obtain them yourself and comply with
each model's license:

- **Vaani ASR encoder / decoder** (`encoder-vaani.onnx`, `decoder_joint-vaani.onnx`,
  `tokens.txt`) — speech recognition models. Verify and comply with the license of
  the Vaani model you use before redistributing. https://huggingface.co/ARTPARK-IISc/Vaani-FastConformer-Multilingual
- **Silero VAD** (`silero_vad.onnx`) — voice activity detection. MIT License.
  © Silero Team. https://github.com/snakers4/silero-vad
- **Speaker embedding** (`voxblink2_samresnet34_ft.onnx`) — speaker diarization /
  embedding model. Verify and comply with the license of the speaker-embedding
  model you use.

> ⚠️ Confirm the license terms of the ASR and speaker-embedding models for your
> intended use (commercial vs. research) before shipping them in an application.
