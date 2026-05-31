/// On-device Vaani ASR for Flutter via `dart:ffi`.
///
/// The public API is [Vaani] (pipeline lifecycle + file transcription) and
/// [VaaniStream] (real-time streaming). The implementation is split across the
/// `src/` part files below; all imports live here so the parts can share the
/// private FFI bindings.
library;

import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

part 'src/bindings.dart';
part 'src/vaani.dart';
part 'src/vaani_stream.dart';
