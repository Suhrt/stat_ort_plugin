part of '../stat_ort_plugin.dart';

// ── FFI typedefs ────────────────────────────────────────────────────────────

typedef VaaniInitC = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Int32);
typedef VaaniInitDart = Pointer<Void> Function(
    Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, int);

typedef VaaniFreeC = Void Function(Pointer<Void>);
typedef VaaniFreeDart = void Function(Pointer<Void>);

typedef VaaniTranscribeC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);
typedef VaaniTranscribeDart = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);

typedef VaaniStringFreeC = Void Function(Pointer<Utf8>);
typedef VaaniStringFreeDart = void Function(Pointer<Utf8>);

typedef VaaniStreamInitC = Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>);
typedef VaaniStreamInitDart = Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>);

typedef VaaniStreamPushChunkC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int16>, Int32);
typedef VaaniStreamPushChunkDart = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int16>, int);

typedef VaaniStreamFlushC = Pointer<Utf8> Function(Pointer<Void>);
typedef VaaniStreamFlushDart = Pointer<Utf8> Function(Pointer<Void>);

typedef VaaniStreamCloseC = Void Function(Pointer<Void>);
typedef VaaniStreamCloseDart = void Function(Pointer<Void>);

const String _libName = 'stat_ort_plugin';

// Open a shared library, turning the low-level loader failure into a clear,
// actionable message instead of a later "symbol not found" on first FFI call.
DynamicLibrary _open(String name) {
  try {
    return DynamicLibrary.open(name);
  } catch (e) {
    throw StateError(
        'Failed to load native library "$name". Ensure ONNX Runtime and the '
        'stat_ort_plugin native code are bundled for this platform. '
        'Original error: $e');
  }
}

final DynamicLibrary _dylib = () {
  // onnxruntime is statically present in the process on iOS/macOS (via the
  // CocoaPod static framework); elsewhere it must be loaded before the plugin
  // so the plugin's undefined ORT symbols resolve against it.
  if (Platform.isIOS || Platform.isMacOS) return DynamicLibrary.process();
  if (Platform.isAndroid) {
    _open('libonnxruntime.so');
    return _open('lib$_libName.so');
  }
  if (Platform.isWindows) {
    _open('onnxruntime.dll');
    return _open('$_libName.dll');
  }
  if (Platform.isLinux) {
    _open('libonnxruntime.so');
    return _open('lib$_libName.so');
  }
  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

// Cache function lookups once at load time. Looking these up on every chunk
// push was the previous approach — adds noticeable overhead at 16 kHz / 32 ms
// frame cadence.
final _stringFree = _dylib
    .lookupFunction<VaaniStringFreeC, VaaniStringFreeDart>('vaani_string_free');

final _streamPush = _dylib
    .lookupFunction<VaaniStreamPushChunkC, VaaniStreamPushChunkDart>(
    'vaani_stream_push_chunk');

final _streamFlush = _dylib
    .lookupFunction<VaaniStreamFlushC, VaaniStreamFlushDart>('vaani_stream_flush');

final _streamClose = _dylib
    .lookupFunction<VaaniStreamCloseC, VaaniStreamCloseDart>('vaani_stream_close');
