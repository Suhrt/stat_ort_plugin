part of '../stat_ort_plugin.dart';

// ── Native library loading ──────────────────────────────────────────────────

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
      'Original error: $e',
    );
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

// The ffigen-generated bindings, bound to the loaded library. This is a
// top-level lazy final, so each isolate initialises its own copy on first use
// — the worker isolates spawned for init and file transcription re-open the
// library independently, which is why `_bindings` can be referenced freely
// from inside `Isolate.run` closures.
final StatOrtPluginBindings _bindings = StatOrtPluginBindings(_dylib);

// ── GC finalizers (auto-free backstops) ─────────────────────────────────────
//
// If a caller forgets dispose()/close(), these free the native resource when
// the Dart wrapper is garbage-collected. A NativeFinalizer needs the raw C
// function pointer (not the Dart wrapper), so we look the symbols up directly.
// Explicit dispose()/close() detaches first, so the resource is never freed
// twice.

final NativeFinalizer _pipelineFinalizer = NativeFinalizer(
  _dylib
      .lookup<NativeFunction<Void Function(Pointer<VaaniPipeline>)>>(
        'vaani_pipeline_free',
      )
      .cast(),
);

final NativeFinalizer _streamFinalizer = NativeFinalizer(
  _dylib
      .lookup<NativeFunction<Void Function(Pointer<VaaniStreamState>)>>(
        'vaani_stream_close',
      )
      .cast(),
);

// Frees the stream's Dart-owned scratch buffer (calloc'd) on GC. calloc uses
// libc malloc/free, so we free it through the process's `free`. Null if `free`
// can't be resolved on this platform — close() still frees it explicitly.
final NativeFinalizer? _callocFinalizer = () {
  try {
    return NativeFinalizer(
      DynamicLibrary.process()
          .lookup<NativeFunction<Void Function(Pointer<Void>)>>('free')
          .cast(),
    );
  } catch (_) {
    return null;
  }
}();
