import 'dart:ffi';
import 'dart:io';

import 'stat_ort_plugin_bindings_generated.dart';

bool isOrtInitialized() {
  return _bindings.check_ort_status() == 1;
}

const String _libName = 'stat_ort_plugin';

/// The dynamic library in which the symbols for [StatOrtPluginBindings] can be found.
final DynamicLibrary _dylib = () {
  if (Platform.isIOS || Platform.isMacOS) {
    // On Apple platforms, CocoaPods links frameworks directly or loads them via process symbols.
    return DynamicLibrary.process();
  }
  if (Platform.isAndroid) {
    // ONNX Runtime must be pre-loaded into memory before opening your plugin library.
    DynamicLibrary.open('libonnxruntime.so');
    return DynamicLibrary.open('lib$_libName.so');
  }
  if (Platform.isWindows) {
    // Pre-load the ONNX Runtime DLL before opening your plugin library.
    DynamicLibrary.open('onnxruntime.dll');
    return DynamicLibrary.open('$_libName.dll');
  }
  if (Platform.isLinux) {
    DynamicLibrary.open('libonnxruntime.so');
    return DynamicLibrary.open('lib$_libName.so');
  }
  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

/// The bindings to the native functions in [_dylib].
final StatOrtPluginBindings _bindings = StatOrtPluginBindings(_dylib);