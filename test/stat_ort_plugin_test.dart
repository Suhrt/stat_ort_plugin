import 'package:flutter_test/flutter_test.dart';
import 'package:stat_ort_plugin/stat_ort_plugin.dart';

// Exercising transcription requires the native library (ONNX Runtime + the
// plugin's compiled C) and the ONNX model files, so end-to-end behaviour is
// covered by an integration test in the example app, not here. This unit test
// only asserts the pure-Dart surface, which must not trigger the lazy
// DynamicLibrary load.
void main() {
  test('public API types are exported', () {
    expect(Vaani, isNotNull);
    expect(VaaniStream, isNotNull);
  });
}
