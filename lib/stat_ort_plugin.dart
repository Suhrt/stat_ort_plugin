import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'package:ffi/ffi.dart';

typedef VaaniInitC =
Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Int32, Int32);
typedef VaaniInitDart =
Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, int, int);

typedef VaaniFreeC = Void Function(Pointer<Void>);
typedef VaaniFreeDart = void Function(Pointer<Void>);

typedef VaaniTranscribeC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);
typedef VaaniTranscribeDart = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);

typedef VaaniStringFreeC = Void Function(Pointer<Utf8>);
typedef VaaniStringFreeDart = void Function(Pointer<Utf8>);

const String _libName = 'stat_ort_plugin';

final DynamicLibrary _dylib = () {
  if (Platform.isIOS || Platform.isMacOS) {
    return DynamicLibrary.process();
  }
  if (Platform.isAndroid) {
    DynamicLibrary.open('libonnxruntime.so');
    return DynamicLibrary.open('lib$_libName.so');
  }
  if (Platform.isWindows) {
    DynamicLibrary.open('onnxruntime.dll');
    return DynamicLibrary.open('$_libName.dll');
  }
  if (Platform.isLinux) {
    DynamicLibrary.open('libonnxruntime.so');
    return DynamicLibrary.open('lib$_libName.so');
  }
  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

class Vaani {
  late final Pointer<Void> _pipeline;
  late final VaaniFreeDart _freeFunc;

  Vaani._(this._pipeline) {
    _freeFunc = _dylib.lookupFunction<VaaniFreeC, VaaniFreeDart>('vaani_pipeline_free');
  }

  static Future<Vaani> create(
      String encPath,
      String decPath,
      String vocabPath,
      int encThreads,
      int decThreads,
      ) async {
    // Look up the function on the current isolate — _dylib is valid here.
    // DynamicLibrary is NOT sendable across isolate boundaries, so we must
    // resolve the native function pointer before entering Isolate.run.
    final initFunc = _dylib.lookupFunction<VaaniInitC, VaaniInitDart>(
      'vaani_pipeline_init',
    );

    final address = await Isolate.run(() {
      final encPathPtr   = encPath.toNativeUtf8();
      final decPathPtr   = decPath.toNativeUtf8();
      final vocabPathPtr = vocabPath.toNativeUtf8();

      final pipeline = initFunc(
        encPathPtr,
        decPathPtr,
        vocabPathPtr,
        encThreads,
        decThreads,
      );

      calloc.free(encPathPtr);
      calloc.free(decPathPtr);
      calloc.free(vocabPathPtr);

      return pipeline.address;
    });

    if (address == 0) {
      throw Exception(
        'vaani_pipeline_init returned null — check model paths and logs (adb logcat -s VaaniCPlugin:I)',
      );
    }

    return Vaani._(Pointer<Void>.fromAddress(address));
  }

  void dispose() {
    _freeFunc(_pipeline);
  }

  Future<String> transcribe(String wavPath) async {
    if (wavPath.isEmpty) return '';

    // Look up both functions on the current isolate before crossing the
    // isolate boundary. The resolved function pointers (plain integers) are
    // safe to capture in the closure; DynamicLibrary itself is not.
    final transcribeFunc = _dylib.lookupFunction<VaaniTranscribeC, VaaniTranscribeDart>(
      'vaani_pipeline_transcribe',
    );
    final stringFreeFunc = _dylib.lookupFunction<VaaniStringFreeC, VaaniStringFreeDart>(
      'vaani_string_free',
    );

    // Pass the pipeline as a raw integer — Pointer is not sendable either.
    final pipelineAddress = _pipeline.address;

    return await Isolate.run(() {
      final pipelinePtr = Pointer<Void>.fromAddress(pipelineAddress);
      final wavPathPtr  = wavPath.toNativeUtf8();

      final resultPtr = transcribeFunc(pipelinePtr, wavPathPtr);
      calloc.free(wavPathPtr);

      if (resultPtr == nullptr) return '';

      final result = resultPtr.toDartString();
      stringFreeFunc(resultPtr);
      return result;
    });
  }
}