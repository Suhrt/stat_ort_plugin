import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data'; // ADDED: For Int16List
import 'package:ffi/ffi.dart';

// CHANGED: Added two Pointer<Utf8> for vad_path and speaker_path
typedef VaaniInitC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Int32);
typedef VaaniInitDart = Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, int);

typedef VaaniFreeC = Void Function(Pointer<Void>);
typedef VaaniFreeDart = void Function(Pointer<Void>);

typedef VaaniTranscribeC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);
typedef VaaniTranscribeDart = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);

typedef VaaniStringFreeC = Void Function(Pointer<Utf8>);
typedef VaaniStringFreeDart = void Function(Pointer<Utf8>);

// ADDED: Stream FFI typedefs
typedef VaaniStreamInitC = Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>);
typedef VaaniStreamInitDart = Pointer<Void> Function(Pointer<Void>, Pointer<Utf8>);

typedef VaaniStreamPushChunkC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int16>, Int32);
typedef VaaniStreamPushChunkDart = Pointer<Utf8> Function(Pointer<Void>, Pointer<Int16>, int);

typedef VaaniStreamCloseC = Void Function(Pointer<Void>);
typedef VaaniStreamCloseDart = void Function(Pointer<Void>);

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

  // CHANGED: Added vadPath and speakerPath as optional parameters
  static Future<Vaani> create(
      String encPath,
      String decPath,
      String vocabPath,
      int encThreads, {
        String? vadPath,
        String? speakerPath,
      }) async {
    final initFunc = _dylib.lookupFunction<VaaniInitC, VaaniInitDart>('vaani_pipeline_init');

    final address = await Isolate.run(() {
      final encPathPtr = encPath.toNativeUtf8();
      final decPathPtr = decPath.toNativeUtf8();
      final vocabPathPtr = vocabPath.toNativeUtf8();

      // CHANGED: Handle nullable paths for optional models
      final vadPathPtr = vadPath != null ? vadPath.toNativeUtf8() : nullptr;
      final speakerPathPtr = speakerPath != null ? speakerPath.toNativeUtf8() : nullptr;

      final pipeline = initFunc(
        encPathPtr,
        decPathPtr,
        vocabPathPtr,
        vadPathPtr,
        speakerPathPtr,
        encThreads,
      );

      calloc.free(encPathPtr);
      calloc.free(decPathPtr);
      calloc.free(vocabPathPtr);
      if (vadPathPtr != nullptr) calloc.free(vadPathPtr);
      if (speakerPathPtr != nullptr) calloc.free(speakerPathPtr);

      return pipeline.address;
    });

    if (address == 0) {
      throw Exception('vaani_pipeline_init returned null.');
    }

    return Vaani._(Pointer<Void>.fromAddress(address));
  }

  void dispose() {
    _freeFunc(_pipeline);
  }

  Future<String> transcribe(String wavPath) async {
    if (wavPath.isEmpty) return '';

    final transcribeFunc = _dylib.lookupFunction<VaaniTranscribeC, VaaniTranscribeDart>('vaani_pipeline_transcribe');
    final stringFreeFunc = _dylib.lookupFunction<VaaniStringFreeC, VaaniStringFreeDart>('vaani_string_free');

    final pipelineAddress = _pipeline.address;

    return await Isolate.run(() {
      final pipelinePtr = Pointer<Void>.fromAddress(pipelineAddress);
      final wavPathPtr = wavPath.toNativeUtf8();

      final resultPtr = transcribeFunc(pipelinePtr, wavPathPtr);
      calloc.free(wavPathPtr);

      if (resultPtr == nullptr) return '';

      final result = resultPtr.toDartString();
      stringFreeFunc(resultPtr);
      return result;
    });
  }

  // ADDED: Initialize a continuous audio stream
  VaaniStream createStream({String? outWavPath}) {
    final initStreamFunc = _dylib.lookupFunction<VaaniStreamInitC, VaaniStreamInitDart>('vaani_stream_init');

    final outWavPathPtr = outWavPath != null ? outWavPath.toNativeUtf8() : nullptr;
    final streamStatePtr = initStreamFunc(_pipeline, outWavPathPtr);

    if (outWavPathPtr != nullptr) calloc.free(outWavPathPtr);

    if (streamStatePtr == nullptr) {
      throw Exception('vaani_stream_init returned null.');
    }

    return VaaniStream._(streamStatePtr);
  }
}

// ADDED: New class to manage the continuous audio stream state
class VaaniStream {
  final Pointer<Void> _streamState;

  VaaniStream._(this._streamState);

  Future<String?> pushChunk(Int16List pcmData) async {
    if (pcmData.isEmpty) return null;

    final pushFunc = _dylib.lookupFunction<VaaniStreamPushChunkC, VaaniStreamPushChunkDart>('vaani_stream_push_chunk');
    final stringFreeFunc = _dylib.lookupFunction<VaaniStringFreeC, VaaniStringFreeDart>('vaani_string_free');

    final stateAddress = _streamState.address;

    // Pass raw data list to isolate. Int16List is copied across the boundary.
    return await Isolate.run(() {
      final statePtr = Pointer<Void>.fromAddress(stateAddress);

      final pcmPtr = calloc<Int16>(pcmData.length);
      final pcmList = pcmPtr.asTypedList(pcmData.length);
      pcmList.setAll(0, pcmData);

      final resultPtr = pushFunc(statePtr, pcmPtr, pcmData.length);
      calloc.free(pcmPtr);

      if (resultPtr == nullptr) return null;

      final result = resultPtr.toDartString();
      // Free the C string only if it contains data, or handle empty string
      if (result.isEmpty) {
        stringFreeFunc(resultPtr);
        return null;
      }

      stringFreeFunc(resultPtr);
      return result;
    });
  }

  void close() {
    final closeFunc = _dylib.lookupFunction<VaaniStreamCloseC, VaaniStreamCloseDart>('vaani_stream_close');
    closeFunc(_streamState);
  }
}