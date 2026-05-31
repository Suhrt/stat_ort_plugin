part of '../stat_ort_plugin.dart';

// ── Vaani pipeline ──────────────────────────────────────────────────────────

class Vaani {
  final Pointer<Void> _pipeline;
  final VaaniFreeDart _freeFunc;
  // Whether this handle owns the native pipeline (and may free it). Handles
  // rebuilt via [Vaani.fromNativeAddress] in another isolate do not.
  final bool _owns;
  bool _disposed = false;

  Vaani._(this._pipeline, {bool owns = true})
      : _owns = owns,
        _freeFunc =
  _dylib.lookupFunction<VaaniFreeC, VaaniFreeDart>('vaani_pipeline_free');

  /// The native pipeline pointer as an integer address.
  ///
  /// The underlying native memory is process-global, so this address can be
  /// sent to another isolate and rebuilt with [Vaani.fromNativeAddress] to
  /// drive streaming off the UI isolate (see the example's streaming worker).
  int get nativeAddress => _pipeline.address;

  /// Rebuilds a [Vaani] handle in another isolate from a [nativeAddress].
  ///
  /// The returned handle shares the same native pipeline as the original;
  /// [dispose] on it is a no-op, so only the original owner frees the pipeline.
  factory Vaani.fromNativeAddress(int address) =>
      Vaani._(Pointer<Void>.fromAddress(address), owns: false);

  static Future<Vaani> create(
      String encPath,
      String decPath,
      String vocabPath,
      int encThreads, {
        String? vadPath,
        String? speakerPath,
      }) async {
    final initFunc =
    _dylib.lookupFunction<VaaniInitC, VaaniInitDart>('vaani_pipeline_init');

    final address = await Isolate.run(() {
      final encPathPtr = encPath.toNativeUtf8();
      final decPathPtr = decPath.toNativeUtf8();
      final vocabPathPtr = vocabPath.toNativeUtf8();
      final vadPathPtr = vadPath?.toNativeUtf8() ?? nullptr;
      final speakerPathPtr = speakerPath?.toNativeUtf8() ?? nullptr;

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
    // Non-owning handles (from fromNativeAddress) must not free the shared
    // pipeline — only the original owner does.
    if (_disposed || !_owns) return;
    _disposed = true;
    _freeFunc(_pipeline);
  }

  Future<String> transcribe(String wavPath) async {
    if (wavPath.isEmpty) return '';

    final transcribeFunc = _dylib
        .lookupFunction<VaaniTranscribeC, VaaniTranscribeDart>('vaani_pipeline_transcribe');

    final pipelineAddress = _pipeline.address;

    return await Isolate.run(() {
      final pipelinePtr = Pointer<Void>.fromAddress(pipelineAddress);
      final wavPathPtr = wavPath.toNativeUtf8();
      final resultPtr = transcribeFunc(pipelinePtr, wavPathPtr);
      calloc.free(wavPathPtr);
      if (resultPtr == nullptr) return '';
      final result = resultPtr.toDartString();
      _stringFree(resultPtr);
      return result;
    });
  }

  VaaniStream createStream({String? outWavPath}) {
    final initStreamFunc = _dylib
        .lookupFunction<VaaniStreamInitC, VaaniStreamInitDart>('vaani_stream_init');

    final outWavPathPtr = outWavPath?.toNativeUtf8() ?? nullptr;
    final streamStatePtr = initStreamFunc(_pipeline, outWavPathPtr);
    if (outWavPathPtr != nullptr) calloc.free(outWavPathPtr);

    if (streamStatePtr == nullptr) {
      throw Exception('vaani_stream_init returned null.');
    }
    return VaaniStream._(streamStatePtr);
  }
}
