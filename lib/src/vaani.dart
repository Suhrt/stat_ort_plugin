part of '../stat_ort_plugin.dart';

// ── Vaani pipeline ──────────────────────────────────────────────────────────

class Vaani implements Finalizable {
  final Pointer<VaaniPipeline> _pipeline;
  // Whether this handle owns the native pipeline (and may free it). Handles
  // rebuilt via [Vaani.fromNativeAddress] in another isolate do not.
  final bool _owns;
  bool _disposed = false;

  Vaani._(this._pipeline, {bool owns = true}) : _owns = owns {
    // Backstop: free the native pipeline on GC if the caller forgets
    // dispose(). Non-owning handles share another handle's pipeline and must
    // never free it, so they are not attached.
    if (_owns) {
      _pipelineFinalizer.attach(this, _pipeline.cast(), detach: this);
    }
  }

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
      Vaani._(Pointer<VaaniPipeline>.fromAddress(address), owns: false);

  static Future<Vaani> create(
    String encPath,
    String decPath,
    String vocabPath,
    int encThreads, {
    String? vadPath,
    String? speakerPath,
  }) async {
    final address = await Isolate.run(() {
      final encPathPtr = encPath.toNativeUtf8();
      final decPathPtr = decPath.toNativeUtf8();
      final vocabPathPtr = vocabPath.toNativeUtf8();
      final vadPathPtr = vadPath?.toNativeUtf8() ?? nullptr;
      final speakerPathPtr = speakerPath?.toNativeUtf8() ?? nullptr;

      final pipeline = _bindings.vaani_pipeline_init(
        encPathPtr.cast(),
        decPathPtr.cast(),
        vocabPathPtr.cast(),
        vadPathPtr.cast(),
        speakerPathPtr.cast(),
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
    return Vaani._(Pointer<VaaniPipeline>.fromAddress(address));
  }

  void dispose() {
    // Non-owning handles (from fromNativeAddress) must not free the shared
    // pipeline — only the original owner does.
    if (_disposed || !_owns) return;
    _disposed = true;
    _pipelineFinalizer.detach(this);
    _bindings.vaani_pipeline_free(_pipeline);
  }

  Future<String> transcribe(String wavPath) async {
    if (_disposed) {
      throw StateError('Vaani has been disposed');
    }
    if (wavPath.isEmpty) return '';

    final pipelineAddress = _pipeline.address;

    return await Isolate.run(() {
      final pipelinePtr = Pointer<VaaniPipeline>.fromAddress(pipelineAddress);
      final wavPathPtr = wavPath.toNativeUtf8();
      final resultPtr = _bindings.vaani_pipeline_transcribe(
        pipelinePtr,
        wavPathPtr.cast(),
      );
      calloc.free(wavPathPtr);
      if (resultPtr == nullptr) return '';
      final result = resultPtr.cast<Utf8>().toDartString();
      _bindings.vaani_string_free(resultPtr);
      return result;
    });
  }

  VaaniStream createStream({String? outWavPath}) {
    if (_disposed) {
      throw StateError('Vaani has been disposed');
    }
    final outWavPathPtr = outWavPath?.toNativeUtf8() ?? nullptr;
    final streamStatePtr = _bindings.vaani_stream_init(
      _pipeline,
      outWavPathPtr.cast(),
    );
    if (outWavPathPtr != nullptr) calloc.free(outWavPathPtr);

    if (streamStatePtr == nullptr) {
      throw Exception('vaani_stream_init returned null.');
    }
    return VaaniStream._(streamStatePtr);
  }
}
