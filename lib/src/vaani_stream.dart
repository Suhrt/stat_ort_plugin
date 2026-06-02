part of '../stat_ort_plugin.dart';

/// A real-time streaming transcription session, created by [Vaani.createStream].
///
/// Silero VAD inside the native plugin only accepts 512-sample frames at
/// 16 kHz. This class buffers incoming PCM and feeds the native side in
/// 512-sample chunks, dropping any fractional tail until either the next
/// [pushChunk] fills it or [finish] is called.
///
/// [finish] must be called before [close] to retrieve any in-progress
/// segment. [close] alone will drop a trailing segment.
class VaaniStream implements Finalizable {
  // Native VAD frame size. Do not change without changing the C side.
  static const int _frameSamples = 512;

  final Pointer<VaaniStreamState> _streamState;

  // Native scratch buffer for one VAD frame. Allocated once, reused per push.
  // Avoids per-chunk malloc/free in the FFI hot path.
  final Pointer<Int16> _frameBuf;

  // Carry-over for partial frames between pushChunk calls. Capacity is
  // strictly < _frameSamples; once we have 512 we send them through.
  final Int16List _carry = Int16List(_frameSamples);
  int _carryLen = 0;

  bool _closed = false;

  VaaniStream._(this._streamState) : _frameBuf = calloc<Int16>(_frameSamples) {
    // Backstops: free the native stream state and the scratch buffer on GC if
    // the caller forgets close(). close() detaches both first.
    _streamFinalizer.attach(this, _streamState.cast(), detach: this);
    _callocFinalizer?.attach(this, _frameBuf.cast(), detach: this);
  }

  /// Push raw 16 kHz mono PCM (int16). Returns any finalised segment text
  /// that the native side emitted during this push, or null if nothing
  /// closed in this call. Multiple segments concatenate into one string.
  ///
  /// Callers should keep calling this with new audio and accumulating the
  /// returned strings; the actual end-of-utterance decision is made by the
  /// native VAD, not by chunk boundaries.
  String? pushChunk(Int16List pcmData) {
    if (_closed) {
      throw StateError('VaaniStream already closed');
    }
    if (pcmData.isEmpty) return null;

    final sb = StringBuffer();
    int srcOffset = 0;
    final srcLen = pcmData.length;

    // Top up the carry first if there's a partial frame waiting.
    if (_carryLen > 0) {
      final need = _frameSamples - _carryLen;
      final take = need < srcLen ? need : srcLen;
      _carry.setRange(_carryLen, _carryLen + take, pcmData, 0);
      _carryLen += take;
      srcOffset += take;

      if (_carryLen == _frameSamples) {
        _sendFrame(_carry, 0, sb);
        _carryLen = 0;
      }
    }

    // Send as many full 512-sample frames as we have, directly from pcmData.
    while (srcOffset + _frameSamples <= srcLen) {
      _sendFrame(pcmData, srcOffset, sb);
      srcOffset += _frameSamples;
    }

    // Stash whatever's left (< 512 samples) into the carry buffer.
    final remaining = srcLen - srcOffset;
    if (remaining > 0) {
      _carry.setRange(0, remaining, pcmData, srcOffset);
      _carryLen = remaining;
    }

    final out = sb.toString();
    return out.isEmpty ? null : out;
  }

  /// Push raw little-endian int16 PCM bytes exactly as a recorder delivers
  /// them (e.g. the `record` package's `startStream`), so callers don't have
  /// to decode bytes to [Int16List] themselves. Tolerates a non-2-byte-aligned
  /// byte offset. Otherwise identical to [pushChunk].
  String? pushChunkBytes(Uint8List bytes) => pushChunk(_bytesToInt16(bytes));

  // Decode little-endian int16 PCM bytes into an Int16List. Uses a zero-copy
  // view when the byte offset is 2-aligned (the common case) and falls back to
  // a byte-wise copy when it isn't (some platform audio buffers are unaligned).
  static Int16List _bytesToInt16(Uint8List data) {
    final int16Length = data.lengthInBytes ~/ 2;
    if (data.offsetInBytes % 2 == 0) {
      return data.buffer.asInt16List(data.offsetInBytes, int16Length);
    }
    final out = Int16List(int16Length);
    final view = ByteData.sublistView(data);
    for (int i = 0; i < int16Length; i++) {
      out[i] = view.getInt16(i * 2, Endian.little);
    }
    return out;
  }

  // Copy one 512-sample frame into the FFI scratch buffer and push it.
  void _sendFrame(Int16List src, int srcOffset, StringBuffer sb) {
    final dst = _frameBuf.asTypedList(_frameSamples);
    // Single copy from Dart heap to native — replaces the previous
    // sublist + Int16List.fromList + setAll triple copy.
    dst.setRange(0, _frameSamples, src, srcOffset);

    final resultPtr = _bindings.vaani_stream_push_chunk(
      _streamState,
      _frameBuf,
      _frameSamples,
    );
    if (resultPtr != nullptr) {
      final s = resultPtr.cast<Utf8>().toDartString();
      _bindings.vaani_string_free(resultPtr);
      if (s.isNotEmpty) sb.write(s);
    }
  }

  /// Tell the native side to emit whatever segment it currently has buffered,
  /// and return that final text. Call this once when the user stops talking,
  /// before [close]. Discards any sub-frame carry (< 512 samples) — that
  /// tail is too short to matter and would only contain trailing silence.
  ///
  /// Safe to call multiple times; subsequent calls return null.
  String? finish() {
    if (_closed) return null;

    // Drop the sub-frame carry. Padding it with zeros would just feed
    // silence into VAD and dilute the trailing audio's energy. The
    // native side's flush operates on whatever full frames it's already
    // received, which is what we want.
    _carryLen = 0;

    final resultPtr = _bindings.vaani_stream_flush(_streamState);
    if (resultPtr == nullptr) return null;
    final s = resultPtr.cast<Utf8>().toDartString();
    _bindings.vaani_string_free(resultPtr);
    return s.isEmpty ? null : s;
  }

  /// Tear down the native stream and free the FFI scratch buffer.
  /// Call [finish] first if you want the trailing segment.
  void close() {
    if (_closed) return;
    _closed = true;
    _streamFinalizer.detach(this);
    _callocFinalizer?.detach(this);
    _bindings.vaani_stream_close(_streamState);
    calloc.free(_frameBuf);
  }
}
