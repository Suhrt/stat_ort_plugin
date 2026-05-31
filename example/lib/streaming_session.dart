import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:stat_ort_plugin/stat_ort_plugin.dart';

/// Drives a [VaaniStream] on a background isolate so the UI isolate never
/// blocks on inference. The mic recorder must run on the main isolate (it's a
/// platform plugin), so the main isolate only forwards raw PCM here; all the
/// heavy VAD/ASR/diarization work happens in the worker.
///
/// Usage:
/// ```dart
/// final session = await StreamingSession.start(
///     pipelineAddress: vaani.nativeAddress, outWavPath: path);
/// session.transcripts.listen((line) => setState(() => transcript += line));
/// // for each mic chunk: session.pushChunk(int16Pcm);
/// final tail = await session.finish(); // returns trailing segment, if any
/// ```
class StreamingSession {
  final Isolate _isolate;
  final SendPort _toWorker;
  final ReceivePort _fromWorker;
  final StreamController<String> _transcripts;
  final Completer<void> _done = Completer<void>();

  StreamingSession._(
      this._isolate, this._toWorker, this._fromWorker, this._transcripts);

  /// Finalised segment lines emitted by the worker as they close.
  Stream<String> get transcripts => _transcripts.stream;

  static Future<StreamingSession> start({
    required int pipelineAddress,
    String? outWavPath,
  }) async {
    final fromWorker = ReceivePort();
    final ready = Completer<SendPort>();
    final transcripts = StreamController<String>.broadcast();
    late final StreamingSession session;

    fromWorker.listen((msg) {
      if (msg is SendPort) {
        ready.complete(msg);
      } else if (msg is _Segment) {
        if (!transcripts.isClosed) transcripts.add(msg.text);
      } else if (msg is _Done) {
        if (msg.tail != null && !transcripts.isClosed) {
          transcripts.add(msg.tail!);
        }
        session._finishUp();
      }
    });

    final isolate = await Isolate.spawn(
      _workerMain,
      _Config(fromWorker.sendPort, pipelineAddress, outWavPath),
    );
    final toWorker = await ready.future;

    session = StreamingSession._(isolate, toWorker, fromWorker, transcripts);
    return session;
  }

  /// Forward one chunk of raw 16 kHz mono int16 PCM bytes (as delivered by the
  /// recorder) to the worker, which decodes and processes it off the UI thread.
  void pushBytes(Uint8List bytes) => _toWorker.send(bytes);

  /// Flush the trailing segment and tear down the worker. Completes once the
  /// worker has emitted any final transcript and exited.
  Future<void> finish() async {
    _toWorker.send(const _Finish());
    await _done.future;
  }

  /// Tear down immediately without flushing — for use from `State.dispose()`,
  /// where awaiting isn't possible.
  void abort() => _finishUp();

  void _finishUp() {
    if (_done.isCompleted) return;
    _fromWorker.close();
    _transcripts.close();
    _isolate.kill(priority: Isolate.beforeNextEvent);
    _done.complete();
  }
}

// ── Worker isolate ──────────────────────────────────────────────────────────

void _workerMain(_Config cfg) {
  final port = ReceivePort();
  cfg.toMain.send(port.sendPort);

  final vaani = Vaani.fromNativeAddress(cfg.pipelineAddress);
  final stream = vaani.createStream(outWavPath: cfg.outWavPath);

  port.listen((msg) {
    if (msg is Uint8List) {
      final seg = stream.pushChunkBytes(msg);
      if (seg != null) cfg.toMain.send(_Segment(seg));
    } else if (msg is _Finish) {
      final tail = stream.finish();
      stream.close();
      cfg.toMain.send(_Done(tail));
      port.close();
    }
  });
}

// ── Messages (must be sendable across isolates) ─────────────────────────────

class _Config {
  final SendPort toMain;
  final int pipelineAddress;
  final String? outWavPath;
  const _Config(this.toMain, this.pipelineAddress, this.outWavPath);
}

class _Segment {
  final String text;
  const _Segment(this.text);
}

class _Finish {
  const _Finish();
}

class _Done {
  final String? tail;
  const _Done(this.tail);
}
