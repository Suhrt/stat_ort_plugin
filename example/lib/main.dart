import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:stat_ort_plugin/stat_ort_plugin.dart';
import 'package:stat_ort_plugin_example/load_asset.dart';
import 'package:stat_ort_plugin_example/streaming_session.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final AudioRecorder _audioRecorder = AudioRecorder();

  Vaani? vaani;
  StreamingSession? _session;
  StreamSubscription<Uint8List>? _recordSub;
  String transcript = "";

  // Models load (hundreds of MB) before anything can run, so the UI stays
  // disabled with a clear status until they're ready. _startingStream covers
  // the brief window between tapping Start and the mic actually capturing —
  // the user shouldn't speak until that clears.
  bool _modelReady = false;
  bool _modelLoading = false;
  bool _transcribing = false;
  bool _streaming = false;
  bool _startingStream = false;

  @override
  void initState() {
    super.initState();
    _loadModels();
  }

  @override
  void dispose() {
    _recordSub?.cancel();
    _audioRecorder.dispose();
    _session?.abort();
    vaani?.dispose();
    super.dispose();
  }

  Future<void> _loadModels() async {
    setState(() => _modelLoading = true);
    try {
      final encPath = await loadAsset('assets/encoder-vaani.onnx');
      final decPath = await loadAsset('assets/decoder_joint-vaani.onnx');
      final tokenPath = await loadAsset('assets/tokens.txt');
      final vadPath = await loadAsset('assets/silero_vad.onnx');
      final speakerPath = await loadAsset('assets/voxblink2_samresnet34_ft.onnx');

      vaani = await Vaani.create(
        encPath,
        decPath,
        tokenPath,
        4,
        vadPath: vadPath,
        speakerPath: speakerPath,
      );
      if (mounted) setState(() => _modelReady = true);
    } catch (e) {
      if (mounted) setState(() => transcript = "Failed to load models: $e");
    } finally {
      if (mounted) setState(() => _modelLoading = false);
    }
  }

  Future<void> transcribeFile() async {
    setState(() {
      _transcribing = true;
      transcript = "";
    });

    try {
      final audioPath = await loadAsset('assets/audio.wav');
      final result = await vaani!.transcribe(audioPath);
      setState(() => transcript = result.isEmpty ? "No speech detected." : result);
    } catch (e) {
      setState(() => transcript = "Error: $e");
    } finally {
      setState(() => _transcribing = false);
    }
  }

  Future<void> startMicStream() async {
    if (!await _audioRecorder.hasPermission()) {
      if (mounted) setState(() => transcript = "Microphone permission denied.");
      return;
    }

    setState(() {
      _streaming = true;
      _startingStream = true;
      transcript = "";
    });

    final directory = await getApplicationDocumentsDirectory();
    final String outPath = '${directory.path}/debug_recording.wav';

    // Drive the VaaniStream on a background isolate. The recorder must run on
    // the main isolate (platform plugin), but inference must NOT — decoding a
    // segment on the UI thread blocks it for seconds and ANRs the app.
    final session = await StreamingSession.start(
      pipelineAddress: vaani!.nativeAddress,
      outWavPath: outPath,
    );
    _session = session;

    session.transcripts.listen((line) {
      if (mounted) setState(() => transcript += line);
    });

    final stream = await _audioRecorder.startStream(const RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: 16000,
      numChannels: 1,
      androidConfig: AndroidRecordConfig(
        audioSource: AndroidAudioSource.voiceRecognition,
      ),
      iosConfig: IosRecordConfig(
        allowHapticsAndSystemSoundsDuringRecording: false,
        categoryOptions: []
       )
    ));

    // Forward raw recorder bytes straight to the worker — the plugin's
    // pushChunkBytes handles the int16 decoding. Cheap, so the UI stays
    // responsive even while the worker decodes a segment.
    _recordSub = stream.listen((data) => _session?.pushBytes(data));

    // The mic is now actually capturing — only now is it safe to speak.
    if (mounted) setState(() => _startingStream = false);
  }

  Future<void> stopStream() async {
    if (await _audioRecorder.isRecording()) {
      await _audioRecorder.stop();
    }
    await _recordSub?.cancel();
    _recordSub = null;

    final session = _session;
    _session = null;
    // Flush the trailing segment; the worker delivers it on the transcripts
    // stream before this completes.
    await session?.finish();

    debugPrint(transcript);
    if (mounted) {
      setState(() {
        _streaming = false;
        _startingStream = false;
      });
    }
  }

  // Single source of truth for the status banner.
  ({String text, Color color, bool busy}) _status() {
    if (!_modelReady) {
      return _modelLoading
          ? (text: 'Loading models…', color: Colors.orange, busy: true)
          : (text: 'Models failed to load', color: Colors.red, busy: false);
    }
    if (_transcribing) {
      return (text: 'Transcribing file…', color: Colors.orange, busy: true);
    }
    if (_streaming && _startingStream) {
      return (text: 'Starting microphone…', color: Colors.orange, busy: true);
    }
    if (_streaming) {
      return (text: '● Listening — speak now', color: Colors.red, busy: false);
    }
    return (text: 'Ready', color: Colors.green, busy: false);
  }

  @override
  Widget build(BuildContext context) {
    final status = _status();
    // While streaming, the only allowed action is Stop (disabled until the mic
    // is live). Otherwise actions require the models to be ready and idle.
    final idle = _modelReady && !_transcribing && !_streaming;

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Vaani Diarization Stream')),
        body: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(16),
            width: double.infinity,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    if (status.busy) ...[
                      const CupertinoActivityIndicator(),
                      const SizedBox(width: 8),
                    ],
                    Text(
                      status.text,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: status.color,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: idle ? transcribeFile : null,
                      child: const Text('Transcribe File'),
                    ),
                    const SizedBox(width: 10),
                    ElevatedButton(
                      onPressed: _streaming
                          ? (_startingStream ? null : stopStream)
                          : (idle ? startMicStream : null),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _streaming ? Colors.red : null,
                        foregroundColor: _streaming ? Colors.white : null,
                      ),
                      child: Text(_streaming ? 'Stop Stream' : 'Start Mic Stream'),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                Container(
                  alignment: Alignment.topLeft,
                  padding: const EdgeInsets.all(12),
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: Colors.grey.shade100,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    transcript,
                    style: const TextStyle(fontSize: 16, fontFamily: 'monospace'),
                  ),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}