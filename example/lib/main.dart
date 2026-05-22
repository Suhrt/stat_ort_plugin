import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:stat_ort_plugin/stat_ort_plugin.dart';
import 'package:stat_ort_plugin_example/load_asset.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  Vaani? vaani;
  String transcript = "";
  bool isProcessing = false;

  @override
  void dispose() {
    // Prevent memory leaks by freeing the C pipeline when the app closes
    vaani?.dispose();
    super.dispose();
  }

  @override
  void initState() {
    init();
    super.initState();
  }

  Future<void> init() async {
    if (isProcessing) return;
    print('loading models');
    loadModel();
  }

  Future<void> loadModel() async {
    Stopwatch stopwatch = Stopwatch()..start();
    final encPath = await loadAsset('assets/encoder-vaani.onnx');
    final decPath = await loadAsset('assets/decoder_joint-vaani.onnx');
    final tokenPath = await loadAsset('assets/tokens.txt');
    print("loaded to memory in ${stopwatch.elapsedMilliseconds}");
    stopwatch.reset();
    vaani = await Vaani.create(encPath, decPath, tokenPath, 4);
    print("model created in ${stopwatch.elapsedMilliseconds}");
    stopwatch.reset();
    final audioPath = await loadAsset('assets/audio.wav');
    final transcript = await vaani!.transcribe(audioPath);
    print("transcription took ${stopwatch.elapsedMilliseconds}");
    print(transcript);
  }

  @override
  Widget build(BuildContext context) {
    const spacerSmall = SizedBox(height: 20);

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Vaani FFI C')),
        body: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(16),
            width: double.infinity,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: isProcessing ? null : init,
                  child: Text(isProcessing ? 'Processing...' : 'Start'),
                ),
                spacerSmall,
                if (isProcessing)
                  const CupertinoActivityIndicator()
                else if (transcript.isNotEmpty)
                  Text(
                    transcript,
                    style: const TextStyle(fontSize: 20),
                    textAlign: TextAlign.center,
                  )
              ],
            ),
          ),
        ),
      ),
    );
  }
}