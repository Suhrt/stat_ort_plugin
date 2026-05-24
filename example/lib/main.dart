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
    vaani?.dispose();
    super.dispose();
  }


  Future<void> transcribe() async {
    //Load models from download/asset delivery. DO NOT load from asset
    final encPath = await loadAsset('assets/encoder-vaani.onnx');
    final decPath = await loadAsset('assets/decoder_joint-vaani.onnx');
    final tokenPath = await loadAsset('assets/tokens.txt');

    vaani = await Vaani.create(encPath, decPath, tokenPath, 4);
    final audioPath = await loadAsset('assets/audio.wav');
    transcript = await vaani!.transcribe(audioPath);

    setState(() {
      isProcessing = false;
    });
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
                  onPressed: isProcessing ? null : transcribe,
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