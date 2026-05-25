import 'dart:io';
import 'dart:typed_data';
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
  VaaniStream? vaaniStream;
  String transcript = "";
  bool isProcessing = false;
  bool isStreaming = false;

  @override
  void dispose() {
    isStreaming = false;
    vaaniStream?.close();
    vaani?.dispose();
    super.dispose();
  }

  Future<void> _initVaani() async {
    if (vaani != null) return;

    // Load base models
    final encPath = await loadAsset('assets/encoder-vaani.onnx');
    final decPath = await loadAsset('assets/decoder_joint-vaani.onnx');
    final tokenPath = await loadAsset('assets/tokens.txt');

    // Load VAD and Diarization models
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
  }

  Future<void> transcribeFile() async {
    setState(() {
      isProcessing = true;
      transcript = "Processing file with Diarization...\n";
    });

    try {
      await _initVaani();
      final audioPath = await loadAsset('assets/audio.wav');
      final result = await vaani!.transcribe(audioPath);

      setState(() {
        transcript = result.isEmpty ? "No speech detected." : result;
      });
    } catch (e) {
      setState(() {
        transcript = "Error: $e";
      });
    } finally {
      setState(() {
        isProcessing = false;
      });
    }
  }

  Future<void> startStreamSimulation() async {
    setState(() {
      isStreaming = true;
      isProcessing = true;
      transcript = "Starting Live Stream...\n";
    });

    try {
      await _initVaani();

      // Initialize stream
      vaaniStream = vaani!.createStream(outWavPath: null);

      final audioPath = await loadAsset('assets/audio.wav');
      final file = File(audioPath);
      final bytes = await file.readAsBytes();

      // Skip 44-byte WAV header
      final pcmBytes = bytes.sublist(44);
      final pcmData = pcmBytes.buffer.asInt16List(pcmBytes.offsetInBytes, pcmBytes.lengthInBytes ~/ 2);

      const chunkSize = 512; // 30ms chunks at 16kHz

      for (int i = 0; i < pcmData.length; i += chunkSize) {
        if (!isStreaming) break;

        int end = i + chunkSize;
        if (end > pcmData.length) end = pcmData.length;

        final chunk = pcmData.sublist(i, end);
        final segmentText = await vaaniStream!.pushChunk(chunk);

        // Output contains [MM:SS.ms - MM:SS.ms] [Speaker X]: Text
        if (segmentText != null && segmentText.isNotEmpty) {
          setState(() {
            transcript += segmentText;
          });
        }

        await Future.delayed(const Duration(milliseconds: 30));
      }
    } catch (e) {
      setState(() {
        transcript += "\nStream Error: $e";
      });
    } finally {
      vaaniStream?.close();
      vaaniStream = null;
      setState(() {
        isStreaming = false;
        isProcessing = false;
      });
    }
  }

  void stopStream() {
    setState(() {
      isStreaming = false;
    });
  }

  @override
  Widget build(BuildContext context) {
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
                    ElevatedButton(
                      onPressed: isProcessing ? null : transcribeFile,
                      child: const Text('Transcribe File'),
                    ),
                    const SizedBox(width: 10),
                    ElevatedButton(
                      onPressed: isProcessing ? stopStream : startStreamSimulation,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: isStreaming ? Colors.red : null,
                        foregroundColor: isStreaming ? Colors.white : null,
                      ),
                      child: Text(isStreaming ? 'Stop Stream' : 'Simulate Stream'),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                if (isProcessing && transcript.endsWith("...\n"))
                  const CupertinoActivityIndicator(),
                Container(
                  alignment: Alignment.topLeft,
                  padding: const EdgeInsets.all(12),
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