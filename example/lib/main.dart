import 'package:flutter/material.dart';
import 'package:stat_ort_plugin/stat_ort_plugin.dart' as stat_ort_plugin;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  // Changed: Replaced sum variables with a boolean for ORT status
  late bool isOrtReady;

  @override
  void initState() {
    super.initState();
    // Changed: Call the ONNX Runtime initialization check
    try {
      isOrtReady = stat_ort_plugin.isOrtInitialized();
    } catch (e, stacktrace) {
      isOrtReady = false;
      debugPrint('ORT INITIALIZATION ERROR: $e');
      debugPrint('STACKTRACE: $stacktrace');
      isOrtReady = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    const textStyle = TextStyle(fontSize: 25);
    const spacerSmall = SizedBox(height: 10);

    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('ORT Native Plugin')),
        body: SingleChildScrollView(
          child: Container(
            // Corrected: .all to EdgeInsets.all
            padding: const EdgeInsets.all(10),
            child: Column(
              children: [
                const Text(
                  'ONNX Runtime Status:',
                  style: textStyle,
                  // Corrected: .center to TextAlign.center
                  textAlign: TextAlign.center,
                ),
                spacerSmall,
                // Changed: Display ORT status instead of sum calculations
                Text(
                  isOrtReady ? '✅ Initialized Successfully' : '❌ Initialization Failed',
                  style: textStyle,
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}