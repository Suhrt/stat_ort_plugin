import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

Future<String> loadAsset(String assetPath) async {
  final byteData = await rootBundle.load(assetPath);

  final dir = await getTemporaryDirectory();
  final file = File(p.join(dir.path, p.basename(assetPath)));

  if (await file.exists() && await file.length() == byteData.lengthInBytes) {
    return file.path;
  }

  await file.writeAsBytes(
    byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes),
    flush: true,
  );

  return file.path;
}