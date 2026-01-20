import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

class LocalHistoryService {
  final Map<String, Map<String, dynamic>> _pending = {};
  Timer? _timer;
  final Duration flushInterval;
  final String fileName;

  LocalHistoryService({
    this.flushInterval = const Duration(hours: 1),
    this.fileName = 'cow_history_local.json',
  });

  Future<void> start() async {
    // ensure file exists
    await _localFile();
    _timer ??= Timer.periodic(flushInterval, (_) async => await _flushToFile());
  }

  void recordChange(String cowId, Map<String, dynamic> data) {
    if (cowId.isEmpty) return;
    final copy = Map<String, dynamic>.from(data);
    copy['lastChanged'] = DateTime.now().toIso8601String();
    copy['cowId'] = cowId;
    _pending[cowId] = copy;
  }

  Future<void> flushNow() async {
    await _flushToFile();
  }

  Future<File> _localFile() async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/$fileName');
    if (!await file.exists()) {
      await file.create(recursive: true);
      await file.writeAsString(jsonEncode(<Map<String, dynamic>>[]));
    }
    return file;
  }

  Future<void> _flushToFile() async {
    if (_pending.isEmpty) return;
    try {
      final file = await _localFile();
      final content = await file.readAsString();
      final List<dynamic> entries = (content.isEmpty) ? [] : jsonDecode(content) as List<dynamic>;
      final now = DateTime.now().toIso8601String();

      for (final entry in _pending.values) {
        final record = {
          'timestamp_written': now,
          'timestamp_changed': entry['lastChanged'],
          'cowId': entry['cowId'],
          'data': entry,
        };
        entries.add(record);
      }

      await file.writeAsString(jsonEncode(entries));
      _pending.clear();
    } catch (_) {
      // ignore
    }
  }

  Future<String?> readAll() async {
    try {
      final f = await _localFile();
      return await f.readAsString();
    } catch (_) {
      return null;
    }
  }

  Future<void> stop() async {
    _timer?.cancel();
    _timer = null;
    await _flushToFile();
  }
}

final localHistoryService = LocalHistoryService();