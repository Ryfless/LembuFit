import 'dart:async';
import 'package:cloud_firestore/cloud_firestore.dart';

class HistoryService {
  final FirebaseFirestore _db = FirebaseFirestore.instance;

  // cache terakhir untuk deteksi perubahan per docId
  final Map<String, Map<String, dynamic>> _lastDocData = {};
  StreamSubscription<QuerySnapshot>? _cowsSub;
  bool _started = false;

  // proteksi: waktu terakhir tulis per cow (throttle)
  final Map<String, DateTime> _lastWriteTime = {};
  final Duration _minWriteInterval = const Duration(seconds: 5);

  /// Idempotent start: jika sudah berjalan, tidak akan membuat listener baru.
  void startListeningToCows() {
    if (_started) return;
    _started = true;

    try {
      _cowsSub = _db.collection('cows').snapshots().listen((snap) {
        try {
          for (final doc in snap.docs) {
            final id = doc.id;
            final data = (doc.data() as Map<String, dynamic>? ?? {});
            final prev = _lastDocData[id];

            final current = <String, dynamic>{
              'temp': data['temp'] ?? '',
              'heart': data['heart'] ?? '',
              'activity': data['activity'] ?? '',
              'status': data['status'] ?? '',
              'name': data['name'] ?? '',
            };

            // first snapshot: cache and skip creating history (avoid initial noise)
            if (prev == null) {
              _lastDocData[id] = Map<String, dynamic>.from(current);
              continue;
            }

            // detect change in tracked fields
            final changed = current.entries.any((e) {
              final key = e.key;
              final prevVal = prev[key]?.toString() ?? '';
              final curVal = e.value?.toString() ?? '';
              return prevVal != curVal;
            });

            if (changed) {
              final now = DateTime.now();
              final last = _lastWriteTime[id];
              if (last != null && now.difference(last) < _minWriteInterval) {
                // skip write because throttled
                _lastDocData[id] = Map<String, dynamic>.from(current);
                continue;
              }
              _lastWriteTime[id] = now;

              // write history entry asynchronously, but do not await inside loop
              unawaited(_writeHistoryEntry(id, current));
              // update cache
              _lastDocData[id] = Map<String, dynamic>.from(current);
            }
          }
        } catch (e) {
          // swallow to keep subscription alive and log minimal info
          // if you have a logger, replace print with logger
          // ignore: avoid_print
          print('HistoryService: listener error: $e');
        }
      }, onError: (err) {
        // subscription error: mark as not started so caller may retry
        // ignore: avoid_print
        print('HistoryService: subscription error: $err');
        _started = false;
      });
    } catch (e) {
      // ignore and reset started flag
      // ignore: avoid_print
      print('HistoryService: start failed: $e');
      _started = false;
    }
  }

  void stop() {
    _cowsSub?.cancel();
    _cowsSub = null;
    _started = false;
    _lastDocData.clear();
    _lastWriteTime.clear();
  }

  Future<void> _writeHistoryEntry(String cowId, Map<String, dynamic> current) async {
    try {
      await _db.collection('cow_histories').add({
        'cowId': cowId,
        'name': current['name'] ?? '',
        'temp': current['temp']?.toString() ?? '',
        'heart': current['heart']?.toString() ?? '',
        'activity': current['activity'] ?? '',
        'status': current['status'] ?? '',
        'raw': current,
        'timestamp': FieldValue.serverTimestamp(),
      });
    } catch (e) {
      // ignore/write to log if you have logging
      // ignore: avoid_print
      print('HistoryService: write error for $cowId: $e');
    }
  }

  /// Stream history entries for a cow ordered by timestamp desc
  Stream<List<Map<String, dynamic>>> streamHistoryForCow(String cowId, {int limit = 200}) {
    final q = _db
        .collection('cow_histories')
        .where('cowId', isEqualTo: cowId)
        .orderBy('timestamp', descending: true)
        .limit(limit);

    return q.snapshots().map((snap) {
      return snap.docs.map((d) {
        final m = d.data() as Map<String, dynamic>;
        m['__id'] = d.id;
        return m;
      }).toList();
    });
  }
}

final historyService = HistoryService();