import 'package:cloud_firestore/cloud_firestore.dart';
import '../models/cow_model.dart';
import 'local_history_service.dart';

class FirestoreService {
  final FirebaseFirestore _db = FirebaseFirestore.instance;
  CollectionReference get _cowsRef => _db.collection('cows');

  // perbaikan: kembalikan Stream<List<Cow>> dengan benar
  Stream<List<Cow>> streamCows() {
    return _cowsRef.snapshots().map((snap) {
      return snap.docs.map((d) => Cow.fromDocument(d)).toList();
    });
  }

  /// Update all cow documents with provided sensor values.
  /// If you want per-cow update, adjust to target a specific doc id.
  Future<void> updateAllCowsRealtime({
    String? temp,
    String? heart,
    String? activity,
  }) async {
    final snap = await _cowsRef.get();
    final batch = _db.batch();

    for (final doc in snap.docs) {
      final data = doc.data() as Map<String, dynamic>? ?? {};
      final existingTemp = (temp ?? data['temp']?.toString() ?? '');
      final existingHeart = (heart ?? data['heart']?.toString() ?? '');
      final existingActivity = (activity ?? data['activity']?.toString() ?? '');
      final status = _computeStatus(
        suhu: existingTemp,
        detakJantung: existingHeart,
        aktivitas: existingActivity,
      );

      final updateMap = {
        'temp': existingTemp,
        'heart': existingHeart,
        'activity': existingActivity,
        'status': status,
        'lastUpdated': FieldValue.serverTimestamp(),
      };

      batch.update(doc.reference, updateMap);

      // record local history (non-blocking, in-memory pending)
      localHistoryService.recordChange(doc.id, {
        'name': data['name'] ?? '',
        'temp': existingTemp,
        'heart': existingHeart,
        'activity': existingActivity,
        'status': status,
      });
    }

    await batch.commit();
  }

  String _computeStatus({
    required String suhu,
    required String detakJantung,
    required String aktivitas,
  }) {
    double suhuVal = double.tryParse(suhu.toString().replaceAll(RegExp(r'[^0-9.]'), '')) ?? 0;
    int detakVal = int.tryParse(detakJantung.toString().replaceAll(RegExp(r'[^0-9]'), '')) ?? 0;
    String aktivitasVal = aktivitas.toLowerCase();

    bool suhuNormal = suhuVal >= 38 && suhuVal <= 39.5;
    bool detakNormal = detakVal >= 40 && detakVal <= 80;
    bool aktivitasNormal = aktivitasVal != 'diam';

    int sakitCount = 0;
    if (suhuVal > 39.5) sakitCount++;
    if (detakVal > 100) sakitCount++;
    if (aktivitasVal == 'diam') sakitCount++;
    if (sakitCount >= 2) return 'Tidak Sehat';

    if (!suhuNormal || !detakNormal) return 'Kurang Sehat';
    if (suhuNormal && detakNormal && !aktivitasNormal) return 'Kurang Aktif';
    if (suhuNormal && detakNormal && aktivitasNormal) return 'Sehat';

    return 'Kurang Sehat';
  }
}

final firestoreService = FirestoreService();