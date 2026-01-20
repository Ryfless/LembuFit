import 'dart:async';
import 'dart:math';
import 'package:firebase_database/firebase_database.dart';
import 'firestore_service.dart';

class RealtimeSensorService {
  final FirebaseDatabase _db = FirebaseDatabase.instance;
  final FirestoreService _firestoreService;

  DatabaseReference get _tempRef => _db.ref('gy906/temperature_object');
  DatabaseReference get _heartRef => _db.ref('max30102/avg_bpm');
  DatabaseReference get _accelRef => _db.ref('mpu6050/accel');
  DatabaseReference get _gyroRef => _db.ref('mpu6050/gyro');

  StreamSubscription<DatabaseEvent>? _subTemp;
  StreamSubscription<DatabaseEvent>? _subHeart;
  StreamSubscription<DatabaseEvent>? _subAccel;
  StreamSubscription<DatabaseEvent>? _subGyro;

  double _prevA = 0;
  double _prevG = 0;
  double? _lastTemp;
  double? _lastHeart;
  double? _lastATotal;
  double? _lastGTotal;
  String _lastActivity = 'Diam';

  RealtimeSensorService(this._firestoreService);

  void start() {
    _subTemp = _tempRef.onValue.listen((e) {
      final val = e.snapshot.value;
      double? temp = _extractNumeric(val);
      if (temp != null) {
        // only update when changed (small tolerance)
        if (_lastTemp == null || (temp - _lastTemp!).abs() > 0.01) {
          _lastTemp = temp;
          _updateFirestore();
        }
      }
    }, onError: (err) {});

    _subHeart = _heartRef.onValue.listen((e) {
      final val = e.snapshot.value;
      double? bpm = _extractNumeric(val);
      if (bpm != null) {
        if (_lastHeart == null || (bpm - _lastHeart!).abs() > 0.5) {
          _lastHeart = bpm;
          _updateFirestore();
        }
      }
    }, onError: (err) {});

    _subAccel = _accelRef.onValue.listen((e) {
      final v = e.snapshot.value;
      final map = _toMapNum(v);
      if (map != null) {
        final ax = map['ax'] ?? map['x'] ?? 0.0;
        final ay = map['ay'] ?? map['y'] ?? 0.0;
        final az = map['az'] ?? map['z'] ?? 0.0;
        _handleMotion(ax, ay, az, isAccel: true);
      }
    }, onError: (err) {});

    _subGyro = _gyroRef.onValue.listen((e) {
      final v = e.snapshot.value;
      final map = _toMapNum(v);
      if (map != null) {
        final gx = map['gx'] ?? map['x'] ?? 0.0;
        final gy = map['gy'] ?? map['y'] ?? 0.0;
        final gz = map['gz'] ?? map['z'] ?? 0.0;
        _handleMotion(gx, gy, gz, isAccel: false);
      }
    }, onError: (err) {});
  }

  void stop() {
    _subTemp?.cancel();
    _subHeart?.cancel();
    _subAccel?.cancel();
    _subGyro?.cancel();
  }

  double? _extractNumeric(dynamic v) {
    // try direct numeric or parseable string
    if (v == null) return null;
    if (v is num) return v.toDouble();
    if (v is String) {
      final parsed = double.tryParse(v);
      if (parsed != null) return parsed;
      // try to find number inside string
      final m = RegExp(r'[-+]?[0-9]*\.?[0-9]+').firstMatch(v);
      if (m != null) return double.tryParse(m.group(0)!);
      return null;
    }
    // if Map or List, try to find numeric recursively
    if (v is Map) {
      for (final entry in v.entries) {
        final found = _extractNumeric(entry.value);
        if (found != null) return found;
      }
    }
    if (v is Iterable) {
      for (final item in v) {
        final found = _extractNumeric(item);
        if (found != null) return found;
      }
    }
    return null;
  }

  Map<String, double>? _toMapNum(dynamic v) {
    try {
      if (v is Map) {
        final out = <String, double>{};
        v.forEach((k, val) {
          final d = _extractNumeric(val);
          if (d != null) out[k.toString()] = d;
        });
        return out;
      }
    } catch (_) {}
    return null;
  }

  void _handleMotion(double a1, double a2, double a3, {required bool isAccel}) {
    if (isAccel) {
      final aTotal = sqrt(a1 * a1 + a2 * a2 + a3 * a3);
      _lastATotal = aTotal;
    } else {
      final gTotal = sqrt(a1 * a1 + a2 * a2 + a3 * a3);
      _lastGTotal = gTotal;
    }

    if (_lastATotal != null && _lastGTotal != null) {
      final a_total = _lastATotal!;
      final g_total = _lastGTotal!;
      final prev_a = _prevA;
      final prev_g = _prevG;

      final delta_a = (a_total - prev_a).abs();
      final delta_g = (g_total - prev_g).abs();

      String activity;
      if (delta_a < 0.3 && delta_g < 0.01) {
        activity = "Diam";
      } else if (delta_a < 1.5) {
        activity = "Berjalan";
      } else {
        activity = "Aktif";
      }

      _prevA = a_total;
      _prevG = g_total;
      _lastActivity = activity;

      _updateFirestore();
    }
  }

  Future<void> _updateFirestore() async {
    try {
      final tempStr = _lastTemp != null ? _lastTemp!.toStringAsFixed(2) : null;
      final heartStr = _lastHeart != null ? _lastHeart!.toStringAsFixed(0) : null;
      final activity = _lastActivity;
      // only update if we have at least one sensor value
      if (tempStr == null && heartStr == null && activity.isEmpty) return;
      await _firestoreService.updateAllCowsRealtime(
        temp: tempStr,
        heart: heartStr,
        activity: activity,
      );
    } catch (e) {
      // optional: print for debug
      // print('updateFirestore error: $e');
    }
  }
}