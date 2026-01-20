import 'dart:async';
import 'package:flutter/material.dart';
import 'package:firebase_database/firebase_database.dart';

class DebugRealtimePage extends StatefulWidget {
  const DebugRealtimePage({super.key});
  @override
  State<DebugRealtimePage> createState() => _DebugRealtimePageState();
}

class _DebugRealtimePageState extends State<DebugRealtimePage> {
  final dbRef = FirebaseDatabase.instance.ref('debug/realtime');
  String _value = '---';
  Stream<DatabaseEvent>? _stream;
  StreamSubscription<DatabaseEvent>? _sub;

  Future<void> writeTimestamp() async {
    final ts = DateTime.now().toIso8601String();
    await dbRef.push().set({'timestamp': ts});
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Wrote $ts')));
  }

  Future<void> readOnce() async {
    final snap = await dbRef.get();
    setState(() {
      _value = snap.value == null ? 'null' : snap.value.toString();
    });
  }

  @override
  void initState() {
    super.initState();
    _stream = dbRef.onValue;
    _sub = _stream!.listen((event) {
      setState(() {
        _value = event.snapshot.value == null ? 'null' : event.snapshot.value.toString();
      });
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Debug Realtime DB')),
      body: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: writeTimestamp,
              child: const Text('Write timestamp to /debug/realtime'),
            ),
            const SizedBox(height: 8),
            ElevatedButton(
              onPressed: readOnce,
              child: const Text('Read once'),
            ),
            const SizedBox(height: 12),
            Expanded(child: SingleChildScrollView(child: Text('Current value:\n\n$_value'))),
          ],
        ),
      ),
    );
  }
}