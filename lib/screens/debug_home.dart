import 'package:flutter/material.dart';

class DebugHome extends StatelessWidget {
  const DebugHome({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Firebase Debug')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton(
              onPressed: () => Navigator.pushNamed(context, '/debug/storage'),
              child: const Text('Debug Storage'),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: () => Navigator.pushNamed(context, '/debug/realtime'),
              child: const Text('Debug Realtime DB'),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
              onPressed: () => Navigator.pushNamed(context, '/debug/firestore'),
              child: const Text('Debug Firestore'),
            ),
          ],
        ),
      ),
    );
  }
}