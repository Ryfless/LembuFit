import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class DebugFirestorePage extends StatefulWidget {
  const DebugFirestorePage({super.key});
  @override
  State<DebugFirestorePage> createState() => _DebugFirestorePageState();
}

class _DebugFirestorePageState extends State<DebugFirestorePage> {
  final col = FirebaseFirestore.instance.collection('debug');

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Debug Firestore')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12.0),
            child: ElevatedButton(
              onPressed: () async {
                final doc = await col.add({
                  'createdAt': FieldValue.serverTimestamp(),
                  'message': 'debug entry ${DateTime.now()}',
                });
                ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Added doc ${doc.id}')));
              },
              child: const Text('Add debug document'),
            ),
          ),
          Expanded(
            child: StreamBuilder<QuerySnapshot>(
              stream: col.orderBy('createdAt', descending: true).snapshots(),
              builder: (context, snap) {
                if (snap.hasError) return Center(child: Text('Error: ${snap.error}'));
                if (!snap.hasData) return const Center(child: CircularProgressIndicator());
                final docs = snap.data!.docs;
                return ListView.builder(
                  itemCount: docs.length,
                  itemBuilder: (context, i) {
                    final d = docs[i];
                    return ListTile(
                      title: Text(d.id),
                      subtitle: Text(d.data().toString()),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}