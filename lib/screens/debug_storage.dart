import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:firebase_storage/firebase_storage.dart';

class DebugStoragePage extends StatefulWidget {
  const DebugStoragePage({super.key});
  @override
  State<DebugStoragePage> createState() => _DebugStoragePageState();
}

class _DebugStoragePageState extends State<DebugStoragePage> {
  final FirebaseStorage _storage = FirebaseStorage.instance;
  List<Reference> _items = [];
  bool _loading = false;

  Future<void> uploadTestFile() async {
    setState(() => _loading = true);
    try {
      final data = Uint8List.fromList('debug-upload-${DateTime.now()}'.codeUnits);
      final ref = _storage.ref().child('debug/test.txt');
      await ref.putData(data, SettableMetadata(contentType: 'text/plain'));
      final url = await ref.getDownloadURL();
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Uploaded: $url')));
      await listDebug();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Upload failed: $e')));
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> listDebug() async {
    setState(() => _loading = true);
    try {
      final res = await _storage.ref().child('debug').listAll();
      setState(() => _items = res.items);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('List failed: $e')));
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  void initState() {
    super.initState();
    listDebug();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Debug Storage')),
      body: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: _loading ? null : uploadTestFile,
              child: const Text('Upload test file to debug/test.txt'),
            ),
            const SizedBox(height: 8),
            ElevatedButton(
              onPressed: _loading ? null : listDebug,
              child: const Text('Refresh list'),
            ),
            const SizedBox(height: 12),
            _loading ? const CircularProgressIndicator() : Expanded(
              child: ListView.builder(
                itemCount: _items.length,
                itemBuilder: (context, i) {
                  final item = _items[i];
                  return ListTile(
                    title: Text(item.name),
                    subtitle: Text(item.fullPath),
                    trailing: IconButton(
                      icon: const Icon(Icons.link),
                      onPressed: () async {
                        try {
                          final url = await item.getDownloadURL();
                          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(url)));
                        } catch (e) {
                          ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Get URL failed: $e')));
                        }
                      },
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}