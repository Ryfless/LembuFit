import 'dart:typed_data';
import 'package:firebase_storage/firebase_storage.dart';

class StorageService {
  final FirebaseStorage _storage = FirebaseStorage.instance;

  /// List all references under [path] (default 'frames').
  Future<List<Reference>> listRefs({String path = 'frames'}) async {
    final ref = _storage.ref(path);
    final result = await ref.listAll();
    final items = List<Reference>.from(result.items);
    items.sort((a, b) => a.name.compareTo(b.name));
    return items;
  }

  /// Try to get download URLs for all items under [path].
  /// Returns only successful urls.
  Future<List<String>> getDownloadUrls({String path = 'frames'}) async {
    final items = await listRefs(path: path);
    final urls = <String>[];
    for (final it in items) {
      try {
        final u = await it.getDownloadURL();
        urls.add(u);
      } catch (_) {
        // skip
      }
    }
    return urls;
  }

  /// Get bytes for a specific reference (maxSize default 2MB).
  Future<Uint8List?> getBytes(Reference ref, {int maxSize = 2 * 1024 * 1024}) async {
    try {
      final data = await ref.getData(maxSize);
      return data;
    } on FirebaseException catch (_) {
      return null;
    } catch (_) {
      return null;
    }
  }

  /// Get bytes for items under [path]. Skips items that fail.
  Future<List<Uint8List>> getBytesList({String path = 'frames', int maxSize = 2 * 1024 * 1024}) async {
    final refs = await listRefs(path: path);
    final out = <Uint8List>[];
    for (final r in refs) {
      try {
        final b = await getBytes(r, maxSize: maxSize);
        if (b != null) out.add(b);
      } catch (_) {}
    }
    return out;
  }

  // -------------------------
  // NEW: get download url for arbitrary storage path (e.g. 'images/cow_1.png')
  Future<String?> getDownloadUrlForPath(String path) async {
    if (path.isEmpty) return null;
    try {
      // if already a URL, return directly
      if (path.startsWith('http://') || path.startsWith('https://')) return path;
      final ref = _storage.ref(path);
      final url = await ref.getDownloadURL();
      return url;
    } on FirebaseException {
      return null;
    } catch (_) {
      return null;
    }
  }
}

final storageService = StorageService();