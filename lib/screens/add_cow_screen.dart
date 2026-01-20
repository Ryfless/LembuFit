import 'package:flutter/material.dart';
import '../models/cow_model.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:permission_handler/permission_handler.dart';

String getCowStatus({
  required double suhu,
  required int detakJantung,
  required String aktivitas,
}) {
  bool suhuNormal = suhu >= 38 && suhu <= 39.5;
  bool detakNormal = detakJantung >= 40 && detakJantung <= 80;
  bool aktivitasNormal = aktivitas.toLowerCase() != 'diam';

  int sakitCount = 0;
  if (suhu > 39.5) sakitCount++;
  if (detakJantung > 100) sakitCount++;
  if (aktivitas.toLowerCase() == 'diam') sakitCount++;
  if (sakitCount >= 2) return 'Tidak Sehat';

  if (!suhuNormal || !detakNormal) return 'Kurang Sehat';
  if (suhuNormal && detakNormal && !aktivitasNormal) return 'Kurang Aktif';
  if (suhuNormal && detakNormal && aktivitasNormal) return 'Sehat';

  return 'Kurang Sehat';
}

class AddCowScreen extends StatefulWidget {
  final Function(Cow) onAdd;
  const AddCowScreen({super.key, required this.onAdd});

  @override
  State<AddCowScreen> createState() => _AddCowScreenState();
}

class _AddCowScreenState extends State<AddCowScreen> {
  final nameController = TextEditingController();
  final idController = TextEditingController();
  final tempController = TextEditingController();
  final heartController = TextEditingController();
  final activityController = TextEditingController();
  File? _imageFile;

  Future<void> _pickImage() async {
    try {
      // Minta permission sebelum akses galeri
      var status = await Permission.photos.request();
      if (status.isGranted) {
        final picker = ImagePicker();
        final picked = await picker.pickImage(source: ImageSource.gallery);
        if (picked != null) {
          setState(() {
            _imageFile = File(picked.path);
          });
        }
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Izin akses galeri diperlukan')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Gagal mengambil gambar: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Tambah Data Sapi'),
        backgroundColor: const Color(0xFF189E8F),
        foregroundColor: Colors.white,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: ListView(
          children: [
            TextField(
              controller: nameController,
              decoration: const InputDecoration(labelText: 'Nama'),
            ),
            TextField(
              controller: idController,
              decoration: const InputDecoration(labelText: 'ID'),
            ),
            TextField(
              controller: tempController,
              decoration: const InputDecoration(labelText: 'Suhu Tubuh'),
              keyboardType: TextInputType.number,
            ),
            TextField(
              controller: heartController,
              decoration: const InputDecoration(labelText: 'Detak Jantung'),
              keyboardType: TextInputType.number,
            ),
            TextField(
              controller: activityController,
              decoration: const InputDecoration(labelText: 'Aktivitas'),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                ElevatedButton.icon(
                  onPressed: _pickImage,
                  icon: const Icon(Icons.image),
                  label: const Text('Upload Gambar'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF189E8F),
                    foregroundColor: Colors.white,
                  ),
                ),
                const SizedBox(width: 12),
                if (_imageFile != null)
                  SizedBox(
                    width: 60,
                    height: 60,
                    child: Image.file(_imageFile!, fit: BoxFit.cover),
                  ),
              ],
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF189E8F),
                foregroundColor: Colors.white,
                minimumSize: const Size.fromHeight(48),
              ),
              onPressed: () {
                double suhu = double.tryParse(tempController.text.replaceAll(RegExp(r'[^0-9.]'), '')) ?? 0;
                int detak = int.tryParse(heartController.text.replaceAll(RegExp(r'[^0-9]'), '')) ?? 0;
                String aktivitas = activityController.text;

                String status = getCowStatus(
                  suhu: suhu,
                  detakJantung: detak,
                  aktivitas: aktivitas,
                );

                final cow = Cow(
                  name: nameController.text,
                  id: idController.text,
                  status: status,
                  temp: tempController.text,
                  heart: heartController.text,
                  activity: activityController.text,
                  image: _imageFile != null ? _imageFile!.path : 'assets/images/cow_1.png',
                );
                widget.onAdd(cow);
                Navigator.of(context).pop();
              },
              child: const Text('Simpan'),
            ),
          ],
        ),
      ),
    );
  }
}