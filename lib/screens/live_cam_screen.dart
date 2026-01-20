import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'home_screen.dart';
import 'list_cow.dart';
import 'package:flutter/services.dart';
import 'package:firebase_storage/firebase_storage.dart';
import '../services/storage_service.dart';
import '../services/firestore_service.dart';
import '../models/cow_model.dart';
import 'package:flutter/animation.dart';

class LiveCamScreen extends StatefulWidget {
  const LiveCamScreen({super.key});

  @override
  State<LiveCamScreen> createState() => _LiveCamScreenState();
}

class _LiveCamScreenState extends State<LiveCamScreen> with SingleTickerProviderStateMixin {
  List<Reference> _frameRefs = [];
  final Map<int, Uint8List> _frameBytes = {};
  int _currentFrame = 0;
  Timer? _timer;
  bool _isPlaying = true;
  bool _loadingFrames = false;
  String? _error;
  int _fps = 2;

  int _selectedIndex = 1;
  // notification state
  bool _showNotification = false;
  late AnimationController _notifController;
  List<Cow> cows = [];
  StreamSubscription<List<Cow>>? _cowSub;

  @override
  void initState() {
    super.initState();
    _loadRefs();
    // notif controller & firestore cows listener
    _notifController = AnimationController(vsync: this, duration: const Duration(milliseconds: 300));
    _cowSub = firestoreService.streamCows().listen((list) {
      if (!mounted) return;
      setState(() => cows = list);
    }, onError: (_) {});
  }

  Future<void> _loadRefs() async {
    // guard entry
    if (!mounted) return;
    setState(() {
      _loadingFrames = true;
      _error = null;
      _frameRefs = [];
      _frameBytes.clear();
      _currentFrame = 0;
    });
    try {
      final refs = await storageService.listRefs(path: 'frames');
      if (!mounted) return;
      if (refs.isEmpty) {
        if (!mounted) return;
        setState(() => _error = 'No frames found in Storage:/frames');
      } else {
        if (!mounted) return;
        setState(() => _frameRefs = refs);
        // prefetch first few frames as bytes
        await _prefetchFrames(0, ahead: 4);
        if (!mounted) return;
        if (_frameRefs.isNotEmpty) _startPlayback();
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Gagal memuat frames: $e');
    } finally {
      if (!mounted) return;
      setState(() => _loadingFrames = false);
    }
  }

  Future<void> _prefetchFrames(int index, {int ahead = 3}) async {
    if (_frameRefs.isEmpty) return;
    for (int i = 0; i <= ahead; i++) {
      final idx = (index + i) % _frameRefs.length;
      if (_frameBytes.containsKey(idx)) continue;
      final ref = _frameRefs[idx];
      final bytes = await storageService.getBytes(ref, maxSize: 3 * 1024 * 1024);
      if (bytes != null) {
        if (!mounted) return;
        // small guard to avoid calling setState after dispose
        setState(() {
          _frameBytes[idx] = bytes;
        });
      }
    }
  }

  void _startPlayback() {
    _timer?.cancel();
    if (!_isPlaying || _frameRefs.isEmpty) return;
    final int intervalMs = (_fps <= 0) ? 100 : (1000 ~/ _fps);
    _timer = Timer.periodic(Duration(milliseconds: intervalMs), (_) {
      if (!mounted || _frameRefs.isEmpty) return;
      setState(() => _currentFrame = (_currentFrame + 1) % _frameRefs.length);
      // prefetch async, it's already guarded
      _prefetchFrames(_currentFrame, ahead: 3);
    });
  }

  void _stopPlayback() {
    _timer?.cancel();
    _timer = null;
  }

  void _togglePlayPause() {
    setState(() => _isPlaying = !_isPlaying);
    if (_isPlaying) {
      _startPlayback();
    } else {
      _stopPlayback();
    }
  }

  @override
  void dispose() {
    // stop timer and subscription first to avoid callbacks after dispose
    _stopPlayback();
    _cowSub?.cancel();
    _notifController.dispose();
    super.dispose();
  }

  void _onNavTap(int index) {
    if (index == 0) {
      Navigator.of(context).pushReplacement(_createRoute(
        target: const HomeScreen(),
        slideFromRight: false,
      ));
    } else if (index == 1) {
      // already LiveCam -> do nothing or keep
    } else if (index == 2) {
      Navigator.of(context).pushReplacement(_createRoute(
        target: const ListCowScreen(),
        slideFromRight: true,
      ));
    }
  }

  Route _createRoute({required Widget target, required bool slideFromRight}) {
    return PageRouteBuilder(
      pageBuilder: (context, animation, secondaryAnimation) => target,
      transitionsBuilder: (context, animation, secondaryAnimation, child) {
        final begin = Offset(slideFromRight ? 1.0 : -1.0, 0.0);
        const end = Offset.zero;
        const curve = Curves.ease;
        final tween =
            Tween(begin: begin, end: end).chain(CurveTween(curve: curve));
        return SlideTransition(
          position: animation.drive(tween),
          child: child,
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Stack(
        children: [
          SafeArea(
            child: Column(
              children: [
                const SizedBox(height: 16),
                Center(
                  child: Text(
                    "Live Cam",
                    style: GoogleFonts.poppins(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: const Color(0xFF189E8F),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          shape: BoxShape.circle,
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey.withOpacity(0.15),
                              blurRadius: 4,
                              offset: const Offset(0, 2),
                            ),
                          ],
                        ),
                        child: IconButton(
                          icon: const Icon(Icons.notifications, color: Color(0xFF189E8F)),
                          onPressed: () {
                            setState(() => _showNotification = !_showNotification);
                            if (_showNotification) _notifController.forward(from: 0);
                          },
                        ),
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 0),
                  child: Align(
                    alignment: Alignment.centerLeft,
                    child: Text("Cam 1", style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                  child: _buildFrameViewer(context),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 0),
                  child: Align(
                    alignment: Alignment.centerLeft,
                    child: Text("Cam 2", style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                  child: Container(
                    height: 140,
                    decoration: BoxDecoration(
                      color: Colors.grey[350],
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const Center(
                      child: Icon(Icons.volume_off, color: Colors.grey, size: 48),
                    ),
                  ),
                ),
              ],
            ),
          ),
          if (_showNotification) _buildNotificationOverlay(),
        ],
      ),
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.symmetric(vertical: 10),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: List.generate(3, (i) {
            final icons = [
              Icons.home,
              Icons.camera,
              Icons.list,
            ];
            final isSelected = i == _selectedIndex;
            return AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: isSelected ? const Color(0xFF189E8F) : Colors.transparent,
              ),
              child: IconButton(
                icon: Icon(
                  icons[i],
                  color: isSelected ? Colors.white : Colors.grey,
                  size: isSelected ? 28 : 24,
                ),
                onPressed: () {
                  if (i != _selectedIndex) {
                    _onNavTap(i);
                  }
                },
              ),
            );
          }),
        ),
      ),
    );
  }

  Widget _buildFrameViewer(BuildContext context) {
    const height = 200.0;
    if (_loadingFrames) {
      return Container(
        height: height,
        decoration: BoxDecoration(color: Colors.black12, borderRadius: BorderRadius.circular(16)),
        child: const Center(child: CircularProgressIndicator()),
      );
    }

    if (_error != null) {
      return Container(
        height: height,
        decoration: BoxDecoration(color: Colors.black12, borderRadius: BorderRadius.circular(16)),
        child: Center(child: Text(_error!, style: const TextStyle(color: Colors.red))),
      );
    }

    if (_frameRefs.isEmpty) {
      return Container(
        height: height,
        decoration: BoxDecoration(color: Colors.black12, borderRadius: BorderRadius.circular(16)),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('No frames found in Storage:/frames'),
              const SizedBox(height: 8),
              ElevatedButton(onPressed: _loadRefs, child: const Text('Reload')),
            ],
          ),
        ),
      );
    }

    final currentIndex = _currentFrame;
    final bytes = _frameBytes[currentIndex];

    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: AspectRatio(
        aspectRatio: 16 / 9,
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (bytes != null)
              Image.memory(bytes, fit: BoxFit.cover)
            else
              FutureBuilder<Uint8List?>(
                future: storageService.getBytes(_frameRefs[currentIndex], maxSize: 3 * 1024 * 1024),
                builder: (context, snap) {
                  if (snap.connectionState == ConnectionState.waiting) {
                    return const Center(child: CircularProgressIndicator());
                  }
                  if (snap.hasData && snap.data != null) {
                    // cache it for next frames
                    WidgetsBinding.instance.addPostFrameCallback((_) {
                      if (mounted) setState(() => _frameBytes[currentIndex] = snap.data!);
                    });
                    return Image.memory(snap.data!, fit: BoxFit.cover);
                  }
                  // fallback: try downloadURL -> network image (may fail on web due to CORS)
                  return FutureBuilder<String?>(
                    future: _getDownloadUrlFallback(currentIndex),
                    builder: (c2, snap2) {
                      if (snap2.connectionState == ConnectionState.waiting) {
                        return const Center(child: CircularProgressIndicator());
                      }
                      if (snap2.hasData && snap2.data != null) {
                        return Image.network(snap2.data!, fit: BoxFit.cover);
                      }
                      return const Center(child: Icon(Icons.broken_image, size: 48, color: Colors.grey));
                    },
                  );
                },
              ),
            // controls overlay (play/pause kept)
            Positioned.fill(
              child: Align(
                alignment: Alignment.center,
                child: GestureDetector(
                  onTap: _togglePlayPause,
                  child: _isPlaying
                      ? const SizedBox.shrink()
                      : Container(
                          decoration: BoxDecoration(color: Colors.black45, shape: BoxShape.circle),
                          padding: const EdgeInsets.all(12),
                          child: const Icon(Icons.play_arrow, color: Colors.white, size: 48),
                        ),
                ),
              ),
            ),
            Positioned(
              left: 8,
              bottom: 8,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(color: Colors.black45, borderRadius: BorderRadius.circular(8)),
                child: Text(
                  '${currentIndex + 1}/${_frameRefs.length}',
                  style: const TextStyle(color: Colors.white, fontSize: 12),
                ),
              ),
            ),
            Positioned(
              right: 8,
              bottom: 8,
              child: Row(
                children: [
                  IconButton(
                    icon: Icon(_isPlaying ? Icons.pause_circle : Icons.play_circle, color: Colors.white),
                    onPressed: _togglePlayPause,
                  ),
                  PopupMenuButton<int>(
                    color: Colors.white,
                    onSelected: (value) {
                      setState(() {
                        _fps = value;
                        if (_isPlaying) _startPlayback();
                      });
                    },
                    itemBuilder: (_) => [6, 12, 24, 30].map((v) {
                      return PopupMenuItem(value: v, child: Text('$v FPS'));
                    }).toList(),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                      decoration: BoxDecoration(color: Colors.black45, borderRadius: BorderRadius.circular(8)),
                      child: Text('$_fps FPS', style: const TextStyle(color: Colors.white)),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<String?> _getDownloadUrlFallback(int index) async {
    try {
      return await _frameRefs[index].getDownloadURL();
    } catch (_) {
      return null;
    }
  }

  Widget _buildNotificationOverlay() {
    return Stack(
      children: [
        AnimatedOpacity(
          opacity: _showNotification ? 0.6 : 0,
          duration: const Duration(milliseconds: 300),
          child: Container(color: Colors.black),
        ),
        Center(
          child: ScaleTransition(
            scale: CurvedAnimation(parent: _notifController, curve: Curves.easeOutBack),
            child: _buildNotificationPopup(),
          ),
        ),
      ],
    );
  }

  Widget _buildNotificationPopup() {
    final notif = _getNotifCows();
    return Container(
      width: 300,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16)),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Align(
            alignment: Alignment.topRight,
            child: IconButton(icon: const Icon(Icons.close), onPressed: () => setState(() => _showNotification = false)),
          ),
          Text("Notifikasi Sapi", style: GoogleFonts.poppins(fontWeight: FontWeight.w600, color: const Color(0xFF189E8F))),
          const SizedBox(height: 8),
          notif.isEmpty
              ? Text("Tidak ada sapi bermasalah.", style: GoogleFonts.poppins(color: Colors.grey))
              : Column(
                  children: notif.map((cow) {
                    return Container(
                      margin: const EdgeInsets.symmetric(vertical: 4),
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(border: Border.all(color: Colors.grey.shade300), borderRadius: BorderRadius.circular(10)),
                      child: Row(
                        children: [
                          Icon(Icons.error, color: _getCowStatusColor(cow.status), size: 20),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                              Text("Status: ${cow.status}", style: GoogleFonts.poppins(fontWeight: FontWeight.w500, fontSize: 13)),
                              Text("${cow.name} • Suhu: ${cow.temp} • Detak: ${cow.heart}", style: GoogleFonts.poppins(color: Colors.grey, fontSize: 12)),
                            ]),
                          ),
                        ],
                      ),
                    );
                  }).toList(),
                ),
        ],
      ),
    );
  }

  List<Cow> _getNotifCows() {
    return cows.where((cow) =>
      cow.status == 'Kurang Sehat' ||
      cow.status == 'Kurang Aktif' ||
      cow.status == 'Tidak Sehat'
    ).toList();
  }

  Color _getCowStatusColor(String status) {
    switch (status) {
      case 'Sehat':
        return Colors.green;
      case 'Kurang Sehat':
      case 'Kurang Aktif':
        return Colors.yellow[700]!;
      case 'Tidak Sehat':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }
}