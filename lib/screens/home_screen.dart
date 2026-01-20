import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import '../models/cow_model.dart';
import 'list_cow.dart';
import 'live_cam_screen.dart';
import '../services/firestore_service.dart';
import '../services/realtime_sensor_service.dart';
import '../services/storage_service.dart'; // <-- added
import 'add_cow_screen.dart'; // Tambahkan import untuk AddCowScreen

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  int _selectedIndex = 0;
  bool _showNotification = false;
  int _currentCowIndex = 0;
  late PageController _pageController;
  late AnimationController _navController;

  List<Cow> cows = [];
  final FirestoreService _firestoreService = firestoreService;
  late final RealtimeSensorService _realtimeService;
  StreamSubscription<List<Cow>>? _cowSub;

  // NEW: cache untuk url gambar per cow id or image path
  final Map<String, String?> _imageUrlCache = {};

  @override
  void initState() {
    super.initState();
    _pageController = PageController();
    _navController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 300),
    );

    _realtimeService = RealtimeSensorService(_firestoreService);
    _realtimeService.start();

    // subscribe to cows collection realtime
    _cowSub = _firestoreService.streamCows().listen((list) {
      setState(() {
        cows = list;
        if (_currentCowIndex >= cows.length) _currentCowIndex = 0;
      });
    }, onError: (e) {
      // keep existing cows or handle error
    });
  }

  @override
  void dispose() {
    _cowSub?.cancel();
    _realtimeService.stop();
    _pageController.dispose();
    _navController.dispose();
    super.dispose();
  }

  void _onNavTap(int index) {
    if (index == 0) {
      setState(() => _selectedIndex = index); // Sudah di home
    } else if (index == 1) {
      Navigator.of(context).pushReplacement(_createRoute(
        target: const LiveCamScreen(),
        slideFromRight: true,
      ));
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

  void _onArrowTap(bool isNext) {
    if (cows.isEmpty) return;
    setState(() {
      if (isNext) {
        _currentCowIndex = (_currentCowIndex + 1) % cows.length;
      } else {
        _currentCowIndex = (_currentCowIndex - 1 + cows.length) % cows.length;
      }
      _pageController.animateToPage(
        _currentCowIndex,
        duration: const Duration(milliseconds: 400),
        curve: Curves.easeInOut,
      );
    });
  }

  void _showAddCowDialog() {
    final nameController = TextEditingController();
    final idController = TextEditingController();
    final statusController = TextEditingController();
    final tempController = TextEditingController();
    final heartController = TextEditingController();
    final activityController = TextEditingController();
    final imageController = TextEditingController();

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Tambah Data Sapi'),
          content: SingleChildScrollView(
            child: Column(
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
                  controller: statusController,
                  decoration: const InputDecoration(labelText: 'Status'),
                ),
                TextField(
                  controller: tempController,
                  decoration: const InputDecoration(labelText: 'Suhu Tubuh'),
                ),
                TextField(
                  controller: heartController,
                  decoration: const InputDecoration(labelText: 'Detak Jantung'),
                ),
                TextField(
                  controller: activityController,
                  decoration: const InputDecoration(labelText: 'Aktivitas'),
                ),
                TextField(
                  controller: imageController,
                  decoration: const InputDecoration(labelText: 'Path Gambar'),
                ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('Batal'),
            ),
            ElevatedButton(
              onPressed: () {
                setState(() {
                  cows.add(Cow(
                    name: nameController.text,
                    id: idController.text,
                    status: statusController.text,
                    temp: tempController.text,
                    heart: heartController.text,
                    activity: activityController.text,
                    image: imageController.text.isNotEmpty
                        ? imageController.text
                        : 'assets/images/cow_1.png',
                  ));
                });
                Navigator.of(context).pop();
              },
              child: const Text('Simpan'),
            ),
          ],
        );
      },
    );
  }

  // NEW helper: resolve image url with cache and fallback path
  Future<String?> _resolveCowImageUrl(Cow cow) async {
    final key = (cow.image.isNotEmpty) ? cow.image : 'images/cow_1.png';
    if (_imageUrlCache.containsKey(key)) return _imageUrlCache[key];
    final url = await storageService.getDownloadUrlForPath(key);
    _imageUrlCache[key] = url; // may be null
    return url;
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
                _buildHeader(),
                Expanded(child: _buildCowCard()),
                _buildBottomNavBar(),
              ],
            ),
          ),
          if (_showNotification) _buildNotificationOverlay(),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Image.asset('assets/images/logo.png', width: 120),
          GestureDetector(
            onTap: () {
              setState(() {
                _showNotification = !_showNotification;
              });
            },
            child: Stack(
              children: [
                const Icon(Icons.notifications, color: Color(0xFF189E8F), size: 30),
                Positioned(
                  right: 2,
                  top: 2,
                  child: Container(
                    height: 12,
                    width: 12,
                    decoration: const BoxDecoration(
                      color: Colors.red,
                      shape: BoxShape.circle,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

 Widget _buildCowCard() {
  if (cows.isEmpty) {
    return const Center(child: CircularProgressIndicator());
  }

  return Stack(
    children: [
      PageView.builder(
        controller: _pageController,
        itemCount: cows.length,
        onPageChanged: (index) {
          setState(() {
            _currentCowIndex = index;
          });
        },
        itemBuilder: (context, index) {
          final cow = cows[index];
          return AnimatedScale(
            scale: index == _currentCowIndex ? 1.0 : 0.95,
            duration: const Duration(milliseconds: 300),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 10),
              child: Card(
                elevation: 5,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      // HEADER CARDBOX SAPI
                      Stack(
                        children: [
                          Container(
                            height: 44,
                            decoration: const BoxDecoration(
                              color: Color(0xFF189E8F),
                              borderRadius: BorderRadius.only(
                                topLeft: Radius.circular(16),
                                topRight: Radius.circular(16),
                              ),
                            ),
                          ),
                          Positioned.fill(
                            child: Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Padding(
                                  padding: const EdgeInsets.only(left: 20),
                                  child: Text(
                                    'Sapi ${index + 1}',
                                    style: GoogleFonts.poppins(
                                      color: Colors.white,
                                      fontWeight: FontWeight.w600,
                                      fontSize: 16,
                                    ),
                                  ),
                                ),
                                Padding(
                                  padding: const EdgeInsets.only(right: 20),
                                  child: GestureDetector(
                                    onTap: () {
                                      Navigator.of(context).push(
                                        MaterialPageRoute(
                                          builder: (context) => AddCowScreen(
                                            onAdd: (newCow) {
                                              setState(() {
                                                cows.add(newCow);
                                              });
                                            },
                                          ),
                                        ),
                                      );
                                    },
                                    child: const Icon(Icons.settings, color: Colors.white, size: 20),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      // Find the CircleAvatar in the card builder and replace with:
                      FutureBuilder<String?>(
                        future: _resolveCowImageUrl(cow),
                        builder: (context, snap) {
                          Widget avatarChild;
                          if (snap.connectionState == ConnectionState.waiting) {
                            avatarChild = const CircleAvatar(
                              radius: 35,
                              backgroundColor: Colors.grey,
                              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                            );
                          } else if (snap.hasData && snap.data != null) {
                            avatarChild = CircleAvatar(
                              radius: 35,
                              backgroundImage: NetworkImage(snap.data!),
                              backgroundColor: Colors.grey[200],
                            );
                          } else {
                            // fallback to asset (local bundled image)
                            final assetPath = (cow.image.isNotEmpty) ? cow.image : 'assets/images/cow_1.png';
                            avatarChild = CircleAvatar(
                              radius: 35,
                              backgroundImage: AssetImage(assetPath),
                              backgroundColor: Colors.grey[200],
                            );
                          }
                          return avatarChild;
                        },
                      ),
                      const SizedBox(height: 10),
                      Text(
                        cow.name,
                        style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
                      ),
                      Text(
                        'ID: ${cow.id}',
                        style: GoogleFonts.poppins(fontSize: 12),
                      ),
                      const SizedBox(height: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                        decoration: BoxDecoration(
                          color: getCowStatusColor(cow.status).withOpacity(0.5),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Text(
                          cow.status,
                          style: GoogleFonts.poppins(
                            fontSize: 12,
                            color: getCowStatusColor(cow.status),
                          ),
                        ),
                      ),
                      const SizedBox(height: 14),
                      _buildLiveDataCard(cow),
                    ],
                  ),
                ),
              ),
            ),
          );
        },
      ),
      _buildArrowButton(isNext: false),
      _buildArrowButton(isNext: true),
    ],
  );
}


  Widget _buildLiveDataCard(Cow cow) {
    return Container(
      padding: const EdgeInsets.all(12),
      margin: const EdgeInsets.only(top: 4),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: const [
              Text('Live Sensor Data',
                  style: TextStyle(fontWeight: FontWeight.w600)),
              Icon(Icons.wifi, size: 18),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              const Icon(Icons.thermostat, size: 18),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    cow.temp,
                    style: GoogleFonts.poppins(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Container(
                    margin: const EdgeInsets.only(top: 2, bottom: 4),
                    height: 1,
                    width: 60,
                    color: Colors.grey.shade200,
                  ),
                  Text(
                    'Suhu tubuh',
                    style: GoogleFonts.poppins(
                      fontSize: 12,
                      color: Colors.grey[500],
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              const Icon(Icons.favorite, size: 18),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    cow.heart,
                    style: GoogleFonts.poppins(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Container(
                    margin: const EdgeInsets.only(top: 2, bottom: 4),
                    height: 1,
                    width: 60,
                    color: Colors.grey.shade200,
                  ),
                  Text(
                    'Detak Jantung',
                    style: GoogleFonts.poppins(
                      fontSize: 12,
                      color: Colors.grey[500],
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              const Icon(Icons.pets, size: 18),
              const SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    cow.activity,
                    style: GoogleFonts.poppins(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Container(
                    margin: const EdgeInsets.only(top: 2, bottom: 4),
                    height: 1,
                    width: 60,
                    color: Colors.grey.shade200,
                  ),
                  Text(
                    'Aktivitas',
                    style: GoogleFonts.poppins(
                      fontSize: 12,
                      color: Colors.grey[500],
                    ),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildArrowButton({required bool isNext}) {
    return Positioned(
      top: 150,
      left: isNext ? null : 10,
      right: isNext ? 10 : null,
      child: GestureDetector(
        onTap: () => _onArrowTap(isNext),
        child: Container(
          padding: const EdgeInsets.all(6),
          decoration: const BoxDecoration(
            color: Color(0xFF189E8F),
            shape: BoxShape.circle,
          ),
          child: Icon(
            isNext ? Icons.arrow_forward_ios : Icons.arrow_back_ios,
            color: Colors.white,
            size: 16,
          ),
        ),
      ),
    );
  }

  Widget _buildBottomNavBar() {
    final icons = [
      Icons.home,   // index 0
      Icons.camera, // index 1
      Icons.list,   // index 2
    ];

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: List.generate(icons.length, (i) {
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
    );
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
            scale: CurvedAnimation(
              parent: _navController..forward(from: 0),
              curve: Curves.easeOutBack,
            ),
            child: _buildNotificationPopup(),
          ),
        ),
      ],
    );
  }

  Widget _buildNotificationPopup() {
    final notifCows = getNotifCows();
    return Container(
      width: 300,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Align(
            alignment: Alignment.topRight,
            child: IconButton(
              icon: const Icon(Icons.close),
              onPressed: () => setState(() => _showNotification = false),
            ),
          ),
          Text("Notifikasi Sapi",
              style: GoogleFonts.poppins(
                  fontWeight: FontWeight.w600, color: Color(0xFF189E8F))),
          const SizedBox(height: 8),
          notifCows.isEmpty
            ? Text("Tidak ada sapi dengan status kurang sehat/aktif atau sakit.",
                style: GoogleFonts.poppins(color: Colors.grey))
            : Column(
                children: notifCows.map((cow) => _buildNotifItem(
                  color: getCowStatusColor(cow.status),
                  text: "Status: ${cow.status}\nSuhu: ${cow.temp}, Detak: ${cow.heart}, Aktivitas: ${cow.activity}",
                  name: cow.name,
                )).toList(),
              ),
        ],
      ),
    );
  }

  Widget _buildNotifItem(
      {required Color color, required String text, required String name}) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        children: [
          Icon(Icons.error, color: color, size: 20),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(text,
                    style: GoogleFonts.poppins(
                        fontWeight: FontWeight.w500, fontSize: 13)),
                Text(name,
                    style: GoogleFonts.poppins(
                        color: Colors.grey, fontSize: 12)),
              ],
            ),
          )
        ],
      ),
    );
  }

// Fungsi status otomatis
String getCowStatus({
  required String suhu,
  required String detakJantung,
  required String aktivitas,
}) {
  double suhuVal = double.tryParse(suhu.replaceAll(RegExp(r'[^0-9.]'), '')) ?? 0;
  int detakVal = int.tryParse(detakJantung.replaceAll(RegExp(r'[^0-9]'), '')) ?? 0;
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

// Fungsi warna status
Color getCowStatusColor(String status) {
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

List<Cow> getNotifCows() {
  return cows.where((cow) =>
    cow.status == 'Kurang Sehat' ||
    cow.status == 'Kurang Aktif' ||
    cow.status == 'Tidak Sehat'
  ).toList();
}
}