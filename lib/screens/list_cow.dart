import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/cow_model.dart';
import 'home_screen.dart';
import 'live_cam_screen.dart';
import '../services/firestore_service.dart';
import '../services/storage_service.dart';

class ListCowScreen extends StatefulWidget {
  const ListCowScreen({super.key});

  @override
  State<ListCowScreen> createState() => _ListCowScreenState();
}

class _ListCowScreenState extends State<ListCowScreen> with SingleTickerProviderStateMixin {
  List<Cow> cows = [];
  List<Cow> filteredCows = [];
  TextEditingController searchController = TextEditingController();
  int _selectedIndex = 2; // List index 2

  final FirestoreService _firestoreService = firestoreService;
  StreamSubscription<List<Cow>>? _cowSub;
  final Map<String, String?> _imageUrlCache = {};

  bool _showNotification = false;
  late AnimationController _navController;

  @override
  void initState() {
    super.initState();
    _loadRealtimeCows();
    searchController.addListener(_onSearchChanged);
    _navController = AnimationController(vsync: this, duration: const Duration(milliseconds: 300));
  }

  Future<void> _loadRealtimeCows() async {
    _cowSub = _firestoreService.streamCows().listen((list) {
      if (!mounted) return;
      setState(() {
        cows = list;
        filteredCows = cows;
      });
    }, onError: (e) {
      // handle error if needed
    });
  }

  @override
  void dispose() {
    searchController.dispose();
    _cowSub?.cancel();
    _navController.dispose();
    super.dispose();
  }

  void _onSearchChanged() {
    final query = searchController.text.toLowerCase();
    setState(() {
      filteredCows = cows.where((cow) {
        return cow.name.toLowerCase().contains(query) ||
            cow.id.toLowerCase().contains(query);
      }).toList();
    });
  }

  Future<String?> _resolveCowImageUrl(Cow cow) async {
    final key = (cow.image.isNotEmpty) ? cow.image : 'images/cow_1.png';
    if (_imageUrlCache.containsKey(key)) return _imageUrlCache[key];
    final url = await storageService.getDownloadUrlForPath(key);
    _imageUrlCache[key] = url;
    return url;
  }

  void _onNavTap(int index) {
    if (index == 0) {
      Navigator.of(context).pushReplacement(_createRoute(
        target: const HomeScreen(),
        slideFromRight: false,
      ));
    } else if (index == 1) {
      Navigator.of(context).pushReplacement(_createRoute(
        target: const LiveCamScreen(),
        slideFromRight: true,
      ));
    } else if (index == 2) {
      // stay on list (current)
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

  // navigasi ke HomeScreen dan buka card yang sesuai (mengirim initialIndex)
  void _openCowInHome() {
    Navigator.of(context).pushReplacement(
      _createRoute(target: const HomeScreen(), slideFromRight: false),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          children: [
            const SizedBox(height: 16),
            Center(
              child: Text(
                "Data Sapi",
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
                  GestureDetector(
                    onTap: () => setState(() => _showNotification = !_showNotification),
                    child: Stack(
                      children: [
                        const Icon(Icons.notifications, color: Color(0xFF189E8F), size: 28),
                        Positioned(
                          right: 2,
                          top: 2,
                          child: Container(
                            height: 10,
                            width: 10,
                            decoration: const BoxDecoration(color: Colors.red, shape: BoxShape.circle),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: TextField(
                controller: searchController,
                decoration: InputDecoration(
                  hintText: "Cari nama/id sapi",
                  prefixIcon: const Icon(Icons.search),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                  contentPadding: const EdgeInsets.symmetric(vertical: 0, horizontal: 16),
                  isDense: true,
                ),
              ),
            ),
            const SizedBox(height: 12),
            Expanded(
              child: filteredCows.isEmpty
                  ? const Center(child: Text("Tidak ada data sapi"))
                  : ListView.separated(
                      itemCount: filteredCows.length,
                      separatorBuilder: (_, __) => const Divider(indent: 20, endIndent: 20, height: 1),
                      itemBuilder: (context, index) {
                        final cow = filteredCows[index];
                        return ListTile(
                          leading: FutureBuilder<String?>(
                            future: _resolveCowImageUrl(cow),
                            builder: (context, snap) {
                              if (snap.connectionState == ConnectionState.waiting) {
                                return const CircleAvatar(radius: 22, backgroundColor: Colors.grey);
                              } else if (snap.hasData && snap.data != null) {
                                return CircleAvatar(radius: 22, backgroundImage: NetworkImage(snap.data!));
                              } else {
                                return CircleAvatar(radius: 22, backgroundImage: AssetImage(cow.image.isNotEmpty ? cow.image : 'assets/images/cow_1.png'));
                              }
                            },
                          ),
                          title: Text(
                            cow.name,
                            style: GoogleFonts.poppins(fontWeight: FontWeight.w600),
                          ),
                          subtitle: Text(
                            cow.id,
                            style: GoogleFonts.poppins(fontSize: 12),
                          ),
                          trailing: Container(
                            width: 20,
                            height: 20,
                            decoration: BoxDecoration(
                              color: getCowStatusColor(cow.status),
                              shape: BoxShape.circle,
                            ),
                          ),
                          onTap: () {
                            _openCowInHome();
                          },
                        );
                      },
                    ),
            ),
          ],
        ),
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
      // notification overlay
      floatingActionButton: _showNotification ? _buildNotificationPopupOverlay() : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  Widget _buildNotificationPopupOverlay() {
    final notifCows = cows.where((cow) =>
        cow.status == 'Kurang Sehat' ||
        cow.status == 'Kurang Aktif' ||
        cow.status == 'Tidak Sehat').toList();

    return Container(
      width: 320,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), boxShadow: [BoxShadow(color: Colors.black26, blurRadius: 8)]),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Align(alignment: Alignment.topRight, child: IconButton(icon: const Icon(Icons.close), onPressed: () => setState(() => _showNotification = false))),
          Text("Notifikasi Sapi", style: GoogleFonts.poppins(fontWeight: FontWeight.w600, color: const Color(0xFF189E8F))),
          const SizedBox(height: 8),
          notifCows.isEmpty
              ? Text("Tidak ada sapi dengan status kurang sehat/aktif atau sakit.", style: GoogleFonts.poppins(color: Colors.grey))
              : Column(children: notifCows.map((cow) => _buildNotifItem(color: getCowStatusColor(cow.status), text: "Status: ${cow.status}\nSuhu: ${cow.temp}, Detak: ${cow.heart}, Aktivitas: ${cow.activity}", name: cow.name)).toList()),
        ],
      ),
    );
  }

  Widget _buildNotifItem({required Color color, required String text, required String name}) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(border: Border.all(color: Colors.grey.shade300), borderRadius: BorderRadius.circular(10)),
      child: Row(
        children: [
          Icon(Icons.error, color: color, size: 20),
          const SizedBox(width: 8),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(text, style: GoogleFonts.poppins(fontWeight: FontWeight.w500, fontSize: 13)), Text(name, style: GoogleFonts.poppins(color: Colors.grey, fontSize: 12))])),
        ],
      ),
    );
  }

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
}