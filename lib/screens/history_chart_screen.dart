import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:fl_chart/fl_chart.dart';
import 'home_screen.dart';
import 'list_cow.dart';
import 'live_cam_screen.dart';
import '../services/firestore_service.dart';
import '../services/history_service.dart';
import '../models/cow_model.dart';
import 'dart:async';

class HistoryChartScreen extends StatefulWidget {
  const HistoryChartScreen({super.key});

  @override
  State<HistoryChartScreen> createState() => _HistoryChartScreenState();
}

class _HistoryChartScreenState extends State<HistoryChartScreen> {
  int _selectedIndex = 3; // Chart index 3

  // cows list from Firestore to select cow
  List<Cow> cows = [];
  String? selectedCowId;
  String selectedCowName = 'Sapi';
  String selectedPeriod = 'Hari';
  StreamSubscription<List<Cow>>? _cowSub;
  StreamSubscription<List<Map<String, dynamic>>>? _historySub;

  // chart data lists
  List<double> suhuData = [];
  List<double> heartData = [];
  List<double> activityData = [];

  @override
  void initState() {
    super.initState();
    // start history recorder (listens to cows changes and writes history)
    historyService.startListeningToCows();

    // subscribe cows list to populate dropdown
    _cowSub = firestoreService.streamCows().listen((list) {
      if (!mounted) return;
      setState(() {
        cows = list;
        if (selectedCowId == null && cows.isNotEmpty) {
          selectedCowId = cows.first.id;
          selectedCowName = cows.first.name;
          _subscribeHistoryForSelectedCow();
        }
      });
    });
  }

  @override
  void dispose() {
    _cowSub?.cancel();
    _historySub?.cancel();
    historyService.stop();
    super.dispose();
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

  void _subscribeHistoryForSelectedCow() {
    _historySub?.cancel();
    if (selectedCowId == null) return;

    // Get a larger limit and then trim by period
    _historySub = historyService.streamHistoryForCow(selectedCowId!, limit: 500).listen((entries) {
      if (!mounted) return;
      // entries are ordered desc by timestamp
      final rev = List<Map<String, dynamic>>.from(entries.reversed); // ascending
      // parse numeric lists
      final temps = <double>[];
      final hearts = <double>[];
      final acts = <double>[];

      for (final e in rev) {
        final t = _toDoubleSafe(e['temp']);
        final h = _toDoubleSafe(e['heart']);
        final aStr = (e['activity'] ?? '').toString().toLowerCase();
        final a = _activityToNumeric(aStr);
        temps.add(t);
        hearts.add(h);
        acts.add(a);
      }

      // slice based on period: Hari -> last 24, Minggu -> last 7, Bulan -> last 30
      int takeCount = 24;
      if (selectedPeriod == 'Minggu') takeCount = 7;
      if (selectedPeriod == 'Bulan') takeCount = 30;
      // if not enough points, keep shorter length
      if (temps.length > takeCount) {
        temps.removeRange(0, temps.length - takeCount);
        hearts.removeRange(0, hearts.length - takeCount);
        acts.removeRange(0, acts.length - takeCount);
      }

      setState(() {
        suhuData = temps;
        heartData = hearts;
        activityData = acts;
      });
    });
  }

  double _toDoubleSafe(dynamic v) {
    if (v == null) return 0.0;
    if (v is num) return v.toDouble();
    final s = v.toString();
    final parsed = double.tryParse(s);
    if (parsed != null) return parsed;
    // try to extract number
    final m = RegExp(r'[-+]?[0-9]*\.?[0-9]+').firstMatch(s);
    if (m != null) return double.tryParse(m.group(0)!) ?? 0.0;
    return 0.0;
  }

  double _activityToNumeric(String a) {
    if (a.contains('diam')) return 0.0;
    if (a.contains('berjalan')) return 1.0;
    return 2.0; // aktif / berlari
  }

  Widget _buildPeriodButton(String label) {
    final isSelected = selectedPeriod == label;
    return Expanded(
      child: GestureDetector(
        onTap: () {
          setState(() => selectedPeriod = label);
          // resubscribe/trim existing data
          _subscribeHistoryForSelectedCow();
        },
        child: Container(
          height: 36,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: isSelected ? const Color(0xFF189E8F) : Colors.white,
            border: Border.all(color: const Color(0xFF189E8F)),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Text(
            label,
            style: GoogleFonts.poppins(
              color: isSelected ? Colors.white : const Color(0xFF189E8F),
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLineChart(List<double> data, {bool showDots = true}) {
    if (data.isEmpty) {
      return Center(child: Text('Tidak ada data', style: GoogleFonts.poppins(color: Colors.grey)));
    }
    final minY = data.reduce((a, b) => a < b ? a : b) - 1;
    final maxY = data.reduce((a, b) => a > b ? a : b) + 1;
    return LineChart(
      LineChartData(
        gridData: FlGridData(show: false),
        titlesData: FlTitlesData(show: false),
        borderData: FlBorderData(
          show: true,
          border: Border.all(color: Colors.grey.shade300),
        ),
        minY: minY,
        maxY: maxY,
        lineBarsData: [
          LineChartBarData(
            spots: List.generate(data.length, (i) => FlSpot(i.toDouble(), data[i])),
            isCurved: true,
            color: const Color(0xFF189E8F),
            barWidth: 3,
            belowBarData: BarAreaData(
              show: true,
              gradient: LinearGradient(
                colors: [
                  const Color(0xFF189E8F).withOpacity(0.15),
                  Colors.transparent,
                ],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              ),
            ),
            dotData: FlDotData(show: showDots),
          ),
        ],
      ),
    );
  }

  Widget _buildBarChart(List<double> data) {
    if (data.isEmpty) {
      return Center(child: Text('Tidak ada data', style: GoogleFonts.poppins(color: Colors.grey)));
    }
    return BarChart(
      BarChartData(
        gridData: FlGridData(show: false),
        titlesData: FlTitlesData(show: false),
        borderData: FlBorderData(
          show: true,
          border: Border.all(color: Colors.grey.shade300),
        ),
        barGroups: List.generate(
          data.length,
          (i) => BarChartGroupData(
            x: i,
            barRods: [
              BarChartRodData(
                toY: data[i],
                color: const Color(0xFF189E8F),
                width: 12,
                borderRadius: BorderRadius.circular(4),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final cowNames = cows.map((c) => {'id': c.id, 'name': c.name}).toList();
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            const SizedBox(height: 16),
            Center(
              child: Text(
                "Grafik Riwayat",
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
                      onPressed: () {},
                    ),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("Pilih Sapi", style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 6),
                  DropdownButtonFormField<String>(
                    value: selectedCowId,
                    items: cowNames
                        .map((m) => DropdownMenuItem(
                              value: m['id'] as String?,
                              child: Text(m['name'] as String),
                            ))
                        .toList(),
                    onChanged: (val) {
                      setState(() {
                        selectedCowId = val;
                        final sel = cows.firstWhere((c) => c.id == val, orElse: () => cows.isNotEmpty ? cows.first : Cow(name:'Sapi',id:'',status:'',temp:'',heart:'',activity:'',image:'' ) );
                        selectedCowName = sel.name;
                        _subscribeHistoryForSelectedCow();
                      });
                    },
                    decoration: InputDecoration(
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      _buildPeriodButton('Hari'),
                      const SizedBox(width: 8),
                      _buildPeriodButton('Minggu'),
                      const SizedBox(width: 8),
                      _buildPeriodButton('Bulan'),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Text("Suhu Tubuh", style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
              child: Container(
                height: 160,
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: Colors.grey.shade300),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: _buildLineChart(suhuData),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Text("Detak Jantung", style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
              child: Container(
                height: 160,
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: Colors.grey.shade300),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: _buildLineChart(heartData),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Text("Aktivitas (0=Diam,1=Berjalan,2=Aktif)", style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
              child: Container(
                height: 160,
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: Colors.grey.shade300),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: _buildBarChart(activityData),
                ),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
      bottomNavigationBar: Padding(
        padding: const EdgeInsets.symmetric(vertical: 10),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: List.generate(4, (i) {
            final icons = [
              Icons.home,
              Icons.camera,
              Icons.list,
              Icons.bar_chart,
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
}