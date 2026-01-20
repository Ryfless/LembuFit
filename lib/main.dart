import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'routes/app_routes.dart'; // ganti import routing
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'services/history_service.dart'; // <-- already present
import 'services/local_history_service.dart'; // <-- added

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  // start local history writer
  await localHistoryService.start();

  // start history recorder to Firestore (optional)
  historyService.startListeningToCows();

  runApp(const LembuFitApp());
}

class LembuFitApp extends StatelessWidget {
  const LembuFitApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LembuFit',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        textTheme: GoogleFonts.poppinsTextTheme(),
        scaffoldBackgroundColor: Colors.white,
      ),
      initialRoute: '/',
      routes: AppRoutes.routes, // gunakan routes dari AppRoutes
    );
  }
}
