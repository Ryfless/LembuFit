import 'package:flutter/material.dart';
import '../screens/start_screen.dart';
import '../screens/login_screen.dart';
import '../screens/register_screen.dart';
import '../screens/home_screen.dart';
import '../screens/list_cow.dart';
import '../screens/live_cam_screen.dart';
import '../screens/debug_home.dart';
import '../screens/debug_storage.dart';
import '../screens/debug_realtime.dart';
import '../screens/debug_firestore.dart';

class AppRoutes {
  static Map<String, WidgetBuilder> get routes => {
    '/': (context) => const StartScreen(),
    '/login': (context) => const LoginScreen(),
    '/register': (context) => const RegisterScreen(),
    '/home': (context) => const HomeScreen(),
    '/list-cow': (context) => const ListCowScreen(),
    '/live-cam': (context) => const LiveCamScreen(),
    // removed history-chart route to prevent opening History screen from routes
    // debug routes
    '/debug': (context) => const DebugHome(),
    '/debug/storage': (context) => const DebugStoragePage(),
    '/debug/realtime': (context) => const DebugRealtimePage(),
    '/debug/firestore': (context) => const DebugFirestorePage(),
  };
}