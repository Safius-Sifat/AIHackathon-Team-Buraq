import 'package:flutter/material.dart';
import 'screens/circuit_canvas.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // final spice = NgSpice();
  // final result = spice.init();
  // print("ngSpice_Init returned: $result");
  runApp(const CircuitSolverApp());
}

class CircuitSolverApp extends StatelessWidget {
  const CircuitSolverApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Qasim',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
      ),
      home: CircuitCanvas(),
    );
  }
}
