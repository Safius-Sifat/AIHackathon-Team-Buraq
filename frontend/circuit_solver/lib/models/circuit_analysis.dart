class CircuitAnalysis {
  final Map<String, double> nodeVoltages;

  CircuitAnalysis({required this.nodeVoltages});

  factory CircuitAnalysis.fromJson(Map<String, dynamic> json) {
    print("Parsing CircuitAnalysis from JSON: ${json['voltages']}");
    return CircuitAnalysis(
      nodeVoltages: Map<String, double>.from(json['voltages'] ?? {}),
    );
  }
}
