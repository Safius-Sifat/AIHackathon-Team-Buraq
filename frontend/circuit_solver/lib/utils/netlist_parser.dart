import '../models/circuit_component.dart';
import '../models/circuit_analysis.dart';
import 'package:flutter/material.dart';

class NetlistParser {
  final Map<String, List<(String, String, String)>> _nodeConnections = {};
  final List<(String, ComponentType)> _components = [];
  final Map<String, Offset> _nodePositions = {};
  CircuitAnalysis? analysisResults;

  // Add getter for nodes
  Set<String> get nodes => _nodeConnections.keys.toSet();
  Map<String, List<(String, String, String)>> get nodeConnections =>
      _nodeConnections;

  List<CircuitComponent> parseNetlist(String netlist) {
    _nodeConnections.clear();
    _components.clear();
    final components = <CircuitComponent>[];
    final lines = netlist.split('\n');
    int componentIndex = 0;

    for (var line in lines) {
      line = line.trim();
      if (line.isEmpty || line.startsWith('*')) continue;

      final parts = line.split(RegExp(r'\s+'));
      if (parts.length < 3) continue;

      final componentName = parts[0];
      final node1 = parts[1];
      final node2 = parts[2];
      final value = parts.length > 3 ? parts[3] : '';

      final type = _getComponentType(componentName[0]);
      if (type != null) {
        final componentId = 'comp_$componentIndex';
        _components.add((componentId, type));

        _nodeConnections.putIfAbsent(node1, () => []).add((
          componentId,
          '1',
          node2,
        ));
        _nodeConnections.putIfAbsent(node2, () => []).add((
          componentId,
          '2',
          node1,
        ));

        components.add(_createComponent(type, value));
        componentIndex++;
      }
    }

    return components;
  }

  List<CircuitComponent> parseNetlistWithAnalysis(
    String netlist,
    Map<String, dynamic> analysisData,
  ) {
    analysisResults = CircuitAnalysis.fromJson(analysisData);
    _nodePositions.clear();

    final components = parseNetlist(netlist);
    _calculateNodePositions();
    return components;
  }

  void _calculateNodePositions() {
    for (var node in _nodeConnections.keys) {
      final connections = _nodeConnections[node]!;

      // Calculate average position of all components connected to this node
      double avgX = 0, avgY = 0;
      int count = 0;

      for (var conn in connections) {
        final componentId = conn.$1;
        final pointId = conn.$2; // '1' for left/top, '2' for right/bottom
        final componentIndex = int.parse(componentId.split('_')[1]);

        // Base position from grid layout
        double x = (componentIndex % 3) * 150 + 100;
        double y = (componentIndex ~/ 3) * 120 + 100;

        // Adjust for connection point
        if (pointId == '1') {
          // Left side connection
          avgX += x;
        } else {
          // Right side connection
          avgX += x + 60; // component width
        }
        avgY += y + 20; // vertical center of component
        count++;
      }

      if (count > 0) {
        _nodePositions[node] = Offset(avgX / count, avgY / count);
      }
    }
  }

  Offset? getNodePosition(String node) => _nodePositions[node];

  double? getNodeVoltage(String node) => analysisResults?.nodeVoltages[node];

  List<ConnectionData> getConnectionData() {
    final connections = <ConnectionData>[];
    final processedNodes = <String>{};

    for (var entry in _nodeConnections.entries) {
      final node = entry.key;
      if (processedNodes.contains(node)) continue;

      final connectedComponents = entry.value;
      for (var i = 0; i < connectedComponents.length; i++) {
        for (var j = i + 1; j < connectedComponents.length; j++) {
          connections.add(
            ConnectionData(
              sourceComponentId: connectedComponents[i].$1,
              sourcePointId: connectedComponents[i].$2,
              targetComponentId: connectedComponents[j].$1,
              targetPointId: connectedComponents[j].$2,
              node: node,
            ),
          );
        }
      }
      processedNodes.add(node);
    }

    return connections;
  }

  List<String> getConnectedComponents(String node) {
    final components = <String>{};
    if (_nodeConnections.containsKey(node)) {
      for (var conn in _nodeConnections[node]!) {
        components.add(conn.$1);
      }
    }
    return components.toList();
  }

  List<(String, String, String, String)> getNodeConnections(String node) {
    final connections = <(String, String, String, String)>[];
    if (_nodeConnections.containsKey(node)) {
      final connList = _nodeConnections[node]!;
      for (var i = 0; i < connList.length; i++) {
        for (var j = i + 1; j < connList.length; j++) {
          connections.add((
            connList[i].$1, // source component id
            connList[i].$2, // source point id
            connList[j].$1, // target component id
            connList[j].$2, // target point id
          ));
        }
      }
    }
    return connections;
  }

  ComponentType? _getComponentType(String designator) {
    switch (designator.toUpperCase()) {
      case 'R':
        return ComponentType.resistor;
      case 'C':
        return ComponentType.capacitor;
      case 'L':
        return ComponentType.inductor;
      case 'V':
        return ComponentType.voltageSource;
      case 'I':
        return ComponentType.currentSource;
      default:
        return null;
    }
  }

  CircuitComponent _createComponent(ComponentType type, String value) {
    const width = 60.0;
    const height = 40.0;

    String parsedValue = '';
    String unit = '';

    if (value.isNotEmpty) {
      final valueMatch = RegExp(r'([0-9.e-]+)([a-zA-Z]+)?').firstMatch(value);
      if (valueMatch != null) {
        parsedValue = valueMatch.group(1) ?? '';
        unit = valueMatch.group(2) ?? _getDefaultUnit(type);
        unit = _normalizeUnit(unit);
      }
    }

    final connectionPoints = [
      ConnectionPointData(id: '1', relativePosition: Offset(0, height / 2)),
      ConnectionPointData(id: '2', relativePosition: Offset(width, height / 2)),
    ];

    return CircuitComponent(
      type: type,
      width: width,
      height: height,
      connectionPoints: connectionPoints,
      value: parsedValue,
      unit: unit,
    );
  }

  String _getDefaultUnit(ComponentType type) {
    switch (type) {
      case ComponentType.resistor:
        return 'Ω';
      case ComponentType.capacitor:
        return 'F';
      case ComponentType.inductor:
        return 'H';
      case ComponentType.voltageSource:
        return 'V';
      case ComponentType.currentSource:
        return 'A';
      default:
        return '';
    }
  }

  String _normalizeUnit(String unit) {
    switch (unit.toLowerCase()) {
      case 'k':
      case 'kohm':
        return 'kΩ';
      case 'r':
      case 'ohm':
        return 'Ω';
      case 'f':
        return 'F';
      case 'u':
      case 'uf':
        return 'µF';
      case 'n':
      case 'nf':
        return 'nF';
      case 'p':
      case 'pf':
        return 'pF';
      case 'h':
        return 'H';
      case 'mh':
        return 'mH';
      case 'uh':
        return 'µH';
      case 'v':
        return 'V';
      case 'mv':
        return 'mV';
      case 'a':
        return 'A';
      case 'ma':
        return 'mA';
      default:
        return unit;
    }
  }
}

class ConnectionData {
  final String sourceComponentId;
  final String sourcePointId;
  final String targetComponentId;
  final String targetPointId;
  final String node;

  ConnectionData({
    required this.sourceComponentId,
    required this.sourcePointId,
    required this.targetComponentId,
    required this.targetPointId,
    required this.node,
  });
}
