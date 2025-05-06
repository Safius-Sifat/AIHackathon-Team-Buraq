import 'package:flutter/material.dart';

enum ComponentType {
  resistor,
  capacitor,
  inductor,
  voltageSource,
  currentSource,
  ground,
}

class ConnectionPointData {
  final String id;
  final Offset relativePosition;

  ConnectionPointData({required this.id, required this.relativePosition});
}

class ConnectionPoint {
  final String componentId;
  final String pointId;
  final Offset? absolutePosition;

  ConnectionPoint({
    required this.componentId,
    required this.pointId,
    this.absolutePosition,
  });
}

class CircuitComponent {
  final ComponentType type;
  final double width;
  final double height;
  final List<ConnectionPointData> connectionPoints;
  final String value; // Add value field
  final String unit; // Add unit field

  CircuitComponent({
    required this.type,
    required this.width,
    required this.height,
    required this.connectionPoints,
    this.value = '', // Default empty value
    this.unit = '', // Default empty unit
  });
}

class PlacedComponent {
  final CircuitComponent component;
  final Offset position;
  final String id;
  final bool isSelected;

  PlacedComponent({
    required this.component,
    required this.position,
    required this.id,
    this.isSelected = false,
  });

  PlacedComponent copyWith({
    CircuitComponent? component,
    Offset? position,
    String? id,
    bool? isSelected,
  }) {
    return PlacedComponent(
      component: component ?? this.component,
      position: position ?? this.position,
      id: id ?? this.id,
      isSelected: isSelected ?? this.isSelected,
    );
  }
}

class Wire {
  final ConnectionPoint startPoint;
  final ConnectionPoint endPoint;
  final String id;
  final bool isSelected;

  Wire({
    required this.startPoint,
    required this.endPoint,
    required this.id,
    this.isSelected = false,
  });

  Wire copyWith({
    ConnectionPoint? startPoint,
    ConnectionPoint? endPoint,
    String? id,
    bool? isSelected,
  }) {
    return Wire(
      startPoint: startPoint ?? this.startPoint,
      endPoint: endPoint ?? this.endPoint,
      id: id ?? this.id,
      isSelected: isSelected ?? this.isSelected,
    );
  }
}
