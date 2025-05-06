import 'package:flutter/material.dart';
import '../models/circuit_component.dart';

class ComponentPalette extends StatelessWidget {
  final Function(CircuitComponent, Offset) onComponentDragged;

  const ComponentPalette({super.key, required this.onComponentDragged});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 100,
      decoration: BoxDecoration(
        border: Border(top: BorderSide(color: Colors.grey.shade300)),
        color: Colors.grey.shade100,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(vertical: 4),
            color: Colors.blue,
            child: const Text(
              'Component Palette',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          Expanded(
            child: ListView(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.all(8),
              children:
                  ComponentType.values.map((type) {
                    return _buildDraggableComponent(context, type);
                  }).toList(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDraggableComponent(BuildContext context, ComponentType type) {
    final component = _createComponentFromType(type);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      child: LongPressDraggable<CircuitComponent>(
        data: component,
        // Make dragging more responsive
        delay: const Duration(milliseconds: 100),
        feedback: Material(
          elevation: 4,
          child: Container(
            width: component.width,
            height: component.height,
            decoration: BoxDecoration(
              color: Colors.white,
              border: Border.all(color: Colors.black),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Center(child: _getComponentIcon(type)),
          ),
        ),
        childWhenDragging: Opacity(
          opacity: 0.5,
          child: _buildComponentIcon(type),
        ),
        child: _buildComponentIcon(type),
        onDragEnd: (details) {
          // Handled by CircuitBoard
        },
      ),
    );
  }

  Widget _buildComponentIcon(ComponentType type) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: Colors.white,
            border: Border.all(color: Colors.grey.shade400),
            borderRadius: BorderRadius.circular(8),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.1),
                blurRadius: 2,
                offset: const Offset(0, 1),
              ),
            ],
          ),
          child: Center(
            child: Icon(
              _getIconForComponentType(type),
              color: _getIconColorForType(type),
              size: 24,
            ),
          ),
        ),
        const SizedBox(height: 4),
        Text(type.name, style: const TextStyle(fontSize: 10)),
      ],
    );
  }

  Widget _getComponentIcon(ComponentType type) {
    return Icon(
      _getIconForComponentType(type),
      color: _getIconColorForType(type),
      size: 28,
    );
  }

  Color _getIconColorForType(ComponentType type) {
    switch (type) {
      case ComponentType.resistor:
        return Colors.orange.shade700;
      case ComponentType.capacitor:
        return Colors.blue.shade600;
      case ComponentType.inductor:
        return Colors.purple.shade600;
      case ComponentType.voltageSource:
        return Colors.red.shade600;
      case ComponentType.currentSource:
        return Colors.amber.shade800;
      case ComponentType.ground:
        return Colors.blue.shade800;
      default:
        return Colors.black87;
    }
  }

  CircuitComponent _createComponentFromType(ComponentType type) {
    // Make components smaller
    double width = 60.0; // Smaller width (was 80.0)
    double height = 40.0; // Smaller height (was 60.0)

    List<ConnectionPointData> connectionPoints = [];

    switch (type) {
      case ComponentType.resistor:
        connectionPoints = [
          ConnectionPointData(id: '1', relativePosition: Offset(0, height / 2)),
          ConnectionPointData(
            id: '2',
            relativePosition: Offset(width, height / 2),
          ),
        ];
        break;
      case ComponentType.capacitor:
        connectionPoints = [
          ConnectionPointData(id: '1', relativePosition: Offset(0, height / 2)),
          ConnectionPointData(
            id: '2',
            relativePosition: Offset(width, height / 2),
          ),
        ];
        break;
      case ComponentType.inductor:
        connectionPoints = [
          ConnectionPointData(id: '1', relativePosition: Offset(0, height / 2)),
          ConnectionPointData(
            id: '2',
            relativePosition: Offset(width, height / 2),
          ),
        ];
        break;
      case ComponentType.voltageSource:
        connectionPoints = [
          ConnectionPointData(id: '1', relativePosition: Offset(width / 2, 0)),
          ConnectionPointData(
            id: '2',
            relativePosition: Offset(width / 2, height),
          ),
        ];
        break;
      case ComponentType.currentSource:
        connectionPoints = [
          ConnectionPointData(id: '1', relativePosition: Offset(width / 2, 0)),
          ConnectionPointData(
            id: '2',
            relativePosition: Offset(width / 2, height),
          ),
        ];
        break;
      case ComponentType.ground:
        connectionPoints = [
          ConnectionPointData(id: '1', relativePosition: Offset(width / 2, 0)),
        ];
        break;
    }

    return CircuitComponent(
      type: type,
      width: width,
      height: height,
      connectionPoints: connectionPoints,
    );
  }

  IconData _getIconForComponentType(ComponentType type) {
    switch (type) {
      case ComponentType.resistor:
        return Icons.power;
      case ComponentType.capacitor:
        return Icons.battery_full;
      case ComponentType.inductor:
        return Icons.loop;
      case ComponentType.voltageSource:
        return Icons.electrical_services;
      case ComponentType.currentSource:
        return Icons.electric_bolt;
      case ComponentType.ground:
        return Icons.arrow_downward;
      default:
        return Icons.device_unknown;
    }
  }
}
