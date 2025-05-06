import 'package:flutter/material.dart';
import '../models/circuit_component.dart';
import '../utils/netlist_parser.dart' show NetlistParser;

class CircuitBoard extends StatelessWidget {
  final List<PlacedComponent> placedComponents;
  final List<Wire> wires;
  final ConnectionPoint? startPoint;
  final bool isDrawingWire;
  final Function(ConnectionPoint) onConnectionPointTapped;
  final Function(String) onWireTapped;
  final Function(String) onComponentTapped;
  final Function(String, Offset) onComponentDragged;
  final Function(CircuitComponent, Offset) onComponentPlaced;
  final bool showGrid;
  final double gridSize;
  final NetlistParser? parser; // Add parser field

  const CircuitBoard({
    super.key,
    required this.placedComponents,
    required this.wires,
    this.startPoint,
    this.isDrawingWire = false,
    required this.onConnectionPointTapped,
    required this.onWireTapped,
    required this.onComponentTapped,
    required this.onComponentDragged,
    required this.onComponentPlaced,
    this.showGrid = true,
    this.gridSize = 20.0,
    this.parser, // Add parser parameter
  });

  Widget _buildNodeVoltage(
    String node,
    Offset position,
    BuildContext context,
    double? voltage,
  ) {
    if (voltage == null) return Container();

    final color = voltage > 0 ? Colors.red.shade700 : Colors.blue.shade700;

    return Positioned(
      left: position.dx - 15, // Center horizontally
      top: position.dy - 25, // Show above the connection point
      child: GestureDetector(
        onTap: () {
          // Show voltage in dialog
          showDialog(
            context: context,
            builder:
                (context) => AlertDialog(
                  title: Text('Node $node'),
                  content: Text('Voltage: ${voltage.toStringAsFixed(2)}V'),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.of(context).pop(),
                      child: const Text('Close'),
                    ),
                  ],
                ),
          );
        },
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(4),
            border: Border.all(color: color.withOpacity(0.5)),
            boxShadow: [
              BoxShadow(
                color: color.withOpacity(0.2),
                blurRadius: 2,
                spreadRadius: 1,
              ),
            ],
          ),
          child: Text(
            '${voltage.toStringAsFixed(1)}V',
            style: TextStyle(
              fontSize: 10,
              color: color,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return DragTarget<CircuitComponent>(
      builder: (context, candidateData, rejectedData) {
        return Stack(
          children: [
            // Grid and base layout
            GestureDetector(
              onTapDown: (details) {
                // Deselect wire if tapping outside any wire
                bool tappedOnWire = false;
                for (var wire in wires) {
                  final startPos = _getConnectionPointPosition(wire.startPoint);
                  final endPos = _getConnectionPointPosition(wire.endPoint);
                  if (_isPointNearWire(
                    details.localPosition,
                    startPos,
                    endPos,
                  )) {
                    tappedOnWire = true;
                    break;
                  }
                }
                if (!tappedOnWire) {
                  onWireTapped(''); // Empty string to indicate deselection
                }
              },
              child: CustomPaint(
                painter: GridPainter(gridSize: gridSize, showGrid: showGrid),
                child: Stack(
                  clipBehavior: Clip.none,
                  children: [
                    // Draw wires first
                    ...wires.map((wire) => _buildWire(wire)),

                    // Draw components
                    ...placedComponents.map(
                      (component) =>
                          _buildDraggableComponent(component, context),
                    ),
                    // Draw node voltages last so they're on top
                    if (parser != null)
                      for (var node in parser!.nodes)
                        for (var comp in placedComponents)
                          ...comp.component.connectionPoints
                              .where((point) {
                                // Find connection points that match this node
                                final connections =
                                    parser!.nodeConnections[node] ?? [];
                                return connections.any(
                                  (conn) =>
                                      conn.$1 == comp.id && conn.$2 == point.id,
                                );
                              })
                              .map((point) {
                                final voltage = parser!.getNodeVoltage(node);
                                if (voltage != null) {
                                  final pos = Offset(
                                    comp.position.dx +
                                        point.relativePosition.dx,
                                    comp.position.dy +
                                        point.relativePosition.dy,
                                  );
                                  return _buildNodeVoltage(
                                    node,
                                    pos,
                                    context,
                                    voltage,
                                  );
                                }
                                return Container();
                              }),
                  ],
                ),
              ),
            ),
          ],
        );
      },
      onAcceptWithDetails: (details) {
        final component = details.data;
        final RenderBox box = context.findRenderObject() as RenderBox;
        final Offset localPosition = box.globalToLocal(details.offset);

        final snappedPosition = Offset(
          (localPosition.dx / gridSize).round() * gridSize,
          (localPosition.dy / gridSize).round() * gridSize,
        );

        onComponentPlaced(component, snappedPosition);
      },
      onWillAcceptWithDetails: (data) => data != null,
    );
  }

  Widget _buildDraggableComponent(
    PlacedComponent component,
    BuildContext context,
  ) {
    return Positioned(
      left: component.position.dx,
      top: component.position.dy,
      child: Material(
        color: Colors.transparent,
        child: Draggable<String>(
          data: component.id,
          maxSimultaneousDrags: 1,
          hitTestBehavior: HitTestBehavior.translucent,
          feedback: Opacity(
            opacity: 0.7,
            child: _buildComponentWidget(component, context),
          ),
          childWhenDragging: Opacity(
            opacity: 0.3,
            child: _buildComponentWidget(component, context),
          ),
          onDragEnd: (details) {
            final RenderBox renderBox = context.findRenderObject() as RenderBox;
            final localPosition = renderBox.globalToLocal(details.offset);

            // Snap to grid
            final snappedPosition = Offset(
              (localPosition.dx / gridSize).round() * gridSize,
              (localPosition.dy / gridSize).round() * gridSize,
            );

            onComponentDragged(component.id, snappedPosition);
          },
          child: _buildComponentWidget(component, context),
        ),
      ),
    );
  }

  Widget _buildComponentWidget(
    PlacedComponent component,
    BuildContext context,
  ) {
    return GestureDetector(
      behavior:
          HitTestBehavior.translucent, // Add this to improve touch detection
      onTap: () => onComponentTapped(component.id),
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Container(
            width: component.component.width,
            height: component.component.height + 2,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.7),
              borderRadius: BorderRadius.circular(2),
              border: Border.all(
                color: component.isSelected ? Colors.blue : Colors.transparent,
                width: 2,
              ),
              boxShadow:
                  component.isSelected
                      ? [
                        BoxShadow(
                          color: Colors.blue.withOpacity(0.3),
                          spreadRadius: 2,
                          blurRadius: 4,
                        ),
                      ]
                      : null,
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _getComponentIcon(component.component.type),
                if (component.component.value.isNotEmpty)
                  Text(
                    '${component.component.value}${component.component.unit}',
                    style: TextStyle(
                      fontSize: 10,
                      color: Colors.black87,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
              ],
            ),
          ),

          // Add connection points as separate elements in the stack
          ...component.component.connectionPoints.map(
            (point) => Positioned(
              left:
                  point.relativePosition.dx -
                  8, // Adjust for connection point size
              top:
                  point.relativePosition.dy -
                  8, // Adjust for connection point size
              child: GestureDetector(
                // Wrap connection point in GestureDetector
                behavior: HitTestBehavior.opaque,
                onTap:
                    () => onConnectionPointTapped(
                      ConnectionPoint(
                        componentId: component.id,
                        pointId: point.id,
                        absolutePosition: Offset(
                          component.position.dx + point.relativePosition.dx,
                          component.position.dy + point.relativePosition.dy,
                        ),
                      ),
                    ),
                child: _buildConnectionPoint(
                  ConnectionPoint(
                    componentId: component.id,
                    pointId: point.id,
                    absolutePosition: Offset(
                      component.position.dx + point.relativePosition.dx,
                      component.position.dy + point.relativePosition.dy,
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _getComponentIcon(ComponentType type) {
    IconData iconData;
    double size = 24.0; // Make smaller (reduced from 32.0)
    Color color = Colors.black87;

    switch (type) {
      case ComponentType.resistor:
        iconData = Icons.power;
        color = Colors.orange.shade700;
        break;
      case ComponentType.capacitor:
        iconData = Icons.battery_full;
        color = Colors.blue.shade600;
        break;
      case ComponentType.inductor:
        iconData = Icons.loop;
        color = Colors.purple.shade600;
        break;
      case ComponentType.voltageSource:
        iconData = Icons.electrical_services;
        color = Colors.red.shade600;
        break;
      case ComponentType.currentSource:
        iconData = Icons.electric_bolt;
        color = Colors.amber.shade800;
        break;
      case ComponentType.ground:
        iconData = Icons.arrow_downward;
        color = Colors.blue.shade700;
        break;
      default:
        iconData = Icons.device_unknown;
    }

    return Icon(iconData, size: size, color: color);
  }

  Widget _buildConnectionPoint(ConnectionPoint point) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        splashColor: Colors.blue.withOpacity(0.3),
        highlightColor: Colors.blue.withOpacity(0.2),
        onTap: () => onConnectionPointTapped(point),
        child: Container(
          width: 16,
          height: 16,
          decoration: BoxDecoration(
            color:
                isDrawingWire && startPoint?.pointId == point.pointId
                    ? Colors.red
                    : Colors.blue,
            shape: BoxShape.circle,
            border: Border.all(color: Colors.black, width: 1),
          ),
        ),
      ),
    );
  }

  Widget _buildWire(Wire wire) {
    final startPos = _getConnectionPointPosition(wire.startPoint);
    final endPos = _getConnectionPointPosition(wire.endPoint);

    return CustomPaint(
      painter: WirePainter(
        startPoint: startPos,
        endPoint: endPos,
        isSelected: wire.isSelected,
        gridSize: gridSize,
      ),
      child: GestureDetector(
        behavior: HitTestBehavior.translucent,
        onTapDown: (details) {
          if (_isPointNearWire(details.localPosition, startPos, endPos)) {
            onWireTapped(wire.id);
          }
        },
      ),
    );
  }

  bool _isPointNearWire(Offset point, Offset start, Offset end) {
    // Calculate distance from point to wire segment
    final wireLength = (end - start).distance;
    final t =
        ((point - start).dx * (end - start).dx +
            (point - start).dy * (end - start).dy) /
        (wireLength * wireLength);

    if (t < 0.0) return (point - start).distance < 10.0;
    if (t > 1.0) return (point - end).distance < 10.0;

    final projection = start + (end - start) * t;
    return (point - projection).distance < 10.0;
  }

  Offset _getConnectionPointPosition(ConnectionPoint point) {
    final component = placedComponents.firstWhere(
      (comp) => comp.id == point.componentId,
    );

    final connectionPoint = component.component.connectionPoints.firstWhere(
      (p) => p.id == point.pointId,
    );

    return Offset(
      component.position.dx + connectionPoint.relativePosition.dx,
      component.position.dy + connectionPoint.relativePosition.dy,
    );
  }
}

class GridPainter extends CustomPainter {
  final double gridSize;
  final bool showGrid;

  GridPainter({required this.gridSize, this.showGrid = true});

  @override
  void paint(Canvas canvas, Size size) {
    if (!showGrid) return;

    final paint =
        Paint()
          ..color = Colors.grey.withOpacity(0.3)
          ..strokeWidth = 0.5;

    for (double i = 0; i <= size.width; i += gridSize) {
      canvas.drawLine(Offset(i, 0), Offset(i, size.height), paint);
    }

    for (double i = 0; i <= size.height; i += gridSize) {
      canvas.drawLine(Offset(0, i), Offset(size.width, i), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class WirePainter extends CustomPainter {
  final Offset startPoint;
  final Offset endPoint;
  final bool isSelected;
  final double gridSize;
  final VoidCallback? onTap;

  WirePainter({
    required this.startPoint,
    required this.endPoint,
    this.isSelected = false,
    this.gridSize = 20.0,
    this.onTap,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Draw wire path
    final paint =
        Paint()
          ..color = isSelected ? Colors.blue.shade700 : Colors.black
          ..strokeWidth = isSelected ? 2.5 : 1.5
          ..style = PaintingStyle.stroke;

    // Create manhattan path
    final path = Path();
    path.moveTo(startPoint.dx, startPoint.dy);

    // Calculate midpoint for manhattan routing
    final midX = startPoint.dx + (endPoint.dx - startPoint.dx) / 2;

    // Draw path
    path.lineTo(midX, startPoint.dy);
    path.lineTo(midX, endPoint.dy);
    path.lineTo(endPoint.dx, endPoint.dy);

    canvas.drawPath(path, paint);

    // Draw selection highlight
    if (isSelected) {
      final highlightPaint =
          Paint()
            ..color = Colors.blue.withOpacity(0.2)
            ..strokeWidth = 8.0
            ..style = PaintingStyle.stroke;
      canvas.drawPath(path, highlightPaint);
    }
  }

  @override
  bool shouldRepaint(covariant WirePainter oldDelegate) {
    return oldDelegate.startPoint != startPoint ||
        oldDelegate.endPoint != endPoint ||
        oldDelegate.isSelected != isSelected;
  }
}
