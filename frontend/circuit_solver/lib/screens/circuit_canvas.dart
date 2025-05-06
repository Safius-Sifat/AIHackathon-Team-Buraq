import 'package:circuit_solver/screens/chat_screen.dart';
import 'package:flutter/material.dart';
import '../utils/network_service.dart';
import '../widgets/component_palette.dart';
import '../widgets/circuit_board.dart';
import '../models/circuit_component.dart';
import '../utils/netlist_parser.dart';
import '../widgets/overlay_loader.dart';

class CircuitCanvas extends StatefulWidget {
  const CircuitCanvas({super.key});

  @override
  State<CircuitCanvas> createState() => _CircuitCanvasState();
}

class _CircuitCanvasState extends State<CircuitCanvas> {
  List<PlacedComponent> _placedComponents = [];
  List<Wire> _wires = [];
  ConnectionPoint? _startPoint;
  bool _isDrawingWire = false;
  String? _selectedWireId;
  String? _selectedComponentId;
  NetlistParser? _parser;
  String? uploadImageId;

  final String _exampleCircuit = '''
{"netlist":"* Circuit Description\\nV1 1 0 DC 10V\\nR1 1 2 2\\nR2 2 0 3\\n.END","voltages":{"1":10.0,"2":6.0,"0":0.0}}
''';

  void _handleComponentPlaced(CircuitComponent component, Offset position) {
    setState(() {
      _placedComponents.add(
        PlacedComponent(
          component: component,
          position: position,
          id: DateTime.now().millisecondsSinceEpoch.toString(),
        ),
      );
    });
  }

  void _handleComponentDragged(String componentId, Offset newPosition) {
    setState(() {
      final index = _placedComponents.indexWhere(
        (comp) => comp.id == componentId,
      );
      if (index != -1) {
        // Update component position
        _placedComponents[index] = _placedComponents[index].copyWith(
          position: newPosition,
        );

        // Deselect everything while dragging
        _selectedWireId = null;
        _selectedComponentId = null;
        _isDrawingWire = false;
        _startPoint = null;

        // Reset selections
        _wires = _wires.map((w) => w.copyWith(isSelected: false)).toList();
        _placedComponents =
            _placedComponents
                .map((c) => c.copyWith(isSelected: false))
                .toList();
      }
    });
  }

  void _handleConnectionPointTapped(ConnectionPoint point) {
    setState(() {
      if (!_isDrawingWire) {
        // Start new wire
        _startPoint = point;
        _isDrawingWire = true;
        // Deselect any selected items when starting wire
        _selectedWireId = null;
        _selectedComponentId = null;
        _wires = _wires.map((w) => w.copyWith(isSelected: false)).toList();
        _placedComponents =
            _placedComponents
                .map((c) => c.copyWith(isSelected: false))
                .toList();
      } else if (_startPoint != null) {
        // Complete wire if connecting to different component
        if (_startPoint!.componentId != point.componentId) {
          final wireId = DateTime.now().millisecondsSinceEpoch.toString();
          _wires.add(
            Wire(startPoint: _startPoint!, endPoint: point, id: wireId),
          );
        }
        // Always reset wire drawing state
        _startPoint = null;
        _isDrawingWire = false;
      }
    });
  }

  void _handleWireTapped(String wireId) {
    setState(() {
      // Deselect if tapping outside (empty wireId) or tapping same wire
      if (wireId.isEmpty || wireId == _selectedWireId) {
        _selectedWireId = null;
        _wires = _wires.map((w) => w.copyWith(isSelected: false)).toList();
      } else {
        _selectedWireId = wireId;
        _wires =
            _wires.map((w) => w.copyWith(isSelected: w.id == wireId)).toList();
      }
    });
  }

  void _deleteSelectedWire() {
    if (_selectedWireId != null) {
      setState(() {
        _wires.removeWhere((wire) => wire.id == _selectedWireId);
        _selectedWireId = null;
      });
    }
  }

  void _handleComponentTapped(String componentId) {
    setState(() {
      if (_selectedComponentId == componentId) {
        _selectedComponentId = null;
        _placedComponents =
            _placedComponents
                .map((c) => c.copyWith(isSelected: false))
                .toList();
      } else {
        _selectedComponentId = componentId;
        _placedComponents =
            _placedComponents
                .map((c) => c.copyWith(isSelected: c.id == componentId))
                .toList();
        // Deselect wire when component is selected
        _selectedWireId = null;
        _wires = _wires.map((w) => w.copyWith(isSelected: false)).toList();
      }
    });
  }

  void _handleCanvasTap(Offset position) {
    setState(() {
      // Deselect component when tapping canvas
      if (_selectedComponentId != null) {
        _selectedComponentId = null;
        _placedComponents =
            _placedComponents
                .map((c) => c.copyWith(isSelected: false))
                .toList();
      }
      // Only cancel wire drawing, wire deselection is handled by CircuitBoard
      if (_isDrawingWire) {
        _startPoint = null;
        _isDrawingWire = false;
      }
    });
  }

  void _deleteSelectedComponent() {
    if (_selectedComponentId != null) {
      setState(() {
        // Remove all wires connected to this component
        _wires.removeWhere(
          (wire) =>
              wire.startPoint.componentId == _selectedComponentId ||
              wire.endPoint.componentId == _selectedComponentId,
        );
        // Remove the component
        _placedComponents.removeWhere(
          (comp) => comp.id == _selectedComponentId,
        );
        _selectedComponentId = null;
      });
    }
  }

  void _importNetlist(String netlistText) {
    try {
      final parser = NetlistParser();
      final components = parser.parseNetlist(netlistText);

      setState(() {
        // Clear existing circuit
        _placedComponents.clear();
        _wires.clear();
        _startPoint = null;
        _isDrawingWire = false;

        // Place components in a grid
        double x = 100;
        double y = 100;
        final componentMap = <String, String>{}; // Track component IDs

        for (var i = 0; i < components.length; i++) {
          final id = 'comp_$i';
          componentMap[id] = id;

          _placedComponents.add(
            PlacedComponent(
              component: components[i],
              position: Offset(x, y),
              id: id,
            ),
          );

          // Move next component position
          x += 120;
          if (x > MediaQuery.of(context).size.width - 150) {
            x = 100;
            y += 100;
          }
        }

        // Create wires based on netlist connections
        final connections = parser.getConnectionData();
        for (var conn in connections) {
          _createWireFromConnection(conn);
        }
      });

      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Circuit imported successfully')));
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error parsing netlist: $e')));
    }
  }

  void _importNetlistWithAnalysis(
    String netlist,
    Map<String, dynamic> analysisData,
  ) {
    try {
      _parser = NetlistParser();
      final components = _parser!.parseNetlistWithAnalysis(
        netlist,
        analysisData,
      );

      setState(() {
        // Clear existing circuit
        _placedComponents.clear();
        _wires.clear();
        _startPoint = null;
        _isDrawingWire = false;

        // Place components optimally
        _placeComponentsWithNodes(components);

        // Create all necessary wires including intersections
        _createAllConnections();
      });
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error creating circuit: $e')));
    }
  }

  void _placeComponentsWithNodes(List<CircuitComponent> components) {
    double x = 100;
    double y = 100;
    final Map<String, Offset> componentPositions = {};
    final Map<String, PlacedComponent> componentMap = {};

    // First pass: Place all components in a grid
    for (var i = 0; i < components.length; i++) {
      final id = 'comp_$i';
      final position = Offset(x, y);
      componentPositions[id] = position;

      final placedComponent = PlacedComponent(
        component: components[i],
        position: position,
        id: id,
      );
      _placedComponents.add(placedComponent);
      componentMap[id] = placedComponent;

      // Move to next position in grid
      x += 150; // Increased spacing for better visibility
      if (x > MediaQuery.of(context).size.width - 200) {
        x = 100;
        y += 120; // Increased vertical spacing
      }
    }

    // Second pass: Adjust positions based on node connections
    if (_parser != null) {
      for (var node in _parser!.nodes) {
        final nodePosition = _parser!.getNodePosition(node);
        if (nodePosition != null) {
          // Find all components connected to this node
          final connectedComponents = _parser!.getConnectedComponents(node);

          // Calculate average position for components sharing this node
          double avgX = 0, avgY = 0;
          int count = 0;

          for (var compId in connectedComponents) {
            if (componentPositions.containsKey(compId)) {
              avgX += componentPositions[compId]!.dx;
              avgY += componentPositions[compId]!.dy;
              count++;
            }
          }

          if (count > 0) {
            avgX /= count;
            avgY /= count;

            // Adjust component positions to be closer to their shared nodes
            for (var compId in connectedComponents) {
              if (componentMap.containsKey(compId)) {
                final oldPos = componentPositions[compId]!;
                final newPos = Offset(
                  (oldPos.dx + avgX) / 2,
                  (oldPos.dy + avgY) / 2,
                );

                // Update component position
                final index = _placedComponents.indexOf(componentMap[compId]!);
                if (index != -1) {
                  _placedComponents[index] = _placedComponents[index].copyWith(
                    position: newPos,
                  );
                  componentPositions[compId] = newPos;
                }
              }
            }
          }
        }
      }
    }
  }

  void _createAllConnections() {
    if (_parser == null) return;

    for (var node in _parser!.nodes) {
      final connections = _parser!.getNodeConnections(node);

      for (var connection in connections) {
        final (sourceId, sourcePointId, targetId, targetPointId) = connection;

        // Find the source and target components
        final sourceComponent = _placedComponents.firstWhere(
          (comp) => comp.id == sourceId,
          orElse: () => throw Exception('Source component not found'),
        );

        final targetComponent = _placedComponents.firstWhere(
          (comp) => comp.id == targetId,
          orElse: () => throw Exception('Target component not found'),
        );

        // Create wire between components
        _wires.add(
          Wire(
            startPoint: ConnectionPoint(
              componentId: sourceId,
              pointId: sourcePointId,
              absolutePosition: _getAbsolutePosition(
                sourceComponent,
                sourcePointId,
              ),
            ),
            endPoint: ConnectionPoint(
              componentId: targetId,
              pointId: targetPointId,
              absolutePosition: _getAbsolutePosition(
                targetComponent,
                targetPointId,
              ),
            ),
            id: 'wire_${node}_${DateTime.now().millisecondsSinceEpoch}',
          ),
        );
      }
    }
  }

  Offset _getAbsolutePosition(PlacedComponent component, String pointId) {
    final connectionPoint = component.component.connectionPoints.firstWhere(
      (p) => p.id == pointId,
    );

    return Offset(
      component.position.dx + connectionPoint.relativePosition.dx,
      component.position.dy + connectionPoint.relativePosition.dy,
    );
  }

  void _createWireFromConnection(ConnectionData connection) {
    // Find the source component and connection point
    final sourceComponent = _placedComponents.firstWhere(
      (comp) => comp.id == connection.sourceComponentId,
    );
    final sourcePoint = ConnectionPoint(
      componentId: sourceComponent.id,
      pointId: connection.sourcePointId,
      absolutePosition: Offset(
        sourceComponent.position.dx +
            sourceComponent.component.connectionPoints
                .firstWhere((p) => p.id == connection.sourcePointId)
                .relativePosition
                .dx,
        sourceComponent.position.dy +
            sourceComponent.component.connectionPoints
                .firstWhere((p) => p.id == connection.sourcePointId)
                .relativePosition
                .dy,
      ),
    );

    // Find the target component and connection point
    final targetComponent = _placedComponents.firstWhere(
      (comp) => comp.id == connection.targetComponentId,
    );
    final targetPoint = ConnectionPoint(
      componentId: targetComponent.id,
      pointId: connection.targetPointId,
      absolutePosition: Offset(
        targetComponent.position.dx +
            targetComponent.component.connectionPoints
                .firstWhere((p) => p.id == connection.targetPointId)
                .relativePosition
                .dx,
        targetComponent.position.dy +
            targetComponent.component.connectionPoints
                .firstWhere((p) => p.id == connection.targetPointId)
                .relativePosition
                .dy,
      ),
    );

    // Create the wire
    _wires.add(
      Wire(
        startPoint: sourcePoint,
        endPoint: targetPoint,
        id: 'wire_${connection.node}_${DateTime.now().millisecondsSinceEpoch}',
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    print(uploadImageId);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Qasim'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: () async {
              try {
                final path = await pickImage();
                showOverlayLoader(context);
                // final Map<String, dynamic> data = jsonDecode(_exampleCircuit);
                final Map<String, dynamic> data = await uploadImage(path);
                Navigator.of(context).pop(); // Close the loader
                // Process the escaped newlines before passing to parser
                uploadImageId = data['imageId'];
                final String netlist = data['netlist'].replaceAll(r'\n', '\n');
                _importNetlistWithAnalysis(netlist, {
                  'voltages': data['voltages'],
                });
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Circuit imported successfully'),
                  ),
                );
              } catch (e) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Error importing circuit: $e')),
                );
                Navigator.of(context).pop(); // Close the loader
              }
            },
            tooltip: 'Import Example Circuit',
          ),
          if (_selectedComponentId != null)
            IconButton(
              icon: const Icon(Icons.delete),
              onPressed: _deleteSelectedComponent,
              tooltip: 'Delete selected component',
            )
          else if (_selectedWireId != null)
            IconButton(
              icon: const Icon(Icons.delete),
              onPressed: _deleteSelectedWire,
              tooltip: 'Delete selected wire',
            ),
          IconButton(
            icon: const Icon(Icons.chat_outlined),
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder:
                      (context) => ChatScreen(
                        hasCircuitImage: uploadImageId != null,
                        imageId: uploadImageId,
                      ),
                ),
              );
              // _showHelpDialog(context);
            },
            tooltip: 'Chat',
          ),
          IconButton(
            icon: const Icon(Icons.delete),
            onPressed: () {
              setState(() {
                _placedComponents.clear();
                _wires.clear();
                _startPoint = null;
                _isDrawingWire = false;
                uploadImageId = null;
              });
            },
            tooltip: 'Clear Canvas',
          ),
        ],
      ),
      body: Column(
        children: [
          if (_isDrawingWire)
            Container(
              color: Colors.blue.shade100,
              padding: EdgeInsets.symmetric(vertical: 6, horizontal: 16),
              child: Row(
                children: [
                  Icon(
                    Icons.info_outline,
                    size: 18,
                    color: Colors.blue.shade800,
                  ),
                  SizedBox(width: 8),
                  Flexible(
                    child: Text(
                      'Tap another connection point to complete the wire',
                      style: TextStyle(
                        color: Colors.blue.shade800,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          Expanded(
            child: InteractiveViewer(
              child: CircuitBoard(
                placedComponents: _placedComponents,
                wires: _wires,
                startPoint: _startPoint,
                isDrawingWire: _isDrawingWire,
                onConnectionPointTapped: _handleConnectionPointTapped,
                onWireTapped: _handleWireTapped,
                onComponentTapped: _handleComponentTapped,
                onComponentDragged: _handleComponentDragged,
                onComponentPlaced: _handleComponentPlaced,
                showGrid: true,
                gridSize: 20.0,
                parser: _parser, // Pass the parser instance
              ),
            ),
          ),
          ComponentPalette(onComponentDragged: _handleComponentPlaced),
        ],
      ),
    );
  }

  void _showHelpDialog(BuildContext context) {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: const Text('How to Use the Circuit Designer'),
            content: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: const [
                  Text(
                    '1. Tap and hold a component from the palette and drag it to the canvas',
                  ),
                  SizedBox(height: 8),
                  Text(
                    '2. Tap on a blue connection point, then tap on another connection point to create a wire',
                  ),
                  SizedBox(height: 8),
                  Text('3. Tap empty space to cancel wire creation'),
                  SizedBox(height: 8),
                  Text('4. Tap and drag components to reposition them'),
                  SizedBox(height: 8),
                  Text('5. Tap on a wire to delete it'),
                ],
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('Got it'),
              ),
            ],
          ),
    );
  }
}
