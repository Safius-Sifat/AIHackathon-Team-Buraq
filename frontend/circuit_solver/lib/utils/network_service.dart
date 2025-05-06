import 'dart:convert';

import 'package:circuit_solver/constants/api.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart';

Future<String> pickImage() async {
  final picker = ImagePicker();
  final pickedFile = await picker.pickImage(source: ImageSource.gallery);
  if (pickedFile == null) throw Exception("No image selected");
  return pickedFile.path;
}

Future<Map<String, dynamic>> uploadImage(String path) async {
  var uri = Uri.parse(netlistUrl);
  var request = http.MultipartRequest('POST', uri);

  request.files.add(
    await http.MultipartFile.fromPath('file', path, filename: basename(path)),
  );

  var response = await request.send();

  final responseData = await http.Response.fromStream(response);

  print(responseData.body);
  return jsonDecode(responseData.body);
  // if (response.statusCode == 200) {
  //   print("Upload successful");
  //   print(responseData.body);
  // } else {
  //   print("Upload failed: ${response.statusCode}");
  //   throw Exception("Upload failed");
  // }
}
