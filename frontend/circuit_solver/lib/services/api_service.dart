import 'dart:convert';

import 'package:http/http.dart' as http;
import '../constants/api.dart';

class ApiService {
  Future<String> sendChatMessage(String question, String imageId) async {
    try {
      var request = http.MultipartRequest("POST", Uri.parse(chatUrl));
      request.fields['question'] = question;
      request.fields['image_id'] = imageId;

      final response = await request.send();

      final responseData = await http.Response.fromStream(response);
      if (response.statusCode == 200) {
        return jsonDecode(responseData.body)['response'];
      } else {
        throw Exception('Failed to send message');
      }
    } catch (e) {
      throw Exception('Error: $e');
    }
  }
}
