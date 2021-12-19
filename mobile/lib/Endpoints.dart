import 'package:http/http.dart' as http;

class Endpoints {

  static String sendScan(modelName) {
    return "http://192.168.0.29:5000/predict/" + modelName;
  }
}