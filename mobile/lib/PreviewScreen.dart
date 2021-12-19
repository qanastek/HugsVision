import 'dart:io';
import 'dart:typed_data';
import 'package:hugs_vision/Endpoints.dart';
import 'package:hugs_vision/main.dart';
import 'package:path/path.dart';
import 'package:async/async.dart';
import 'package:http/http.dart' as http;
import 'package:pytorch_mobile/model.dart';
import 'dart:io' as Io;
import 'dart:convert';

import 'package:share/share.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

import 'package:pytorch_mobile/pytorch_mobile.dart';

class PreviewScreen extends StatefulWidget {
  final String imgPath;
  final String fileName;
  final bool onDevice;
  final String modelName;
  PreviewScreen({this.imgPath, this.fileName, this.onDevice, this.modelName});

  @override
  _PreviewScreenState createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {

  // Upload the picture to the server
  upload(String imageFile, BuildContext ctx, bool onDevice) async {

    // Display the local Path
    print("Image Path: " + imageFile);

    print("-----------------");
    print(onDevice);
    print("-----------------");

    if(onDevice) {

      Model customModel = await PyTorchMobile.loadModel('assets/models/model.pt');
      var prediction = await customModel.getImagePrediction(File(imageFile), 224, 224, "assets/models/labels.csv");

      print(prediction);
      setState(() {
        MyHomePage.classValue = prediction;
      });

    }
    else {

      print("Server URL: " + Endpoints.sendScan(widget.modelName));

      var postUri = Uri.parse(Endpoints.sendScan(widget.modelName));
      http.MultipartRequest request = new http.MultipartRequest("POST", postUri);

      http.MultipartFile multipartFile = await http.MultipartFile.fromPath('file', imageFile);
      request.files.add(multipartFile);

      http.StreamedResponse response = await request.send();
      var responseString = await response.stream.bytesToString();
      Map<String, dynamic> res = jsonDecode(responseString);

      print(res);
      print(res["label"]);
      setState(() {
        MyHomePage.classValue = res["label"];
      });

      // If founded
      if(response.statusCode == 200) {
        print("found");
      } else {
        print("not found");
      }

    }

    Navigator.pop(ctx);
    Navigator.pop(ctx);

  }

  Future getBytes () async {
    Uint8List bytes = File(widget.imgPath).readAsBytesSync();
    return ByteData.view(bytes.buffer);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          automaticallyImplyLeading: true,
        ),
        body: Container(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              Expanded(
                flex: 2,
                child: Image.file(File(widget.imgPath),fit: BoxFit.cover,),
              ),
              Align(
                alignment: Alignment.bottomCenter,
                child: Container(
                  width: double.infinity,
                  height: 60,
                  color: Colors.black,
                  child: Center(
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        mainAxisSize: MainAxisSize.max,
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [

                          RaisedButton(
                            color: Color(0xFF333F4D),
                            child: Text(
                              "Start again",
                              style: TextStyle(
                                color: Colors.white,
                              ),
                            ),
                            onPressed: (){
                              // Go back
                              Navigator.pop(context);
                            },
                          ),

                          RaisedButton(
                            color: Colors.orange,
                            child: Text(
                              "Use the photo",
                              style: TextStyle(
                                color: Colors.black,
                              ),
                            ),
                            onPressed: (){
                              getBytes().then((bytes) {
                                print('here now');
                                print(widget.imgPath);
                                this.upload(widget.imgPath, context, widget.onDevice);
                              });
                            },
                          ),

                        ],
                      )
                  ),
                ),
              )
            ],
          ),
        )
    );
  }
}