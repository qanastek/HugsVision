import 'dart:io';

import 'package:flutter/material.dart';
import 'package:hugs_vision/TakePictureScreen.dart';

import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:pytorch_mobile/enums/dtype.dart';
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HugsVision HAM10000',
      theme: ThemeData(
        primarySwatch: Colors.orange,
      ),
      home: MyHomePage(title: 'HugsVision HAM10000'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);

  final String title;
  static String classValue = "Aucune";

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {

  ImagePicker _picker = ImagePicker();
  bool onDeviceStatus = true;

  var models = [
    "VGG16",
    "DeiT",
    "DenseNet121",
    "MobileNetV2",
    "ShuffleNetV2"
  ];

  String modelName = "DenseNet121";

  void _updateVue() {
    setState(() {
      MyHomePage.classValue = MyHomePage.classValue;
    });
  }

  // Take a picture
  void _takePicture() {
    Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => TakePictureScreen(onDevice: onDeviceStatus, modelName: modelName))
    ).then((value) => {
      _updateVue()
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Container(
        margin: const EdgeInsets.only(top: 35.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: <Widget>[

            Container(
              margin: const EdgeInsets.only(left: 35.0),
              alignment: Alignment.topLeft,
              child: Column(
                children: [

                  Row(
                    children: [
                      Text(
                        "Computation:",
                        style: TextStyle(
                          color: Colors.black,
                          fontSize: 20,

                        ),
                      ),
                    ],
                  ),

                  ListTile(
                    title: Text("On device"),
                    leading: Radio(
                      value: true,
                      groupValue: onDeviceStatus,
                      onChanged: (value) {
                        setState(() {
                          onDeviceStatus = value;
                          print(onDeviceStatus);
                        });
                      },
                      activeColor: Colors.green,
                    ),
                  ),

                  ListTile(
                    title: Text("Online"),
                    leading: Radio(
                      value: false,
                      groupValue: onDeviceStatus,
                      onChanged: (value) {
                        setState(() {
                          onDeviceStatus = value;
                          print(onDeviceStatus);
                        });
                      },
                      activeColor: Colors.green,
                    ),
                  ),

                  Container(
                      color: Colors.white,
                      child: (onDeviceStatus == true) ? Text("") : Column(
                        children: [

                          Row(
                            children: [
                              Text(
                                "Models:",
                                style: TextStyle(
                                  color: Colors.black,
                                  fontSize: 20,

                                ),
                              ),
                            ],
                          ),

                          for(String model in models) ListTile(
                            title: Text(model),
                            leading: Radio(
                              value: model,
                              groupValue: modelName,
                              onChanged: (value) {
                                setState(() {
                                  modelName = value;
                                  print(modelName);
                                });
                              },
                              activeColor: Colors.green,
                            ),
                          ),

                        ],
                      )
                  ),

                  Row(
                    children: [

                      Text(
                        "Predicted class: ",
                        style: TextStyle(
                          color: Colors.black,
                          fontSize: 20,
                        ),
                      ),

                      Text(
                        MyHomePage.classValue,
                        style: TextStyle(
                          color: Colors.black,
                          fontSize: 20,
                          fontWeight: FontWeight.w500,
                        ),
                      ),

                    ],
                  ),

                ],
              ),
            ),

          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _takePicture,
        tooltip: 'Take a picture',
        child: Icon(Icons.photo_camera_outlined),
      ),
    );
  }
}
