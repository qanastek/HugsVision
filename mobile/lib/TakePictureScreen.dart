import 'package:flutter/cupertino.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'package:hugs_vision/PreviewScreen.dart';

class TakePictureScreen extends StatefulWidget {

  final bool onDevice;
  final String modelName;

  const TakePictureScreen({
    Key key,
    this.onDevice,
    this.modelName,
  }) : super(key: key);

  @override
  TakePictureScreenState createState() => TakePictureScreenState();
}

class TakePictureScreenState extends State<TakePictureScreen> {

  List<CameraDescription> cameras;
  CameraController controller;

  @override
  void initState() {
    super.initState();

    // Initialize the camera
    this.initCamera();
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  void initCamera() async {

    this.cameras = await availableCameras();

    controller = CameraController(cameras[0], ResolutionPreset.medium);

    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {});
    });
  }

  void _takePicture() async {

    try {
      final p = await getTemporaryDirectory();
      final name = DateTime.now();
      final path = "${p.path}/$name.png";

      await controller.takePicture(path).then((value) {

        print('here');
        print(path);

        Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => PreviewScreen(
              imgPath: path,
              fileName: "$name.png",
              onDevice: widget.onDevice,
              modelName: widget.modelName,
            ))
        );
      });

    } catch (e) {
      print(e);
    }
  }

  Widget cameraPreview() {

    if (controller == null || !controller.value.isInitialized) {
      return Text(
        'Loading',
        style: TextStyle(
            color: Colors.black,
            fontSize: 20.0,
            fontWeight: FontWeight.bold
        ),
      );
    }

    return AspectRatio(
      aspectRatio: controller.value.aspectRatio,
      child: CameraPreview(controller),
    );
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
              child: cameraPreview(), // The camera preview
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
                          color: Colors.orange,
                          child: Icon(
                            Icons.camera_alt,
                            color: Colors.black,
                          ),
                          onPressed: (){
                            _takePicture();
                          },
                        ),

                      ],
                    )
                ),
              ),
            )

          ],
        ),
      ),
    );
  }
}