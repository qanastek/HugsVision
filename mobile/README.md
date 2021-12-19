# HugsVision Mobile

This is a Flutter 2.0 application for demonstrating mobile device capabilities for the skin cancer identification task using the [docker image](https://hub.docker.com/repository/docker/qanastek/hugsvision-api-cpu-only) of the HugsVision inference API (CPU Only) and [Pytorch Mobile](https://pub.dev/packages/pytorch_mobile).

## Commands

Run the API server:

```dart
docker pull qanastek/hugsvision-api-cpu-only
docker run -d --name huggingface_api -p 80:5000 qanastek/hugsvision-api-cpu-only
```

Run the app:

* Flutter Pub Get
* 

Generate icons:

```dart
flutter packages pub run flutter_launcher_icons:main
```
