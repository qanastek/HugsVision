# HugsVision docker image with CPU-only PyTorch API backend for HAM10000 Skin Cancer

## Download Docker Image

Available at : [`https://hub.docker.com/repository/docker/qanastek/hugsvision-api-cpu-only`](https://hub.docker.com/repository/docker/qanastek/hugsvision-api-cpu-only)

## Swagger Documentation

Available at : [`http://localhost:80/docs`](http://localhost:80/docs)

## Models Available in the image

Due to size restrictions, we only left :

```plain
DenseNet121
MobileNetV2
ShuffleNetV2
```

## Build & Run Docker Image

Build : `docker build --tag hugsvision:latest .`

Run : `docker run -p 80:8888 hugsvision:latest`

## Send request to the API from Python

```python
requests.post(
    f"http://127.0.0.1:8888/predict/ShuffleNetV2",
    files = {
        'file': open("images/akiec.jpg", 'rb')
    }
).json()
```

## Models Specification Sheet

<html style="color: white;">
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-v0zy{background-color:#efefef;color:#000000;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-4jb6{background-color:#ffffff;color:#333333;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
<tr>
<th class="tg-v0zy">Model</th>
<th class="tg-v0zy">Accuracy</th>
<th class="tg-v0zy">Size</th>
</tr>
</thead>
<tbody>
<tr>
<td class="tg-4jb6">VGG16</td>
<td class="tg-4jb6">38.27%</td>
<td class="tg-4jb6">512.0 MB</td>
</tr>
<tr>
<td class="tg-4jb6">DeiT</td>
<td class="tg-4jb6">71.60%</td>
<td class="tg-4jb6">327.0 MB</td>
</tr>
<tr>
<td class="tg-4jb6">DenseNet121</td>
<td class="tg-4jb6">77.78%</td>
<td class="tg-4jb6">27.1 MB</td>
</tr>
<tr>
<td class="tg-4jb6">MobileNetV2</td>
<td class="tg-4jb6">75.31%</td>
<td class="tg-4jb6">8.77 MB</td>
</tr>
<tr>
<td class="tg-4jb6">ShuffleNetV2</td>
<td class="tg-4jb6">76.54%</td>
<td class="tg-4jb6">4.99 MB</td>
</tr>
</tbody>
</table>
</html>