# Object Detection and Recognition using YOLO

## Prerequisites

<ul>
<li>Python 3</li>
<li>OpenCV 4</li>
<li>Numpy</li>
<li>YOLOv3 pre-trained models:</li>
  <ul>
  <li>YOLOv3 config: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg</li>
  <li>YOLOv3 weights: https://pjreddie.com/media/files/yolov3.weights</li>
  <li>Class names: https://github.com/pjreddie/darknet/blob/master/data/coco.names</li>
  </ul>
</ul>

## Usage

To detect object in image, just run:

```Python
python yolo_detect_image.py --image name_of_your_image_here
```

For example, with this input image:

<img src="https://github.com/minhthangdang/minhthangdang.github.io/blob/master/YOLO-example.png?raw=true" alt="YOLO input image" title="YOLO input image">
<br><br>

The output will be
<img src="https://github.com/minhthangdang/minhthangdang.github.io/blob/master/YOLO-output.png?raw=true" alt="YOLO input image" title="YOLO input image">
<br><br>

Similarly, to detect object in video, just run:

```python
python yolo_detect_video.py --video name_of_your_video_here
```

An example can be seen below:

<iframe width="560" height="315" src="https://www.youtube.com/embed/5Zt7ohK2Rjk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Full tutorial is available at http://dangminhthang.com/computer-vision/object-detection-and-recognition-using-yolo/
