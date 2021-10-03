# Homework 6

This homework covers some use of GStreamer and model optimization.  

## Part 1: GStreamer

1. In the lab, you used the Ndida sink nv3dsink; Nvidia provides a another sink, nveglglessink.  Convert the following sink to use nveglglessink.
```
gst-launch-1.0 v4l2src device=/dev/video0 ! xvimagesink
```
Ans:

```
gst-launch-1.0 v4l2src device=/dev/video0 ! 'video/x-raw'  ! nvvidconv ! nvegltransform  ! nveglglessink sync=false
```

2. What is the difference between a property and a capability?  How are they each expressed in a pipeline?

Ans:  Capabilities describe the types of media that may stream over a pad created from the template of an element. Properties are used to describe extra information for capabilities and consists of a key (a string) and a value. One example is give below - "audio/x-raw" is the capability and properties are given below as key value pairs.

```
Pad Templates:
  SRC template: 'src'
    Availability: Always
    Capabilities:
      audio/x-raw
                 format: F32LE
                   rate: [ 1, 2147483647 ]
               channels: [ 1, 256 ]


```


3. Explain the following pipeline, that is explain each piece of the pipeline, desribing if it is an element (if so, what type), property, or capability.  What does this pipeline do?

```
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw, framerate=30/1 ! videoconvert ! agingtv scratch-lines=10 ! videoconvert ! xvimagesink sync=false
```

The description of the pipeline is as follows

   a. 'v4l2src device=/dev/video0' : v4l2src is the video source element and its device property is set to video cam (device=/dev/video0) \
   b. 'video/x-raw, framerate=30/1': This sets the capabilities of the media output from v4l2src (it can output different media types) \
   c. 'videoconvert': videoconvert is filter converted video element converts from one colorspace to other. It is needed here as the next element agingtv only supports few media types \
   d. 'agingtv' : This element adds special effect - ages a video stream in realtime, changes the colors and adds scratches and dust. The scratch-lines property is set to 10 to add 10 scratch lines \
   
   e. 'xvimagesink' : XvImageSink renders video frames to a drawable on a local display. sync=false disables A/V sync.
   

4. GStreamer pipelines may also be used from Python and OpenCV. 

Client:
```
 gst-launch-1.0 -v v4l2src ! video/x-raw ! jpegenc ! rtpjpegpay ! udpsink host=127.0.0.1 port=5000

```
Server:

```
import numpy as np
import cv2

# use gstreamer for video directly; set the fps
server_cmd='udpsrc port=5000 ! application/x-rtp,media=video,payload=26,clock-rate=90000,encoding-name=JPEG,framerate=30/1 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink'
cap= cv2.VideoCapture(server_cmd)

#cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```



## Part 2: Model optimization and quantization


Like in the lab, you'll want to first baseline the your model, looking a the image of images per second it can process.  You may train the model using your Jetson device and the Jetson Inference scripts or train on a GPU eanabled server/virtual machine.  Once you have your baseline, follow the steps/examples outlined in the Jetson Inference to run your model with TensorRT (the defaults used are fine) and determine the number of images per second that are processed.

You may use either the container apporach or build the library from source.

For part 2, you'll need to submit:
- The base model you used
```
The base line model is resnet18()
```

- A description of your data set: The cat and dog dataset as below:

```
$ cd jetson-inference/python/training/classification/data
$ wget https://nvidia.box.com/shared/static/o577zd8yp3lmxf5zhm38svrbrv45am3y.gz -O cat_dog.tar.gz
$ tar xvzf cat_dog.tar.gz
```
- How long you trained your model, how many epochs you specified, and the batch size.
- 
```
Epoch: [34] completed, elapsed time 176.884 seconds
Total time :34*3 minutes
Batch Size ; 16
```
- Native Pytorch baseline
- 
```
Epoch: [34] completed, elapsed time 176.884 seconds
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Test: [  0/125]  Time  0.425 ( 0.425)  Loss 4.8451e-01 (4.8451e-01)  Acc@1  62.50 ( 62.50)  Acc@5 100.00 (100.00)
Test: [ 10/125]  Time  0.086 ( 0.118)  Loss 7.2746e-01 (6.3123e-01)  Acc@1  50.00 ( 60.23)  Acc@5 100.00 (100.00)
Test: [ 20/125]  Time  0.092 ( 0.109)  Loss 4.6943e-01 (6.3898e-01)  Acc@1  75.00 ( 60.71)  Acc@5 100.00 (100.00)
Test: [ 30/125]  Time  0.090 ( 0.102)  Loss 4.4111e-01 (6.2322e-01)  Acc@1  75.00 ( 62.50)  Acc@5 100.00 (100.00)
Test: [ 40/125]  Time  0.087 ( 0.099)  Loss 6.9490e-01 (5.9237e-01)  Acc@1  75.00 ( 65.85)  Acc@5 100.00 (100.00)
Test: [ 50/125]  Time  0.091 ( 0.101)  Loss 8.4453e-01 (5.9428e-01)  Acc@1  75.00 ( 66.91)  Acc@5 100.00 (100.00)
Test: [ 60/125]  Time  0.088 ( 0.100)  Loss 6.8690e-01 (5.9127e-01)  Acc@1  50.00 ( 66.80)  Acc@5 100.00 (100.00)
Test: [ 70/125]  Time  0.088 ( 0.098)  Loss 3.8540e-01 (5.5636e-01)  Acc@1  87.50 ( 69.37)  Acc@5 100.00 (100.00)
Test: [ 80/125]  Time  0.086 ( 0.097)  Loss 3.3433e-01 (5.2345e-01)  Acc@1 100.00 ( 72.07)  Acc@5 100.00 (100.00)
Test: [ 90/125]  Time  0.087 ( 0.096)  Loss 3.4846e-01 (5.0587e-01)  Acc@1  87.50 ( 73.76)  Acc@5 100.00 (100.00)
Test: [100/125]  Time  0.085 ( 0.095)  Loss 4.1540e-01 (4.8987e-01)  Acc@1  87.50 ( 75.25)  Acc@5 100.00 (100.00)
Test: [110/125]  Time  0.085 ( 0.094)  Loss 2.6516e-01 (4.7060e-01)  Acc@1 100.00 ( 76.80)  Acc@5 100.00 (100.00)
Test: [120/125]  Time  0.087 ( 0.094)  Loss 1.7805e-01 (4.6015e-01)  Acc@1 100.00 ( 77.79)  Acc@5 100.00 (100.00)
 * Acc@1 77.900 Acc@5 100.000
saved checkpoint to:  models/cat_dog/checkpoint.pth.tar

```

- TensorRT performance numbers
```
detected model format - ONNX  (extension '.onnx')
[TRT]    desired precision specified for GPU: FASTEST
[TRT]    requested fasted precision for device GPU without providing valid calibrator, disabling INT8
[TRT]    [MemUsageChange] Init CUDA: CPU +354, GPU +0, now: CPU 377, GPU 6745 (MiB)
[TRT]    native precisions detected for GPU:  FP32, FP16, INT8
[TRT]    selecting fastest native precision for GPU:  FP16
[TRT]    attempting to open engine cache file models/cat_dog/resnet18.onnx.1.1.8001.GPU.FP16.engine
[TRT]    loading network plan from engine cache... models/cat_dog/resnet18.onnx.1.1.8001.GPU.FP16.engine
[TRT]    device GPU, loaded models/cat_dog/resnet18.onnx


[image]  loaded 'data/cat_dog/test/dog/100.jpg'  (500x391, 3 channels)
class 0000 - 0.352090  (cat)
class 0001 - 0.647910  (dog)
Accuracy of TensorRT model is 0.7
```
