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
- A description of your data set
- How long you trained your model, how many epochs you specified, and the batch size.
- Native Pytorch baseline
- TensorRT performance numbers

