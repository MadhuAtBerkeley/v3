import jetson.inference
import jetson.utils

import argparse
import sys

labels = ['cat', 'dog']


# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage() + jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())


parser.add_argument("input_URI", type=str, default="input_0", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="output_0", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument('--headless', action='store_true', default=(), help="run without display")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
        opt = parser.parse_known_args()[0]
except:
        print("")
        parser.print_help()
        sys.exit(0)


# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)
font = jetson.utils.cudaFont()
input_URI = opt.input_URI
total_labels = 0
correct_labels = 0

for label in labels:
   
    opt.input_URI = input_URI+'/'+label
# create video sources & outputs
    input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

# process frames until the user exits
    while True:
        # capture the next image
        img = input.Capture()

        # classify the image
        class_id, confidence = net.Classify(img)
        #print(class_id, confidence)
        total_labels += 1
        if label == labels[class_id]:
            correct_labels += 1
        # find the object description

        # print out performance info
        # exit on input/output EOS
        if not input.IsStreaming(): # or not output.IsStreaming():
                break



accuracy = correct_labels*1.0/total_labels
print("Accuracy of TensorRT model is {}".format(accuracy))
