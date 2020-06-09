import cv2
import numpy as np
import argparse
from utils import process_frame, draw_prediction

# Define constants
# CONF_THRESHOLD is confidence threshold. Only detection with confidence greater than this will be retained
# NMS_THRESHOLD is used for non-max suppression
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to input image', required=True)
args = parser.parse_args()

# Read image from command line arguments
image = cv2.imread(args.image)
# Create blob from image
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Read COCO dataset classes
with open('coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the network with YOLOv3 weights and config using darknet framework
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg", "darknet")

# Get the output layer names used for forward pass
outNames = net.getUnconnectedOutLayersNames()

# Set the input
net.setInput(blob)

# Run forward pass
outs = net.forward(outNames)

# Process output and draw predictions
process_frame(image, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD)

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite("out.png", image)