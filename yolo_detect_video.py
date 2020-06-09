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
parser.add_argument('--video', help='Path to input video', required=True)
args = parser.parse_args()

# Read COCO dataset classes
with open('coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load the network with YOLOv3 weights and config using darknet framework
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg", "darknet")

# Get the output layer names used for forward pass
outNames = net.getUnconnectedOutLayersNames()

writer = None

cap = cv2.VideoCapture(args.video)

while(cap.isOpened()):

    ret, frame = cap.read()

    if not ret:
        break;

    # Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input
    net.setInput(blob)

    # Run forward pass
    outs = net.forward(outNames)

    # Process output and draw predictions
    process_frame(frame, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD)

    # Save video
    if writer is None:
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        writer = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frameWidth, frameHeight))

    writer.write(frame)
    

# cleaning up
cap.release()
writer.release()
cv2.destroyAllWindows()
