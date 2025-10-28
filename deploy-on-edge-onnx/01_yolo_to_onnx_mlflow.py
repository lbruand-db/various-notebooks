# Databricks notebook source
# MAGIC %pip install ultralytics onnxruntime onnx
# MAGIC %pip install --upgrade typing_extensions
# MAGIC %restart_python

# COMMAND ----------

import onnx
import mlflow

# COMMAND ----------

# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

# COMMAND ----------

# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")


# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
path = model.export(format="onnx")
print(path)

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog lucasbruand_edp_forecast;
# MAGIC create schema if not exists pomobility;
# MAGIC use schema pomobility;

# COMMAND ----------

onnx_model2 = onnx.load_model(path)

# COMMAND ----------

# Define the image URL
image_url = "https://ultralytics.com/images/bus.jpg"

# Download the image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')
plt.show()

# COMMAND ----------

# Convert the image to an OpenCV format (BGR instead of RGB)
# OpenCV uses BGR by default, whereas most image libraries use RGB
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Load the YOLOv8 model
# Specify the model version to use ('yolov8n.pt' is the nano version, which is lightweight and fast)
model = YOLO(path) #YOLO("yolov8n.pt")  # You can choose different model versions (e.g., 'yolov8s.pt', 'yolov8m.pt')

# Run inference on the image
# This will detect objects in the image and return the results
results = model(image_cv)

# Iterate over the detection results
for result in results:
    boxes = result.boxes  # Get the bounding boxes for detected objects
    for box in boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates in (x1, y1, x2, y2) format
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
        
        cls = int(box.cls[0])  # Get the class ID of the detected object
        conf = float(box.conf[0])  # Get the confidence score of the detection
        
        class_name = model.names[cls]  # Get the class name from the class ID
        
        # Print the detected object's class, confidence, and bounding box coordinates
        print(f"Class: {class_name}, Confidence: {conf:.2f}, Bounding Box: ({x1}, {y1}) - ({x2}, {y2})")

        # Draw bounding boxes on the image
        # Use a blue rectangle (BGR: (255, 0, 0)) and thickness of 2 pixels
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Put the class name and confidence score above the bounding box
        cv2.putText(image_cv, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Convert the image back to RGB format for display
# This is necessary because matplotlib expects images in RGB format
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes
plt.figure(figsize=(10, 10))  # Set the figure size
plt.imshow(image_rgb)  # Show the image
plt.axis('off')  # Hide the axes
plt.title('Detected Objects')  # Set the title of the image
plt.show()  # Display the plot

# COMMAND ----------

from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# Define the input schema for file metadata and image data
input_schema = Schema([
    ColSpec("string", "path"),                     # File path
    ColSpec("datetime", "modificationTime"),      # Modification time
    ColSpec("long", "length"),                     # File size in bytes
    ColSpec("binary", "content")                   # Image content as binary data
])

# Define the output schema: output as a JSON string
output_schema = Schema([
    ColSpec("string", "detections")  # Serialized JSON string of the detections array
])

# Create a new signature with both input and output specifications
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
print(signature)

# COMMAND ----------


print(onnx_model2.__class__)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

mlflow.onnx.log_model(onnx_model2,
                      artifact_path="yolo_onnx",                      
                      registered_model_name="yolov8n",
                      signature=signature,
                      save_as_external_data=False)

# COMMAND ----------

# load model from UC
testing_onnx = mlflow.onnx.load_model(f"models:/yolov8n/5")
# save model locally
mlflow.onnx.save_model(testing_onnx, "yolov8n_model_local.onnx", save_as_external_data=False)

# COMMAND ----------

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml


class YOLOv8FromONNX:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(self.input_image)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image

    def main(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        # Create an inference session using the ONNX model and specify execution providers
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocess the image data
        img_data = self.preprocess()

        # Run inference using the preprocessed image data
        outputs = session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, outputs)  # output image

# COMMAND ----------

model_wrapper = YOLOv8FromONNX(onnx_model="yolov8n_model_local.onnx/model.onnx",
                input_image="bus.jpg",
                confidence_thres=0.5,
                iou_thres=0.4)
results_onnx = model_wrapper.main()

# COMMAND ----------

# Display the image with bounding boxes
plt.figure(figsize=(10, 10))  # Set the figure size
plt.imshow(results_onnx)  # Show the image
plt.axis('off')  # Hide the axes
plt.title('Detected Objects')  # Set the title of the image
plt.show()  # Display the plot

# COMMAND ----------


