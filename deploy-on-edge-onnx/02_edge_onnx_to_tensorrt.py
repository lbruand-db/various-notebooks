# Databricks notebook source
# MAGIC %pip install tensorrt onnxruntime onnx

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog lucasbruand_edp_forecast;
# MAGIC create schema if not exists pomobility;
# MAGIC use schema pomobility;

# COMMAND ----------

import mlflow
import mlflow.onnx
from pathlib import Path

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
# load model from UC

# save model locally
model_local_name = "yolov8n_model_local.onnx"
if not Path(model_local_name).exists():
  testing_onnx = mlflow.onnx.load_model(f"models:/yolov8n/5")
  mlflow.onnx.save_model(testing_onnx, model_local_name, save_as_external_data=False)

# COMMAND ----------

import os

import random
import sys

import numpy as np

import tensorrt as trt
from PIL import Image



# COMMAND ----------

# this is from : https://github.com/NVIDIA/TensorRT/blob/main/samples/python/introductory_parser_samples/onnx_resnet50.py



# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file, path=None, output_engine="output.engine"):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<30)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model_fp:
        if not parser.parse(model=model_fp.read(), path=path):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    engine_bytes = builder.build_serialized_network(network, config)
    with open(output_engine, 'wb') as f:
        f.write(engine_bytes)

  

build_engine_onnx( model_file=(Path(model_local_name) / "model.onnx"), path=model_local_name )



# COMMAND ----------

dir(engine_bytes)

# COMMAND ----------


