# Various notebooks


## deploy on the edge

an example of how to deploy an onnx on a tensorrt edge


## glued pca torch

What to do when you have a torch model trained on PCAed data but you still want to revert back to the original (non PCA) column name, for example when you want SHAP interpretation to the original columns


## IGN (WIP)

Downloading BDTOPO vectorial database from IGN that provides a map of all building in France


## image to LLM

How to run an image inside an LLM from a notebook.

```
.
├── deploy-on-edge-onnx
│   ├── 01_yolo_to_onnx_mlflow.py
│   ├── 02_edge_onnx_to_tensorrt.py
│   └── bus.jpg
├── glued-pca-torch-iris
│   ├── pytorch_pca_glued_together_iris.ipynb
│   └── pytorch_spark_pca_glued_together_iris.ipynb
├── ign
│   ├── 01_download_from_ign.ipynb
│   └── 02_extract_load_gpkg.ipynb
├── image-to-llm
│   ├── example-notebook-image-llama4.ipynb
└── README.md
```
