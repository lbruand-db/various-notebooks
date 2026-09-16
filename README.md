# Various notebooks

A grab-bag of self-contained Databricks / Python notebooks and experiments. Each directory
is an independent mini-project.

| Project | What it does |
|---------|--------------|
| [`audio-transcription`](audio-transcription/) | Transcribe audio with OpenAI Whisper. |
| [`deploy-on-edge-onnx`](deploy-on-edge-onnx/) | Export a YOLO model to ONNX (logged in MLflow) and convert it to TensorRT for edge deployment. |
| [`embeddings`](embeddings/) | Generate text embeddings on Databricks: `ai_query`, loading GTE / E5 models in-process, sentence-transformers, and a Spark UDF. |
| [`gen-pdf-typst`](gen-pdf-typst/) | Generate PDFs from a notebook using [Typst](https://typst.app/). |
| [`glued-pca-torch-iris`](glued-pca-torch-iris/) | Keep the original (pre-PCA) column names for a torch model trained on PCA'd data — e.g. so SHAP explains the original features. |
| [`gte-romance-lang`](gte-romance-lang/) | Benchmark GTE / BGE embeddings on STS across English, French, Italian and Spanish — FMAPI English models vs. a multilingual model run on CPU. |
| [`ign`](ign/) _(WIP)_ | Download the IGN **BDTOPO** vector database (a map of every building in France) and load the GeoPackage. |
| [`image-to-llm`](image-to-llm/) | Pass an image into an LLM (Llama 4) from a notebook. |
| [`load_testing_tpm_ai_gateway_v2`](load_testing_tpm_ai_gateway_v2/) | Load-test the AI Gateway v2 tokens-per-minute limits. |
| [`openapi`](openapi/) | Working with an OpenAPI specification from a notebook. |
| [`orthoimagery_resnet50`](orthoimagery_resnet50/) | ResNet-50 on orthoimagery, including handling different image resolutions. |
| [`sparkxgbclassifier2pmml`](sparkxgbclassifier2pmml/) | Convert a Spark MLlib XGBoost classifier to PMML. |
| [`workaround_vector_search_dbsql`](workaround_vector_search_dbsql/) | Workarounds for Databricks Vector Search from DBSQL: REST API, multi-index, and storage-optimized endpoints. |
