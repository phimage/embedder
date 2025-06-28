# Generic Text Embedder

A generic C++ application for generating text embeddings using ONNX models.

This is just used for testing. It's better to provide a service to do so (that does not load the model each time).

## Getting Models

You can download ONNX embedding models from Hugging Face using the `huggingface-cli` tool.

### Install Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
```

See the [official documentation](https://huggingface.co/docs/huggingface_hub/main/guides/cli) for more details.

### Download a Model

For example, to download the `nomic-embed-text-v1` model:

```bash
huggingface-cli download Xenova/nomic-embed-text-v1
```

This will download the model to your local cache. You can then set the environment variable to point to the cached model:

```bash
export EMBEDDING_MODEL_PATH=$HOME/.cache/huggingface/hub/models--Xenova--nomic-embed-text-v1/snapshots/0b85f78966a655763985a595b770f221374dda10
```

Note: The exact snapshot hash (the long string at the end) may vary depending on the model version.

## Building

Prerequisites:
- CMake 3.12+
- ONNX Runtime libraries
- C++17 compatible compiler

```bash
cmake .
make
```

## Usage

The embedder can be used in two ways:

### Method 1: Specify model path as argument (traditional)

```bash
./embedder <model_path> <input_text> [--verbose]
```

### Method 2: Use environment variable (new)

```bash
export EMBEDDING_MODEL_PATH=/path/to/model
./embedder <input_text> [--verbose]
```

### Arguments

- `model_path`: Path to directory containing the model and vocabulary files (optional if `EMBEDDING_MODEL_PATH` is set)
- `input_text`: Text to generate embedding for (wrap in quotes if it contains spaces)
- `--verbose`: Optional flag to enable verbose output (shows model info and embedding dimension)

### Examples

```bash
# Traditional usage with explicit model path
./embedder ./model_directory "Hello world"

# Using environment variable
export EMBEDDING_MODEL_PATH=./model_directory
./embedder "Hello world"

# With verbose output
export EMBEDDING_MODEL_PATH=./model_directory
./embedder "Hello world" --verbose

# Mixing approaches (environment variable as fallback)
export EMBEDDING_MODEL_PATH=./default_model
./embedder ./specific_model "Hello world"  # Uses ./specific_model
./embedder "Hello world"                   # Uses ./default_model
```

## Model Directory Structure

The embedder supports two directory structures:

### Option 1: Direct model placement

```
model_directory/
├── model.onnx
└── vocab.txt
```

### Option 2: ONNX subdirectory

```
model_directory/
├── onnx/
│   └── model.onnx
└── vocab.txt
```

## Output

Without `--verbose`: Outputs the full embedding as space-separated floating-point numbers.

With `--verbose`: Additionally shows:
- Model loading confirmation
- Input/output node information  
- Vocabulary size
- Embedding dimension

