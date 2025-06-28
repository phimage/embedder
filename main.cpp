#include "embedder.hpp"
#include <iostream>
#include <vector>
#include <filesystem>
#include <string>
#include <cstdlib>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [model_path] <input_text> [--verbose]" << std::endl;
    std::cout << "  model_path: Path to directory containing model.onnx (or onnx/model.onnx) and vocab.txt" << std::endl;
    std::cout << "              If not provided, uses EMBEDDING_MODEL_PATH environment variable" << std::endl;
    std::cout << "  input_text: Text to generate embedding for" << std::endl;
    std::cout << "  --verbose:  Enable verbose output (optional)" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string model_dir;
    std::string input_text;
    bool verbose = false;
    
    // First, check for --verbose flag anywhere in arguments
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verbose") {
            verbose = true;
            break;
        }
    }
    
    // Parse arguments - flexible to handle both old and new usage
    if (argc < 2) {
        printUsage(argv[0]);
        return -1;
    }
    
    // Count non-verbose arguments
    int non_verbose_args = 0;
    std::vector<std::string> non_verbose_argv;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) != "--verbose") {
            non_verbose_argv.push_back(argv[i]);
            non_verbose_args++;
        }
    }
    
    if (non_verbose_args >= 2) {
        // Traditional usage: program model_path input_text [--verbose]
        model_dir = non_verbose_argv[0];
        input_text = non_verbose_argv[1];
    } else if (non_verbose_args == 1) {
        // New usage: program input_text [--verbose] (use env variable for model path)
        const char* env_model_path = std::getenv("EMBEDDING_MODEL_PATH");
        if (!env_model_path) {
            std::cerr << "Error: No model path provided and EMBEDDING_MODEL_PATH environment variable not set." << std::endl;
            printUsage(argv[0]);
            return -1;
        }
        
        model_dir = env_model_path;
        input_text = non_verbose_argv[0];
    } else {
        printUsage(argv[0]);
        return -1;
    }
    
    Embedder embedder;
    embedder.setVerbose(verbose);
    
    // Try to find model file (check onnx/model.onnx first, then model.onnx)
    std::string model_path;
    std::string onnx_subdir_path = model_dir + "/onnx/model.onnx";
    std::string direct_path = model_dir + "/model.onnx";
    
    if (std::filesystem::exists(onnx_subdir_path)) {
        model_path = onnx_subdir_path;
    } else if (std::filesystem::exists(direct_path)) {
        model_path = direct_path;
    } else {
        std::cerr << "Model file not found. Looked for:" << std::endl;
        std::cerr << "  " << onnx_subdir_path << std::endl;
        std::cerr << "  " << direct_path << std::endl;
        return -1;
    }
    
    // Load the ONNX model
    if (!embedder.loadModel(model_path)) {
        std::cerr << "Failed to load model from " << model_path << std::endl;
        return -1;
    }
    
    // Load vocabulary
    std::string vocab_path = model_dir + "/vocab.txt";
    if (!embedder.loadTokenizer(vocab_path)) {
        std::cerr << "Failed to load tokenizer from " << vocab_path << std::endl;
        return -1;
    }
    
    // Generate embedding
    std::vector<float> embedding = embedder.getEmbedding(input_text);
    
    if (!embedding.empty()) {
        // Print full embedding
        for (size_t i = 0; i < embedding.size(); ++i) {
            std::cout << embedding[i];
            if (i < embedding.size() - 1) {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
        
        if (verbose) {
            std::cout << "Embedding dimension: " << embedding.size() << std::endl;
        }
    } else {
        std::cerr << "Failed to generate embedding" << std::endl;
        return -1;
    }
    
    return 0;
}