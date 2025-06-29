#include "embedder.hpp"
#include <iostream>
#include <vector>
#include <filesystem>
#include <string>
#include <cstdlib>
#include <algorithm>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [model_path] <input_text_or_mode> [--verbose] [--batch] [--delimiter=DELIM]" << std::endl;
    std::cout << "  model_path: Path to directory containing model.onnx (or onnx/model.onnx) and vocab.txt" << std::endl;
    std::cout << "              If not provided, uses EMBEDDING_MODEL_PATH environment variable" << std::endl;
    std::cout << "  input_text: Text to generate embedding for (single mode)" << std::endl;
    std::cout << "  --batch:    Enable batch processing mode (reads texts from stdin)" << std::endl;
    std::cout << "  --delimiter=DELIM: Set custom delimiter for batch mode (default: \\0 null byte)" << std::endl;
    std::cout << "  --verbose:  Enable verbose output (optional)" << std::endl;
    std::cout << std::endl;
    std::cout << "Batch Mode Input Formats:" << std::endl;
    std::cout << "  Default (null-delimited): printf \"Text1\\0Text2\\0Text3\\0\" | " << program_name << " --batch" << std::endl;
    std::cout << "  Custom delimiter:         echo \"Text1|||Text2|||Text3\" | " << program_name << " --batch --delimiter=\"|||\"" << std::endl;
    std::cout << "  Line-based (unsafe):      echo -e \"Text1\\nText2\\nText3\" | " << program_name << " --batch --delimiter=\"\\n\"" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  Single text: " << program_name << " \"Hello world\"" << std::endl;
    std::cout << "  Batch safe:  printf \"Hello world\\0Text with\\nnewlines\\0\" | " << program_name << " --batch" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string model_dir;
    std::string input_text;
    bool verbose = false;
    bool batch_mode = false;
    std::string delimiter = "\0"; // Default to null delimiter
    
    // Parse arguments - handle flags and delimiter
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--batch") {
            batch_mode = true;
        } else if (arg.substr(0, 12) == "--delimiter=") {
            delimiter = arg.substr(12);
            // Handle common escaped characters
            if (delimiter == "\\n") delimiter = "\n";
            else if (delimiter == "\\t") delimiter = "\t";
            else if (delimiter == "\\r") delimiter = "\r";
            else if (delimiter == "\\0") delimiter = "\0";
        }
    }
    
    // Parse arguments - flexible to handle both old and new usage
    if (argc < 2 && !batch_mode) {
        printUsage(argv[0]);
        return -1;
    }
    
    // Count non-flag arguments
    int non_flag_args = 0;
    std::vector<std::string> non_flag_argv;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg != "--verbose" && arg != "--batch" && arg.substr(0, 12) != "--delimiter=") {
            non_flag_argv.push_back(arg);
            non_flag_args++;
        }
    }
    
    if (batch_mode) {
        // Batch mode: program [model_path] --batch [--verbose]
        if (non_flag_args >= 1) {
            model_dir = non_flag_argv[0];
        } else {
            const char* env_model_path = std::getenv("EMBEDDING_MODEL_PATH");
            if (!env_model_path) {
                std::cerr << "Error: No model path provided and EMBEDDING_MODEL_PATH environment variable not set." << std::endl;
                printUsage(argv[0]);
                return -1;
            }
            model_dir = env_model_path;
        }
    } else if (non_flag_args >= 2) {
        // Traditional usage: program model_path input_text [--verbose]
        model_dir = non_flag_argv[0];
        input_text = non_flag_argv[1];
    } else if (non_flag_args == 1) {
        // New usage: program input_text [--verbose] (use env variable for model path)
        const char* env_model_path = std::getenv("EMBEDDING_MODEL_PATH");
        if (!env_model_path) {
            std::cerr << "Error: No model path provided and EMBEDDING_MODEL_PATH environment variable not set." << std::endl;
            printUsage(argv[0]);
            return -1;
        }
        
        model_dir = env_model_path;
        input_text = non_flag_argv[0];
    } else {
        printUsage(argv[0]);
        return -1;
    }
    
    Embedder embedder;
    embedder.setVerbose(verbose);
    
    // Load model from folder (this will automatically load config files)
    if (!embedder.loadModelFolder(model_dir)) {
        std::cerr << "Failed to load model from folder " << model_dir << std::endl;
        return -1;
    }
    
    // Generate embedding(s)
    if (batch_mode) {
        // Batch processing mode - read texts from stdin using delimiter
        std::vector<std::string> texts;
        
        if (verbose) {
            if (delimiter == "\0") {
                std::cout << "Reading null-delimited texts from stdin..." << std::endl;
            } else if (delimiter == "\n") {
                std::cout << "Reading line-delimited texts from stdin..." << std::endl;
            } else {
                std::cout << "Reading texts from stdin with delimiter: '" << delimiter << "'..." << std::endl;
            }
        }
        
        // Read all input data
        std::string input_data;
        std::string chunk;
        while (std::getline(std::cin, chunk)) {
            input_data += chunk + "\n";
        }
        
        // Remove the last newline if it exists
        if (!input_data.empty() && input_data.back() == '\n') {
            input_data.pop_back();
        }
        
        // Split by delimiter
        if (delimiter == "\0") {
            // Special handling for null delimiter - read until null bytes
            size_t start = 0;
            size_t pos = 0;
            while (pos < input_data.length()) {
                if (input_data[pos] == '\0') {
                    if (pos > start) {
                        texts.push_back(input_data.substr(start, pos - start));
                    }
                    start = pos + 1;
                }
                pos++;
            }
            // Add the last text if no trailing null
            if (start < input_data.length()) {
                texts.push_back(input_data.substr(start));
            }
        } else {
            // Split by custom delimiter
            size_t start = 0;
            size_t pos = input_data.find(delimiter);
            
            while (pos != std::string::npos) {
                if (pos > start) {
                    texts.push_back(input_data.substr(start, pos - start));
                }
                start = pos + delimiter.length();
                pos = input_data.find(delimiter, start);
            }
            
            // Add the last text
            if (start < input_data.length()) {
                texts.push_back(input_data.substr(start));
            }
        }
        
        // Remove empty texts
        texts.erase(std::remove_if(texts.begin(), texts.end(), 
                                   [](const std::string& s) { return s.empty(); }), 
                    texts.end());
        
        if (texts.empty()) {
            std::cerr << "No texts provided for batch processing" << std::endl;
            if (verbose) {
                std::cerr << "Input data length: " << input_data.length() << " bytes" << std::endl;
                std::cerr << "Delimiter: ";
                if (delimiter == "\0") std::cerr << "\\0 (null byte)";
                else if (delimiter == "\n") std::cerr << "\\n (newline)";
                else std::cerr << "'" << delimiter << "'";
                std::cerr << std::endl;
            }
            return -1;
        }
        
        if (verbose) {
            std::cout << "Processing " << texts.size() << " texts in batch..." << std::endl;
            std::cout << "First text preview: '" << (texts[0].length() > 50 ? 
                         texts[0].substr(0, 50) + "..." : texts[0]) << "'" << std::endl;
        }
        
        std::vector<std::vector<float>> batch_embeddings = embedder.getBatchEmbeddings(texts);
        
        if (!batch_embeddings.empty()) {
            for (size_t i = 0; i < batch_embeddings.size(); ++i) {
                if (verbose) {
                    std::cout << "Embedding " << (i + 1) << ": ";
                }
                
                const auto& embedding = batch_embeddings[i];
                for (size_t j = 0; j < embedding.size(); ++j) {
                    std::cout << embedding[j];
                    if (j < embedding.size() - 1) {
                        std::cout << " ";
                    }
                }
                std::cout << std::endl;
            }
            
            if (verbose) {
                std::cout << "Processed " << batch_embeddings.size() << " embeddings successfully" << std::endl;
                std::cout << "Embedding dimension: " << batch_embeddings[0].size() << std::endl;
            }
        } else {
            std::cerr << "Failed to generate batch embeddings" << std::endl;
            return -1;
        }
    } else {
        // Single text processing mode
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
    }
    
    return 0;
}