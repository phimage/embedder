#include "embedder.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <regex>

Embedder::Embedder() 
    : env_(ORT_LOGGING_LEVEL_WARNING, "embedder")
    , max_sequence_length_(512)
    , pad_token_id_(0)
    , cls_token_id_(101)
    , sep_token_id_(102)
    , unk_token_id_(100)
    , verbose_(false) {
    
    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

Embedder::~Embedder() = default;

void Embedder::setVerbose(bool verbose) {
    verbose_ = verbose;
}

bool Embedder::loadModel(const std::string& model_path) {
    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        
        // Get input and output names
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input names
        size_t num_input_nodes = session_->GetInputCount();
        input_names_.clear();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(std::string(input_name.get()));
        }
        
        // Output names
        size_t num_output_nodes = session_->GetOutputCount();
        output_names_.clear();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(std::string(output_name.get()));
        }
        
        if (verbose_) {
            std::cout << "Model loaded successfully!" << std::endl;
            std::cout << "Input nodes: " << num_input_nodes << std::endl;
            std::cout << "Output nodes: " << num_output_nodes << std::endl;
            
            // Print input and output names for debugging
            std::cout << "Input names: ";
            for (const auto& name : input_names_) {
                std::cout << "'" << name << "' ";
            }
            std::cout << std::endl;
            
            std::cout << "Output names: ";
            for (const auto& name : output_names_) {
                std::cout << "'" << name << "' ";
            }
            std::cout << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

bool Embedder::loadTokenizer(const std::string& vocab_path) {
    // Enhanced vocabulary loader for Nomic-style tokenizers
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "Error opening vocabulary file: " << vocab_path << std::endl;
        return false;
    }
    
    vocab_.clear();
    id_to_token_.clear();
    
    std::string line;
    int token_id = 0;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            // Handle different vocabulary formats
            size_t tab_pos = line.find('\t');
            if (tab_pos != std::string::npos) {
                // Format: token\tid or token\tfrequency
                std::string token = line.substr(0, tab_pos);
                vocab_[token] = token_id;
                id_to_token_[token_id] = token;
            } else {
                // Simple format: one token per line
                vocab_[line] = token_id;
                id_to_token_[token_id] = line;
            }
            token_id++;
        }
    }
    
    // Ensure special tokens exist
    if (vocab_.find("[PAD]") != vocab_.end()) pad_token_id_ = vocab_["[PAD]"];
    if (vocab_.find("[UNK]") != vocab_.end()) unk_token_id_ = vocab_["[UNK]"];
    if (vocab_.find("[CLS]") != vocab_.end()) cls_token_id_ = vocab_["[CLS]"];
    if (vocab_.find("[SEP]") != vocab_.end()) sep_token_id_ = vocab_["[SEP]"];
    
    if (verbose_) {
        std::cout << "Vocabulary loaded with " << vocab_.size() << " tokens" << std::endl;
        std::cout << "Special tokens - PAD: " << pad_token_id_ 
                  << ", UNK: " << unk_token_id_ 
                  << ", CLS: " << cls_token_id_ 
                  << ", SEP: " << sep_token_id_ << std::endl;
    }
    return true;
}

void Embedder::setSpecialTokens(int pad_id, int unk_id, int cls_id, int sep_id) {
    pad_token_id_ = pad_id;
    unk_token_id_ = unk_id;
    cls_token_id_ = cls_id;
    sep_token_id_ = sep_id;
    
    if (verbose_) {
        std::cout << "Updated special tokens - PAD: " << pad_token_id_ 
                  << ", UNK: " << unk_token_id_ 
                  << ", CLS: " << cls_token_id_ 
                  << ", SEP: " << sep_token_id_ << std::endl;
    }
}

std::string Embedder::preprocessText(const std::string& text) {
    std::string processed = text;
    
    // Convert to lowercase
    std::transform(processed.begin(), processed.end(), processed.begin(), ::tolower);
    
    // Add spaces around punctuation for better tokenization
    std::regex punct_regex(R"([.,!?;:()\[\]{}\"'])");
    processed = std::regex_replace(processed, punct_regex, " $& ");
    
    // Normalize whitespace
    std::regex whitespace_regex(R"(\s+)");
    processed = std::regex_replace(processed, whitespace_regex, " ");
    
    // Trim
    processed.erase(0, processed.find_first_not_of(" \t\n\r"));
    processed.erase(processed.find_last_not_of(" \t\n\r") + 1);
    
    return processed;
}

std::vector<std::string> Embedder::basicTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string processed = preprocessText(text);
    
    std::istringstream iss(processed);
    std::string word;
    
    while (iss >> word) {
        if (!word.empty()) {
            tokens.push_back(word);
        }
    }
    
    return tokens;
}

std::vector<std::string> Embedder::wordpieceTokenize(const std::string& word) {
    std::vector<std::string> subwords;
    
    if (vocab_.find(word) != vocab_.end()) {
        // Word exists in vocabulary as-is
        subwords.push_back(word);
        return subwords;
    }
    
    // Try to break down the word into subwords
    std::string remaining = word;
    
    while (!remaining.empty()) {
        std::string longest_match;
        size_t longest_len = 0;
        
        // Find the longest matching prefix in vocabulary
        for (size_t len = std::min(remaining.length(), size_t(20)); len > 0; --len) {
            std::string candidate = remaining.substr(0, len);
            
            // For subword pieces (except the first), add "##" prefix
            if (!subwords.empty()) {
                candidate = "##" + candidate;
            }
            
            if (vocab_.find(candidate) != vocab_.end()) {
                longest_match = candidate;
                longest_len = len;
                break;
            }
        }
        
        if (longest_match.empty()) {
            // No match found, use unknown token
            subwords.push_back("[UNK]");
            break;
        }
        
        subwords.push_back(longest_match);
        remaining = remaining.substr(longest_len);
    }
    
    return subwords;
}

int Embedder::getTokenId(const std::string& token) {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
        return it->second;
    }
    return unk_token_id_;
}

std::vector<int> Embedder::tokenize(const std::string& text) {
    std::vector<int> token_ids;
    
    // Add CLS token at the beginning
    token_ids.push_back(cls_token_id_);
    
    // Basic tokenization (word-level)
    std::vector<std::string> words = basicTokenize(text);
    
    // Apply wordpiece tokenization to each word
    for (const std::string& word : words) {
        std::vector<std::string> subwords = wordpieceTokenize(word);
        
        for (const std::string& subword : subwords) {
            int token_id = getTokenId(subword);
            token_ids.push_back(token_id);
        }
    }
    
    // Add SEP token at the end
    token_ids.push_back(sep_token_id_);
    
    if (verbose_ && !words.empty()) {
        std::cout << "Tokenized '" << text << "' into " << token_ids.size() << " tokens" << std::endl;
        std::cout << "First few tokens: ";
        for (size_t i = 0; i < std::min(size_t(10), token_ids.size()); ++i) {
            std::cout << token_ids[i] << " ";
        }
        std::cout << std::endl;
    }
    
    return token_ids;
}

std::vector<int> Embedder::encodeText(const std::string& text) {
    std::vector<int> tokens = tokenize(text);
    
    // Truncate or pad to max_sequence_length
    if (tokens.size() > max_sequence_length_) {
        tokens.resize(max_sequence_length_);
        tokens[max_sequence_length_ - 1] = sep_token_id_; // Ensure SEP token at end
    } else {
        // Pad with pad tokens
        tokens.resize(max_sequence_length_, pad_token_id_);
    }
    
    return tokens;
}

void Embedder::normalizeEmbedding(std::vector<float>& embedding) {
    // L2 normalization
    float norm = 0.0f;
    for (float value : embedding) {
        norm += value * value;
    }
    norm = std::sqrt(norm);
    
    if (norm > 0.0f) {
        for (float& value : embedding) {
            value /= norm;
        }
    }
}

std::vector<float> Embedder::getEmbedding(const std::string& text) {
    if (!session_) {
        std::cerr << "Model not loaded!" << std::endl;
        return {};
    }
    
    try {
        // Encode text to token IDs
        std::vector<int> input_ids = encodeText(text);
        
        // Create attention mask (1 for real tokens, 0 for padding)
        std::vector<int> attention_mask(max_sequence_length_);
        for (size_t i = 0; i < input_ids.size(); ++i) {
            attention_mask[i] = (input_ids[i] != pad_token_id_) ? 1 : 0;
        }
        
        // Create token type IDs (all 0s for single sentence)
        std::vector<int> token_type_ids(max_sequence_length_, 0);
        
        // Create input tensors
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(max_sequence_length_)};
        
        // Convert int to int64_t for ONNX
        std::vector<int64_t> input_ids_int64(input_ids.begin(), input_ids.end());
        std::vector<int64_t> attention_mask_int64(attention_mask.begin(), attention_mask.end());
        std::vector<int64_t> token_type_ids_int64(token_type_ids.begin(), token_type_ids.end());
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids_int64.data(), input_ids_int64.size(),
            input_shape.data(), input_shape.size()));
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, token_type_ids_int64.data(), token_type_ids_int64.size(),
            input_shape.data(), input_shape.size()));
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask_int64.data(), attention_mask_int64.size(),
            input_shape.data(), input_shape.size()));
        
        // Run inference
        std::vector<const char*> input_names_cstr;
        std::vector<const char*> output_names_cstr;
        
        for (const auto& name : input_names_) {
            input_names_cstr.push_back(name.c_str());
        }
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }
        
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                          input_names_cstr.data(), input_tensors.data(), input_tensors.size(),
                                          output_names_cstr.data(), output_names_cstr.size());
        
        // Extract embeddings from output tensor
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Typically, we want the [CLS] token embedding (first token) or mean pooling
        size_t embedding_dim = output_shape[2]; // [batch_size, seq_len, embedding_dim]
        std::vector<float> embedding(output_data, output_data + embedding_dim);
        
        // Normalize the embedding
        normalizeEmbedding(embedding);
        
        return embedding;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return {};
    }
}

std::string Embedder::detokenize(const std::vector<int>& token_ids) {
    std::string result;
    
    for (size_t i = 0; i < token_ids.size(); ++i) {
        int token_id = token_ids[i];
        
        // Skip special tokens
        if (token_id == pad_token_id_ || token_id == cls_token_id_ || token_id == sep_token_id_) {
            continue;
        }
        
        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) {
            std::string token = it->second;
            
            // Handle subword tokens (remove ## prefix)
            if (token.substr(0, 2) == "##") {
                result += token.substr(2);
            } else {
                if (!result.empty()) {
                    result += " ";
                }
                result += token;
            }
        }
    }
    
    return result;
}