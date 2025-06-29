#include "embedder.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <regex>
#include <thread>
#include <filesystem>

Embedder::Embedder() 
    : env_(ORT_LOGGING_LEVEL_WARNING, "embedder")
    , max_sequence_length_(512)
    , pad_token_id_(0)
    , cls_token_id_(101)
    , sep_token_id_(102)
    , unk_token_id_(100)
    , mask_token_id_(103)
    , verbose_(false)
    , tokenizer_type_("bert")
    , do_lower_case_(true)
    , do_basic_tokenize_(true)
    , unk_token_("[UNK]")
    , pad_token_("[PAD]")
    , cls_token_("[CLS]")
    , sep_token_("[SEP]")
    , mask_token_("[MASK]") {
    
    // Configure for optimized CPU performance on Apple Silicon
    // Use all available performance cores
    session_options_.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    session_options_.SetInterOpNumThreads(1); // Avoid thread contention
    
    // Enable all CPU optimizations
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    
    // Apple Silicon specific optimizations
    session_options_.AddConfigEntry("session.disable_cpu_ep_fallback", "0");
    session_options_.AddConfigEntry("session.use_env_allocators", "1");
    
    if (verbose_) {
        std::cout << "Optimized CPU execution configured for Apple Silicon" << std::endl;
        std::cout << "Using " << std::thread::hardware_concurrency() << " threads" << std::endl;
    }
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

void Embedder::setMaxSequenceLength(int max_length) {
    max_sequence_length_ = max_length;
    if (verbose_) {
        std::cout << "Updated max sequence length to: " << max_sequence_length_ << std::endl;
    }
}

std::string Embedder::preprocessText(const std::string& text) {
    std::string processed = text;
    
    // Convert to lowercase only if configured to do so
    if (do_lower_case_) {
        std::transform(processed.begin(), processed.end(), processed.begin(), ::tolower);
    }
    
    // Handle Chinese characters - add spaces around them for proper tokenization
    std::regex chinese_regex(u8"[\u4e00-\u9fff]");
    processed = std::regex_replace(processed, chinese_regex, " $& ");
    
    // More conservative punctuation handling - only add spaces for major punctuation
    std::regex punct_regex(R"([.!?;])");
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
    
    if (word.empty()) {
        return subwords;
    }
    
    // If the whole word is in vocabulary, return it
    if (vocab_.find(word) != vocab_.end()) {
        subwords.push_back(word);
        return subwords;
    }
    
    // WordPiece algorithm: try to break word into subwords
    std::string remaining = word;
    bool is_bad = false;
    
    while (!remaining.empty()) {
        std::string longest_match;
        size_t longest_len = 0;
        
        // Try all possible substrings starting from the longest
        for (size_t len = remaining.length(); len > 0; --len) {
            std::string candidate = remaining.substr(0, len);
            
            // For continuation pieces, add "##" prefix
            if (subwords.size() > 0) {
                candidate = "##" + candidate;
            }
            
            // Check if this candidate exists in vocabulary
            if (vocab_.find(candidate) != vocab_.end()) {
                longest_match = candidate;
                longest_len = len;
                break;
            }
        }
        
        if (longest_match.empty()) {
            // No valid subword found, mark as unknown
            is_bad = true;
            break;
        }
        
        subwords.push_back(longest_match);
        remaining = remaining.substr(longest_len);
    }
    
    if (is_bad) {
        // If we couldn't tokenize the word, return [UNK]
        return {"[UNK]"};
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
        // This must be done AFTER padding in encodeText
        std::vector<int> attention_mask(max_sequence_length_);
        for (size_t i = 0; i < max_sequence_length_; ++i) {
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

std::vector<std::vector<int>> Embedder::batchTokenize(const std::vector<std::string>& texts) {
    std::vector<std::vector<int>> batch_tokens;
    batch_tokens.reserve(texts.size());
    
    for (const std::string& text : texts) {
        batch_tokens.push_back(tokenize(text));
    }
    
    return batch_tokens;
}

std::vector<std::vector<int>> Embedder::batchEncodeText(const std::vector<std::string>& texts) {
    std::vector<std::vector<int>> batch_encoded;
    batch_encoded.reserve(texts.size());
    
    for (const std::string& text : texts) {
        batch_encoded.push_back(encodeText(text));
    }
    
    return batch_encoded;
}

void Embedder::normalizeBatchEmbeddings(std::vector<std::vector<float>>& embeddings) {
    for (auto& embedding : embeddings) {
        normalizeEmbedding(embedding);
    }
}

std::vector<std::vector<float>> Embedder::getBatchEmbeddings(const std::vector<std::string>& texts) {
    if (!session_) {
        std::cerr << "Model not loaded!" << std::endl;
        return {};
    }
    
    if (texts.empty()) {
        return {};
    }
    
    try {
        size_t batch_size = texts.size();
        
        // Encode all texts to token IDs
        std::vector<std::vector<int>> batch_input_ids = batchEncodeText(texts);
        
        // Create attention masks and token type IDs for the entire batch
        std::vector<std::vector<int>> batch_attention_masks;
        std::vector<std::vector<int>> batch_token_type_ids;
        
        batch_attention_masks.reserve(batch_size);
        batch_token_type_ids.reserve(batch_size);
        
        for (const auto& input_ids : batch_input_ids) {
            // Create attention mask (1 for real tokens, 0 for padding)
            std::vector<int> attention_mask(max_sequence_length_);
            for (size_t i = 0; i < max_sequence_length_; ++i) {
                attention_mask[i] = (input_ids[i] != pad_token_id_) ? 1 : 0;
            }
            batch_attention_masks.push_back(attention_mask);
            
            // Create token type IDs (all 0s for single sentence)
            std::vector<int> token_type_ids(max_sequence_length_, 0);
            batch_token_type_ids.push_back(token_type_ids);
        }
        
        // Flatten the batch data for ONNX tensor creation
        std::vector<int64_t> flat_input_ids;
        std::vector<int64_t> flat_attention_mask;
        std::vector<int64_t> flat_token_type_ids;
        
        flat_input_ids.reserve(batch_size * max_sequence_length_);
        flat_attention_mask.reserve(batch_size * max_sequence_length_);
        flat_token_type_ids.reserve(batch_size * max_sequence_length_);
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < max_sequence_length_; ++j) {
                flat_input_ids.push_back(static_cast<int64_t>(batch_input_ids[i][j]));
                flat_attention_mask.push_back(static_cast<int64_t>(batch_attention_masks[i][j]));
                flat_token_type_ids.push_back(static_cast<int64_t>(batch_token_type_ids[i][j]));
            }
        }
        
        // Create input tensors with batch dimension
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(max_sequence_length_)};
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, flat_input_ids.data(), flat_input_ids.size(),
            input_shape.data(), input_shape.size()));
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, flat_token_type_ids.data(), flat_token_type_ids.size(),
            input_shape.data(), input_shape.size()));
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, flat_attention_mask.data(), flat_attention_mask.size(),
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
        
        // Parse the batch output
        size_t embedding_dim = output_shape[2]; // [batch_size, seq_len, embedding_dim]
        size_t seq_len = output_shape[1];
        
        std::vector<std::vector<float>> batch_embeddings;
        batch_embeddings.reserve(batch_size);
        
        for (size_t i = 0; i < batch_size; ++i) {
            // Extract [CLS] token embedding (first token) for each sample in the batch
            size_t offset = i * seq_len * embedding_dim;
            std::vector<float> embedding(output_data + offset, output_data + offset + embedding_dim);
            batch_embeddings.push_back(embedding);
        }
        
        // Normalize all embeddings
        normalizeBatchEmbeddings(batch_embeddings);
        
        if (verbose_) {
            std::cout << "Processed batch of " << batch_size << " texts" << std::endl;
            std::cout << "Output shape: [" << output_shape[0] << ", " << output_shape[1] << ", " << output_shape[2] << "]" << std::endl;
            std::cout << "Embedding dimension: " << embedding_dim << std::endl;
        }
        
        return batch_embeddings;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during batch inference: " << e.what() << std::endl;
        return {};
    }
}

bool Embedder::loadModelFolder(const std::string& model_folder_path) {
    if (verbose_) {
        std::cout << "Loading model from folder: " << model_folder_path << std::endl;
    }
    
    // Load configuration files first (even if model loading fails, this shows what we'd load)
    std::string config_path = model_folder_path + "/config.json";
    if (std::filesystem::exists(config_path)) {
        loadConfig(config_path);
    }
    
    std::string tokenizer_config_path = model_folder_path + "/tokenizer_config.json";
    if (std::filesystem::exists(tokenizer_config_path)) {
        loadTokenizerConfig(tokenizer_config_path);
    }
    
    std::string special_tokens_path = model_folder_path + "/special_tokens_map.json";
    if (std::filesystem::exists(special_tokens_path)) {
        loadSpecialTokensMap(special_tokens_path);
    }
    
    // Load vocabulary
    std::string vocab_path = model_folder_path + "/vocab.txt";
    if (std::filesystem::exists(vocab_path)) {
        if (!loadTokenizer(vocab_path)) {
            std::cerr << "Failed to load vocabulary from " << vocab_path << std::endl;
            return false;
        }
    } else {
        // Try tokenizer.json format
        std::string tokenizer_json_path = model_folder_path + "/tokenizer.json";
        if (std::filesystem::exists(tokenizer_json_path)) {
            // TODO: Implement tokenizer.json loading
            std::cout << "Warning: tokenizer.json format not yet implemented" << std::endl;
        }
    }
    
    // Load BPE merges if they exist
    std::string merges_path = model_folder_path + "/merges.txt";
    if (std::filesystem::exists(merges_path)) {
        loadBPEMerges(merges_path);
    }
    
    // Update special token IDs based on loaded configuration
    initializeDefaultTokens();
    
    // Find the ONNX model file
    std::string model_file = findModelFile(model_folder_path, {".onnx", ".ort"});
    if (model_file.empty()) {
        std::cerr << "No ONNX model file found in " << model_folder_path << std::endl;
        return false;
    }
    
    // Load the ONNX model
    if (!loadModel(model_file)) {
        return false;
    }
    
    return true;
}

bool Embedder::loadConfig(const std::string& config_path) {
    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "Error opening config file: " << config_path << std::endl;
            return false;
        }
        
        nlohmann::json config;
        file >> config;
        
        // Handle different ways models specify max sequence length
        if (config.contains("max_position_embeddings")) {
            max_sequence_length_ = config["max_position_embeddings"];
        } else if (config.contains("n_positions")) {
            max_sequence_length_ = config["n_positions"];
        } else if (config.contains("max_length")) {
            max_sequence_length_ = config["max_length"];
        }
        
        if (config.contains("model_type")) {
            tokenizer_type_ = config["model_type"];
        }
        
        if (verbose_) {
            std::cout << "Loaded config: model_type=" << tokenizer_type_ 
                      << ", max_length=" << max_sequence_length_ << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}

bool Embedder::loadTokenizerConfig(const std::string& tokenizer_config_path) {
    try {
        std::ifstream file(tokenizer_config_path);
        if (!file.is_open()) {
            std::cerr << "Error opening tokenizer config file: " << tokenizer_config_path << std::endl;
            return false;
        }
        
        nlohmann::json config;
        file >> config;
        
        if (config.contains("do_lower_case")) {
            do_lower_case_ = config["do_lower_case"];
        }
        
        if (config.contains("do_basic_tokenize")) {
            do_basic_tokenize_ = config["do_basic_tokenize"];
        }
        
        // Handle different ways models specify max sequence length
        if (config.contains("max_len")) {
            max_sequence_length_ = config["max_len"];
        } else if (config.contains("model_max_length")) {
            max_sequence_length_ = config["model_max_length"];
        }
        
        // Extract special tokens from tokenizer config if available
        if (config.contains("cls_token")) {
            cls_token_ = config["cls_token"];
        }
        if (config.contains("sep_token")) {
            sep_token_ = config["sep_token"];
        }
        if (config.contains("pad_token")) {
            pad_token_ = config["pad_token"];
        }
        if (config.contains("unk_token")) {
            unk_token_ = config["unk_token"];
        }
        if (config.contains("mask_token")) {
            mask_token_ = config["mask_token"];
        }
        
        // Handle added_tokens_decoder (HuggingFace format)
        if (config.contains("added_tokens_decoder")) {
            auto added_tokens = config["added_tokens_decoder"];
            for (auto& [id_str, token_info] : added_tokens.items()) {
                int id = std::stoi(id_str);
                if (token_info.contains("content")) {
                    std::string content = token_info["content"];
                    if (content == "[PAD]") pad_token_id_ = id;
                    else if (content == "[UNK]") unk_token_id_ = id;
                    else if (content == "[CLS]") cls_token_id_ = id;
                    else if (content == "[SEP]") sep_token_id_ = id;
                    else if (content == "[MASK]") mask_token_id_ = id;
                }
            }
        }
        
        if (verbose_) {
            std::cout << "Loaded tokenizer config: do_lower_case=" << do_lower_case_ 
                      << ", do_basic_tokenize=" << do_basic_tokenize_
                      << ", max_length=" << max_sequence_length_ << std::endl;
            std::cout << "Special token IDs from tokenizer_config: PAD=" << pad_token_id_
                      << ", UNK=" << unk_token_id_ 
                      << ", CLS=" << cls_token_id_
                      << ", SEP=" << sep_token_id_
                      << ", MASK=" << mask_token_id_ << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading tokenizer config: " << e.what() << std::endl;
        return false;
    }
}

bool Embedder::loadSpecialTokensMap(const std::string& special_tokens_path) {
    try {
        std::ifstream file(special_tokens_path);
        if (!file.is_open()) {
            std::cerr << "Error opening special tokens file: " << special_tokens_path << std::endl;
            return false;
        }
        
        nlohmann::json tokens;
        file >> tokens;
        
        if (tokens.contains("unk_token")) {
            if (tokens["unk_token"].is_string()) {
                unk_token_ = tokens["unk_token"];
            } else if (tokens["unk_token"].contains("content")) {
                unk_token_ = tokens["unk_token"]["content"];
            }
        }
        
        if (tokens.contains("pad_token")) {
            if (tokens["pad_token"].is_string()) {
                pad_token_ = tokens["pad_token"];
            } else if (tokens["pad_token"].contains("content")) {
                pad_token_ = tokens["pad_token"]["content"];
            }
        }
        
        if (tokens.contains("cls_token")) {
            if (tokens["cls_token"].is_string()) {
                cls_token_ = tokens["cls_token"];
            } else if (tokens["cls_token"].contains("content")) {
                cls_token_ = tokens["cls_token"]["content"];
            }
        }
        
        if (tokens.contains("sep_token")) {
            if (tokens["sep_token"].is_string()) {
                sep_token_ = tokens["sep_token"];
            } else if (tokens["sep_token"].contains("content")) {
                sep_token_ = tokens["sep_token"]["content"];
            }
        }
        
        if (tokens.contains("mask_token")) {
            if (tokens["mask_token"].is_string()) {
                mask_token_ = tokens["mask_token"];
            } else if (tokens["mask_token"].contains("content")) {
                mask_token_ = tokens["mask_token"]["content"];
            }
        }
        
        if (verbose_) {
            std::cout << "Loaded special tokens: UNK='" << unk_token_ 
                      << "', PAD='" << pad_token_ 
                      << "', CLS='" << cls_token_
                      << "', SEP='" << sep_token_
                      << "', MASK='" << mask_token_ << "'" << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading special tokens: " << e.what() << std::endl;
        return false;
    }
}

bool Embedder::loadBPEMerges(const std::string& merges_path) {
    try {
        std::ifstream file(merges_path);
        if (!file.is_open()) {
            std::cerr << "Error opening merges file: " << merges_path << std::endl;
            return false;
        }
        
        bpe_merges_.clear();
        std::string line;
        int rank = 0;
        
        // Skip header line if it exists
        if (std::getline(file, line) && line.find("#version") != 0) {
            // This wasn't a header, process it
            std::istringstream iss(line);
            std::string part1, part2;
            if (iss >> part1 >> part2) {
                bpe_merges_[part1 + " " + part2] = std::to_string(rank++);
            }
        }
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::istringstream iss(line);
            std::string part1, part2;
            if (iss >> part1 >> part2) {
                bpe_merges_[part1 + " " + part2] = std::to_string(rank++);
            }
        }
        
        if (verbose_) {
            std::cout << "Loaded " << bpe_merges_.size() << " BPE merges" << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading BPE merges: " << e.what() << std::endl;
        return false;
    }
}

std::string Embedder::findModelFile(const std::string& folder_path, const std::vector<std::string>& extensions) {
    try {
        // Common model file patterns to search for
        std::vector<std::string> search_patterns = {
            "model.onnx",           // Standard ONNX model
            "model_fp16.onnx",      // Half-precision model
            "model_quantized.onnx", // Quantized model
            "pytorch_model.onnx",   // PyTorch converted model
        };
        
        // First, check common subdirectories where models are often stored
        std::vector<std::string> subdirs = {"", "onnx/", "models/"};
        
        for (const std::string& subdir : subdirs) {
            std::string search_path = folder_path;
            if (!subdir.empty()) {
                search_path += "/" + subdir;
                // Remove trailing slash if it exists
                if (search_path.back() == '/') {
                    search_path.pop_back();
                }
            }
            
            if (!std::filesystem::exists(search_path)) {
                continue;
            }
            
            // Check for specific model file patterns first
            for (const std::string& pattern : search_patterns) {
                std::string full_path = search_path + "/" + pattern;
                if (std::filesystem::exists(full_path)) {
                    if (verbose_) {
                        std::cout << "Found model file: " << full_path << std::endl;
                    }
                    return full_path;
                }
            }
            
            // Then check for any file with the specified extensions
            for (const auto& entry : std::filesystem::directory_iterator(search_path)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    for (const std::string& ext : extensions) {
                        if (filename.length() >= ext.length() && 
                            filename.substr(filename.length() - ext.length()) == ext) {
                            if (verbose_) {
                                std::cout << "Found model file: " << entry.path().string() << std::endl;
                            }
                            return entry.path().string();
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error searching for model file: " << e.what() << std::endl;
    }
    return "";
}

void Embedder::initializeDefaultTokens() {
    // Update special token IDs based on loaded tokens
    if (vocab_.find(unk_token_) != vocab_.end()) {
        unk_token_id_ = vocab_[unk_token_];
    }
    if (vocab_.find(pad_token_) != vocab_.end()) {
        pad_token_id_ = vocab_[pad_token_];
    }
    if (vocab_.find(cls_token_) != vocab_.end()) {
        cls_token_id_ = vocab_[cls_token_];
    }
    if (vocab_.find(sep_token_) != vocab_.end()) {
        sep_token_id_ = vocab_[sep_token_];
    }
    if (vocab_.find(mask_token_) != vocab_.end()) {
        mask_token_id_ = vocab_[mask_token_];
    }
    
    if (verbose_) {
        std::cout << "Initialized special token IDs: UNK=" << unk_token_id_ 
                  << ", PAD=" << pad_token_id_ 
                  << ", CLS=" << cls_token_id_
                  << ", SEP=" << sep_token_id_
                  << ", MASK=" << mask_token_id_ << std::endl;
    }
}

std::vector<std::string> Embedder::bpeTokenize(const std::string& text) {
    // Simple BPE implementation for GPT-style models
    std::vector<std::string> tokens;
    
    if (bpe_merges_.empty()) {
        // Fallback to basic tokenization if no BPE merges loaded
        return basicTokenize(text);
    }
    
    // This is a simplified BPE implementation
    // For production use, you'd want a more complete BPE tokenizer
    std::string processed = preprocessText(text);
    std::istringstream iss(processed);
    std::string word;
    
    while (iss >> word) {
        if (!word.empty()) {
            // For now, just add the word as-is
            // A full BPE implementation would apply the merge rules
            tokens.push_back(word);
        }
    }
    
    return tokens;
}