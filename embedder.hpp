#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <unordered_map>
#include "nlohmann_json.hpp"

class Embedder {
private:
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    
    // Tokenizer-related members
    std::map<std::string, int> vocab_;
    std::map<int, std::string> id_to_token_;
    std::unordered_map<std::string, std::string> bpe_merges_;
    int max_sequence_length_;
    int pad_token_id_;
    int cls_token_id_;
    int sep_token_id_;
    int unk_token_id_;
    int mask_token_id_;
    bool verbose_;
    
    // Tokenization configuration
    std::string tokenizer_type_;  // "bert", "roberta", "gpt2", etc.
    bool do_lower_case_;
    bool do_basic_tokenize_;
    std::string unk_token_;
    std::string pad_token_;
    std::string cls_token_;
    std::string sep_token_;
    std::string mask_token_;
    
public:
    Embedder();
    ~Embedder();
    
    void setVerbose(bool verbose);
    bool loadModel(const std::string& model_path);
    bool loadTokenizer(const std::string& vocab_path);
    bool loadModelFolder(const std::string& model_folder_path);  // New method
    std::vector<float> getEmbedding(const std::string& text);
    
    // Batch processing methods
    std::vector<std::vector<float>> getBatchEmbeddings(const std::vector<std::string>& texts);
    
    // Public tokenizer methods for advanced usage
    std::vector<int> tokenize(const std::string& text);
    std::vector<int> encodeText(const std::string& text);
    
    // Batch tokenization methods
    std::vector<std::vector<int>> batchTokenize(const std::vector<std::string>& texts);
    std::vector<std::vector<int>> batchEncodeText(const std::vector<std::string>& texts);
    
    // Advanced tokenization options
    void setSpecialTokens(int pad_id, int unk_id, int cls_id, int sep_id);
    void setMaxSequenceLength(int max_length);
    std::string detokenize(const std::vector<int>& token_ids);
    
    // Configuration loading methods (public for testing)
    bool loadConfig(const std::string& config_path);
    bool loadTokenizerConfig(const std::string& tokenizer_config_path);
    bool loadSpecialTokensMap(const std::string& special_tokens_path);
    bool loadBPEMerges(const std::string& merges_path);
    
private:
    // Tokenization methods
    std::string preprocessText(const std::string& text);
    std::vector<std::string> basicTokenize(const std::string& text);
    std::vector<std::string> wordpieceTokenize(const std::string& word);
    std::vector<std::string> bpeTokenize(const std::string& text);  // New BPE method
    int getTokenId(const std::string& token);
    void normalizeEmbedding(std::vector<float>& embedding);
    void normalizeBatchEmbeddings(std::vector<std::vector<float>>& embeddings);
    
    // Helper methods
    std::string findModelFile(const std::string& folder_path, const std::vector<std::string>& extensions);
    void initializeDefaultTokens();
};