#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <map>

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
    int max_sequence_length_;
    int pad_token_id_;
    int cls_token_id_;
    int sep_token_id_;
    int unk_token_id_;
    bool verbose_;
    
public:
    Embedder();
    ~Embedder();
    
    void setVerbose(bool verbose);
    bool loadModel(const std::string& model_path);
    bool loadTokenizer(const std::string& vocab_path);
    std::vector<float> getEmbedding(const std::string& text);
    
    // Public tokenizer methods for advanced usage
    std::vector<int> tokenize(const std::string& text);
    std::vector<int> encodeText(const std::string& text);
    
    // Advanced tokenization options
    void setSpecialTokens(int pad_id, int unk_id, int cls_id, int sep_id);
    std::string detokenize(const std::vector<int>& token_ids);
    
private:
    std::string preprocessText(const std::string& text);
    std::vector<std::string> basicTokenize(const std::string& text);
    std::vector<std::string> wordpieceTokenize(const std::string& word);
    int getTokenId(const std::string& token);
    void normalizeEmbedding(std::vector<float>& embedding);
};