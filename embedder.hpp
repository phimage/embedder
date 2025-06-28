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
    int max_sequence_length_;
    int pad_token_id_;
    int cls_token_id_;
    int sep_token_id_;
    bool verbose_;
    
public:
    Embedder();
    ~Embedder();
    
    void setVerbose(bool verbose);
    bool loadModel(const std::string& model_path);
    bool loadTokenizer(const std::string& vocab_path);
    std::vector<float> getEmbedding(const std::string& text);
    
private:
    std::vector<int> tokenize(const std::string& text);
    std::vector<int> encodeText(const std::string& text);
    void normalizeEmbedding(std::vector<float>& embedding);
};