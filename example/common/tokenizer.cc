#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    std::ifstream file(filepath, std::ios::binary);
    CHECK(file.is_open()) << "Failed to open " << filepath;
    auto header = ReadSeveralBytesFromIfstream(1024, &file);
    magic_number_ = BytesToType<uint32_t>(header, 0);
    int32_t version = BytesToType<int32_t>(header, 4);
    int32_t vocab_size = BytesToType<int32_t>(header, 8);
    
    if (kEotMap.count(magic_number_)) {
        eot_token_ = kEotMap.at(magic_number_);
    } else {
        LOG(WARNING) << "Unknown magic number: " << magic_number_ << ", using default EOT.";
        eot_token_ = 0;
    }

    for (int i = 0; i < vocab_size; ++i) {
        uint8_t len_byte;
        file.read(reinterpret_cast<char*>(&len_byte), 1);
        int len = static_cast<int>(len_byte);
        std::string token_str(len, '\0');
        file.read(&token_str[0], len);
        token_table_.push_back(token_str);
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    if (token_id < token_table_.size()) return token_table_[token_id];
    return "";
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t kRngState = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        
        // Forward pass
        auto logits_vec = model.Forward({x}); 
        auto logits = logits_vec[0]; // (batch, seq_len, vocab)
        
        // Extract next token logits (at seq_len t-1)
        auto next_token_logits = logits->Slice(1, t - 1, t, 1); // (batch, 1, vocab)
        
        // Copy to CPU for sampling
        auto cpu_logits = std::make_shared<infini_train::Tensor>(next_token_logits->To(infini_train::Device(infini_train::DeviceType::kCPU, 0)));
        float* logits_ptr = static_cast<float*>(cpu_logits->DataPtr());
        int64_t vocab_size = cpu_logits->Dims().back();
        
        // Softmax and Sample
        std::vector<float> probs(vocab_size);
        float max_logit = -1e9;
        for(int i=0; i<vocab_size; ++i) max_logit = std::max(max_logit, logits_ptr[i]);
        float sum = 0.0f;
        for(int i=0; i<vocab_size; ++i) {
            probs[i] = std::exp(logits_ptr[i] - max_logit);
            sum += probs[i];
        }
        for(int i=0; i<vocab_size; ++i) probs[i] /= sum;
        
        float coin = RandomF32(kRngState);
        int token = SampleMult(probs.data(), vocab_size, coin);
        
        // Output
        std::string text = Decode(token);
        std::cout << text << std::flush;
        
        // Update input
        x_buff[t] = token;
        auto new_x = x_tensor.To(device);
        x = std::make_shared<infini_train::Tensor>(new_x);
    }
    std::cout << std::endl;
}
} // namespace infini_train
