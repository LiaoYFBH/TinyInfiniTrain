#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << "Failed to open " << path;
    
    // 读取 Header
    const size_t header_size = 1024;
    std::vector<uint8_t> header = ReadSeveralBytesFromIfstream(header_size, &file);
    
    int32_t magic = BytesToType<int32_t>(header, 0);
    int32_t version = BytesToType<int32_t>(header, 4);
    int32_t num_toks = BytesToType<int32_t>(header, 8);
    
    CHECK(kTypeMap.count(magic)) << "Unknown magic " << magic;
    TinyShakespeareType type = kTypeMap.at(magic);
    size_t token_size = kTypeToSize.at(type);
    
    // 读取 token 数据
    size_t data_size = num_toks * token_size;
    std::vector<uint8_t> data = ReadSeveralBytesFromIfstream(data_size, &file);
    
    // 构建 Tensor - 转换为 INT64 以兼容 embedding kernel
    int64_t rows = num_toks / sequence_length;
    std::vector<int64_t> dims = {rows, static_cast<int64_t>(sequence_length)};
    infini_train::Tensor tensor(dims, infini_train::DataType::kINT64);
    int64_t* tensor_ptr = static_cast<int64_t*>(tensor.DataPtr());
    
    // 将原始 token 数据转换为 INT64
    size_t total_tokens = rows * sequence_length;
    if (type == TinyShakespeareType::kUINT16) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(data.data());
        for (size_t i = 0; i < total_tokens; ++i) {
            tensor_ptr[i] = static_cast<int64_t>(src[i]);
        }
    } else if (type == TinyShakespeareType::kUINT32) {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(data.data());
        for (size_t i = 0; i < total_tokens; ++i) {
            tensor_ptr[i] = static_cast<int64_t>(src[i]);
        }
    }
    
    // 返回结果
    TinyShakespeareFile result;
    result.type = type;
    result.dims = dims;
    result.tensor = std::move(tensor);
    return result;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
// =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件，使用成员初始化列表初始化 const 成员
    // =================================== 作业 ===================================
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),  // 使用 INT64 大小
      num_samples_(text_file_.dims[0]) {
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
