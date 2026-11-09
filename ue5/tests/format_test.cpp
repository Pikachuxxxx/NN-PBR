#include "NNPBRFormat.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint8_t> Read(const std::filesystem::path &File)
{
    std::ifstream In(File, std::ios::binary);
    if (!In)
        throw std::runtime_error(File.string());
    return {std::istreambuf_iterator<char>(In), std::istreambuf_iterator<char>()};
}

static void Require(bool Ok)
{
    if (!Ok)
        throw std::runtime_error("Unexpected decoder result");
}

int main(int argc, char **argv)
{
    if (argc != 2)
        return 2;
    const std::filesystem::path Dir(argv[1]);
    for (int i = 0; i < 4; ++i)
    {
        auto bytes = Read(Dir / ("latent_0" + std::to_string(i) + ".bc6.dds"));
        NNPBR::Pyramid p{};
        Require(NNPBR::ParseDDS(bytes.data(), bytes.size(), p));
        Require(p.Count && p.Levels[p.Count - 1].Offset + p.Levels[p.Count - 1].Bytes == bytes.size());
        Require(!NNPBR::ParseDDS(bytes.data(), bytes.size() - 1, p));
        bytes[128] = 96; // signed BC6H
        Require(!NNPBR::ParseDDS(bytes.data(), bytes.size(), p));
        bytes[128] = 95;
        bytes[28] = 255;
        Require(!NNPBR::ParseDDS(bytes.data(), bytes.size(), p));
    }
    auto blob = Read(Dir / "decoder_fp16.bin");
    std::array<float, 344> values{};
    Require(NNPBR::DecodeWeights(blob.data(), blob.size(), values));
    Require(!NNPBR::DecodeWeights(blob.data(), blob.size() - 1, values));
    blob[0] = 0;
    blob[1] = 0xbc;
    blob[2] = 1;
    blob[3] = 0;
    Require(NNPBR::DecodeWeights(blob.data(), blob.size(), values));
    Require(values[0] == -1 && values[1] == std::ldexp(1.f, -24));
    blob[1] = 0x7c;
    Require(!NNPBR::DecodeWeights(blob.data(), blob.size(), values));
}
