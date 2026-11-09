#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace NNPBR
{
struct Level
{
    uint32_t Width;
    size_t Offset, Bytes;
};
struct Pyramid
{
    uint32_t Count;
    std::array<Level, 15> Levels;
};

inline uint32_t Read32(const uint8_t *P)
{
    return uint32_t(P[0]) | uint32_t(P[1]) << 8 | uint32_t(P[2]) << 16 | uint32_t(P[3]) << 24;
}

// NN-PBR uses square power-of-two unsigned BC6H textures with mips >= 4x4.
inline bool ParseDDS(const uint8_t *P, size_t Size, Pyramid &Out)
{
    if (Size < 148 || Read32(P) != 0x20534444 || Read32(P + 4) != 124 || Read32(P + 76) != 32 ||
        Read32(P + 84) != 0x30315844 || Read32(P + 128) != 95 || Read32(P + 132) != 3 || Read32(P + 136) != 0 ||
        Read32(P + 140) != 1 || Read32(P + 144) != 0)
        return false;
    const uint32_t Width = Read32(P + 16);
    const uint32_t Count = Read32(P + 28);
    if (Width < 4 || Width > 16384 || (Width & (Width - 1)) || Read32(P + 12) != Width || Count < 1 ||
        Count > Out.Levels.size() || (Width >> (Count - 1)) < 4)
        return false;
    size_t Offset = 148;
    for (uint32_t I = 0; I < Count; ++I)
    {
        const uint32_t W = Width >> I;
        const size_t Bytes = size_t(W / 4) * (W / 4) * 16;
        if (Bytes > Size - Offset)
            return false;
        Out.Levels[I] = {W, Offset, Bytes};
        Offset += Bytes;
    }
    Out.Count = Count;
    return Offset == Size;
}

inline bool DecodeWeights(const uint8_t *P, size_t Size, std::array<float, 344> &Out)
{
    if (Size != Out.size() * 2)
        return false;
    for (size_t I = 0; I < Out.size(); ++I)
    {
        const uint16_t Bits = P[2 * I] | (uint16_t(P[2 * I + 1]) << 8);
        const int Exponent = (Bits >> 10) & 31;
        const int Mantissa = Bits & 1023;
        if (Exponent == 31)
            return false;
        const float Value =
            Exponent ? std::ldexp(float(1024 + Mantissa), Exponent - 25) : std::ldexp(float(Mantissa), -24);
        Out[I] = (Bits & 0x8000) ? -Value : Value;
    }
    return true;
}
} // namespace NNPBR
