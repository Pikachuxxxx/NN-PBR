#pragma once

#include "CoreMinimal.h"
#include "NNPBRFormat.h"

class UTexture2D;
class UMaterialInterface;

struct FNNPBRLayer
{
    TArray<uint8> Bytes;
    NNPBR::Pyramid Layout{};
};

struct FNNPBRExport
{
    FNNPBRLayer Layers[4];
    std::array<float, 344> Weights{};
};

namespace NNPBR
{
NNPBRRUNTIME_API bool ReadExport(const FString &Directory, FNNPBRExport &Out, FString &Error);
NNPBRRUNTIME_API UMaterialInterface *ResolveParent(UMaterialInterface *Parent, FString &Error);
NNPBRRUNTIME_API UTexture2D *Upload(UObject *Owner, const FNNPBRLayer &Layer);
} // namespace NNPBR
