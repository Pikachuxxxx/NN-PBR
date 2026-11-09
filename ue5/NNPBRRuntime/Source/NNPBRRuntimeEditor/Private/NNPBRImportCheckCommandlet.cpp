#include "NNPBRImportCheckCommandlet.h"
#include "NNPBREditor.h"
#include "NNPBRExport.h"
#include "NNPBRLatentTexture.h"
#include "Materials/MaterialInstanceConstant.h"
#include "Misc/PackageName.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"

UNNPBRImportCheckCommandlet::UNNPBRImportCheckCommandlet()
{
    IsEditor = true;
    IsClient = false;
    IsServer = false;
    LogToConsole = true;
}

int32 UNNPBRImportCheckCommandlet::Main(const FString &Params)
{
    FString Metadata, Package, Error;
    if (!FParse::Value(*Params, TEXT("Metadata="), Metadata) ||
        !FParse::Value(*Params, TEXT("Package="), Package) || !FPackageName::IsValidLongPackageName(Package))
    {
        UE_LOG(LogTemp, Error, TEXT("Supply -Metadata=<absolute metadata.json> -Package=/Game/... [-Verify]."));
        return 1;
    }
    if (!FParse::Param(*Params, TEXT("Verify")))
    {
        if (NNPBREditor::ImportMaterial(Metadata, Package, Error))
            return 0;
        UE_LOG(LogTemp, Error, TEXT("%s"), *Error);
        return 1;
    }
    FNNPBRExport Expected;
    if (!NNPBR::ReadExport(FPaths::GetPath(Metadata), Expected, Error))
    {
        UE_LOG(LogTemp, Error, TEXT("%s"), *Error);
        return 1;
    }
    const FString Path = Package + TEXT(".") + FPackageName::GetLongPackageAssetName(Package);
    auto *Instance = LoadObject<UMaterialInstanceConstant>(nullptr, *Path);
    if (!Instance)
    {
        UE_LOG(LogTemp, Error, TEXT("Cannot reload %s"), *Path);
        return 1;
    }
    for (int32 I = 0; I < 86; ++I)
    {
        FLinearColor Value;
        const float *W = Expected.Weights.data() + 4 * I;
        if (!Instance->GetVectorParameterValue(FMaterialParameterInfo(*FString::Printf(TEXT("NNW%02d"), I)),
                                               Value) ||
            Value != FLinearColor(W[0], W[1], W[2], W[3]))
        {
            UE_LOG(LogTemp, Error, TEXT("Weight mismatch at vector %d"), I);
            return 1;
        }
    }
    for (int32 I = 0; I < 4; ++I)
    {
        UTexture *Bound = nullptr;
        Instance->GetTextureParameterValue(FMaterialParameterInfo(*FString::Printf(TEXT("NNLatent%d"), I)),
                                           Bound);
        auto *Texture = Cast<UNNPBRLatentTexture>(Bound);
        const auto &Layer = Expected.Layers[I];
        if (!Texture || Texture->GetMipCount() != int32(Layer.Layout.Count) ||
            Texture->GetSurfaceWidth() != Layer.Layout.Levels[0].Width ||
            Texture->GetDDS().Num() != Layer.Bytes.Num() ||
            FMemory::Memcmp(Texture->GetDDS().GetData(), Layer.Bytes.GetData(), Layer.Bytes.Num()) != 0)
        {
            UE_LOG(LogTemp, Error, TEXT("DDS or mip inventory mismatch at latent %d"), I);
            return 1;
        }
    }
    UE_LOG(LogTemp, Display, TEXT("Reloaded all 344 weights and four exact BC6H mip chains from %s"), *Path);
    return 0;
}
