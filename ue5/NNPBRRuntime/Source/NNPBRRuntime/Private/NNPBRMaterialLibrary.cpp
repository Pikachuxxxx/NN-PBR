#include "NNPBRMaterialLibrary.h"
#include "NNPBRFormat.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "PixelFormat.h"
#include "TextureResource.h"

THIRD_PARTY_INCLUDES_START
#include <openssl/sha.h>
THIRD_PARTY_INCLUDES_END

namespace
{
struct FLayer
{
    TArray<uint8> Bytes;
    NNPBR::Pyramid Layout{};
};

bool StringEquals(const FJsonObject &Obj, const TCHAR *Key, const TCHAR *Expected)
{
    FString Value;
    return Obj.TryGetStringField(Key, Value) && Value == Expected;
}

bool NumberEquals(const FJsonObject &Obj, const TCHAR *Key, int32 Expected)
{
    double Value;
    return Obj.TryGetNumberField(Key, Value) && Value == Expected;
}

bool LoadVerified(const FString &Dir, const FString &Name, const FJsonObject &Hashes, TArray<uint8> &Bytes,
                  FString &Error)
{
    FString Expected;
    const FString Path = Dir / Name;
    const int64 Size = IFileManager::Get().FileSize(*Path);
    if (!Hashes.TryGetStringField(Name, Expected) || Expected.Len() != SHA256_DIGEST_LENGTH * 2 || Size < 1 ||
        Size > 512LL * 1024 * 1024 || !FFileHelper::LoadFileToArray(Bytes, *Path))
    {
        Error = FString::Printf(TEXT("Missing, oversized or unreadable export file: %s"), *Name);
        return false;
    }
    uint8 Digest[SHA256_DIGEST_LENGTH];
    SHA256(Bytes.GetData(), Bytes.Num(), Digest);
    static constexpr TCHAR Hex[] = TEXT("0123456789abcdef");
    FString Actual;
    Actual.Reserve(2 * SHA256_DIGEST_LENGTH);
    for (const uint8 Byte : Digest)
    {
        Actual.AppendChar(Hex[Byte >> 4]);
        Actual.AppendChar(Hex[Byte & 15]);
    }
    if (!Actual.Equals(Expected, ESearchCase::IgnoreCase))
    {
        Error = TEXT("SHA256 mismatch: ") + Name;
        return false;
    }
    return true;
}

bool CheckInventory(const TArray<TSharedPtr<FJsonValue>> &Inventory, const FLayer (&Layers)[4])
{
    bool Seen[4][15]{};
    int32 Count = 0;
    for (const FLayer &Layer : Layers)
        Count += Layer.Layout.Count;
    if (Count != Inventory.Num())
        return false;
    for (const auto &Item : Inventory)
    {
        const TSharedPtr<FJsonObject> *Entry;
        int32 L, M;
        const TArray<TSharedPtr<FJsonValue>> *Shape;
        if (!Item.IsValid() || !Item->TryGetObject(Entry) ||
            !(*Entry)->TryGetNumberField(TEXT("latent_index"), L) ||
            !(*Entry)->TryGetNumberField(TEXT("mip_index"), M) || L < 0 || L >= 4 || M < 0 ||
            uint32(M) >= Layers[L].Layout.Count || Seen[L][M] ||
            !(*Entry)->TryGetArrayField(TEXT("shape_chw"), Shape) || Shape->Num() != 3)
            return false;
        const double W = Layers[L].Layout.Levels[M].Width;
        for (int32 C = 0; C < 3; ++C)
        {
            double Value;
            if (!(*Shape)[C]->TryGetNumber(Value) || Value != (C == 0 ? 3.0 : W))
                return false;
        }
        Seen[L][M] = true;
    }
    return true;
}

UTexture2D *Upload(UObject *Owner, const FLayer &Layer)
{
    UTexture2D *Texture = NewObject<UTexture2D>(Owner, NAME_None, RF_Transient);
    auto *Data = new FTexturePlatformData();
    Data->SizeX = Data->SizeY = Layer.Layout.Levels[0].Width;
    Data->PixelFormat = PF_BC6H;
    Data->SetNumSlices(1);
    Texture->SetPlatformData(Data);
    for (uint32 Index = 0; Index < Layer.Layout.Count; ++Index)
    {
        const auto &Level = Layer.Layout.Levels[Index];
        auto *Mip = new FTexture2DMipMap();
        Data->Mips.Add(Mip);
        Mip->SizeX = Mip->SizeY = Level.Width;
        Mip->SizeZ = 1;
        Mip->BulkData.Lock(LOCK_READ_WRITE);
        void *Dest = Mip->BulkData.Realloc(Level.Bytes);
        FMemory::Memcpy(Dest, Layer.Bytes.GetData() + Level.Offset, Level.Bytes);
        Mip->BulkData.Unlock();
    }
    Texture->NeverStream = true;
    Texture->SRGB = false;
    Texture->Filter = TF_Trilinear;
    Texture->AddressX = Texture->AddressY = TA_Clamp;
    Texture->UpdateResource();
    return Texture;
}
} // namespace

UMaterialInstanceDynamic *UNNPBRMaterialLibrary::LoadNeuralMaterial(UObject *WorldContextObject,
                                                                    const FString &ExportDirectory,
                                                                    UMaterialInterface *ParentMaterial,
                                                                    FString &Error)
{
    Error.Reset();
    if (!IsInGameThread() || !IsValid(WorldContextObject) || ExportDirectory.IsEmpty())
    {
        Error = TEXT("Supply an export directory and world object on the game thread.");
        return nullptr;
    }
    if (!GPixelFormats[PF_BC6H].Supported)
    {
        Error = TEXT("The active rendering device does not support BC6H.");
        return nullptr;
    }
    const FString Dir =
        FPaths::IsRelative(ExportDirectory) ? FPaths::ProjectContentDir() / ExportDirectory : ExportDirectory;
    FString Text;
    TSharedPtr<FJsonObject> Meta;
    const FString MetaPath = Dir / TEXT("metadata.json");
    if (IFileManager::Get().FileSize(*MetaPath) > 1024 * 1024 ||
        !FFileHelper::LoadFileToString(Text, *MetaPath) ||
        !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Meta) || !Meta.IsValid())
    {
        Error = TEXT("Cannot read export metadata: ") + MetaPath;
        return nullptr;
    }
    const TSharedPtr<FJsonObject> *Decoder;
    const TSharedPtr<FJsonObject> *Hashes;
    const TArray<TSharedPtr<FJsonValue>> *Resolutions, *Biases, *Inventory;
    int32 ReferenceSize;
    bool Signed;
    if (!NumberEquals(*Meta, TEXT("version"), 5) || !NumberEquals(*Meta, TEXT("latent_count"), 4) ||
        !NumberEquals(*Meta, TEXT("bc6_mode"), 10) || !NumberEquals(*Meta, TEXT("endpoint_bits"), 6) ||
        !NumberEquals(*Meta, TEXT("index_bits"), 3) ||
        !Meta->TryGetBoolField(TEXT("bc6_signed_mode"), Signed) || Signed ||
        !StringEquals(*Meta, TEXT("bc6_format"), TEXT("DXGI_FORMAT_BC6H_UF16")) ||
        !StringEquals(*Meta, TEXT("address_mode"), TEXT("clamp_to_edge")) ||
        !StringEquals(*Meta, TEXT("channel_schema"), TEXT("albedo_srgb01_normal_xy_signed_orm_signed_v1")) ||
        !Meta->TryGetObjectField(TEXT("decoder"), Decoder) || !NumberEquals(**Decoder, TEXT("in_dim"), 12) ||
        !NumberEquals(**Decoder, TEXT("hidden_dim"), 16) || !NumberEquals(**Decoder, TEXT("out_dim"), 8) ||
        !StringEquals(**Decoder, TEXT("weights_fp16_blob"), TEXT("decoder_fp16.bin")) ||
        !Meta->TryGetObjectField(TEXT("sha256"), Hashes) ||
        !Meta->TryGetNumberField(TEXT("reference_resolution"), ReferenceSize) || ReferenceSize < 4 ||
        !Meta->TryGetArrayField(TEXT("latent_resolutions"), Resolutions) || Resolutions->Num() != 4 ||
        !Meta->TryGetArrayField(TEXT("lod_biases"), Biases) || Biases->Num() != 4 ||
        !Meta->TryGetArrayField(TEXT("latent_files"), Inventory))
    {
        Error = TEXT("Unsupported or incomplete NN-PBR v5 metadata.");
        return nullptr;
    }
    TArray<uint8> Blob;
    std::array<float, 344> Weights;
    if (!LoadVerified(Dir, TEXT("decoder_fp16.bin"), **Hashes, Blob, Error))
        return nullptr;
    if (!NNPBR::DecodeWeights(Blob.GetData(), Blob.Num(), Weights))
    {
        Error = TEXT("Expected 344 finite little-endian FP16 decoder values.");
        return nullptr;
    }
    FLayer Layers[4];
    for (int32 I = 0; I < 4; ++I)
    {
        FLayer &Layer = Layers[I];
        const FString File = FString::Printf(TEXT("latent_%02d.bc6.dds"), I);
        if (!LoadVerified(Dir, File, **Hashes, Layer.Bytes, Error))
            return nullptr;
        double Resolution, Bias;
        if (!NNPBR::ParseDDS(Layer.Bytes.GetData(), Layer.Bytes.Num(), Layer.Layout) ||
            !(*Resolutions)[I]->TryGetNumber(Resolution) || Resolution != Layer.Layout.Levels[0].Width ||
            !(*Biases)[I]->TryGetNumber(Bias) ||
            !FMath::IsNearlyEqual(Bias, static_cast<double>(FMath::Log2(float(Resolution) / ReferenceSize)),
                                  1e-6))
        {
            Error = TEXT("Invalid DDS format, mip payload or metadata: ") + File;
            return nullptr;
        }
    }
    if (!CheckInventory(*Inventory, Layers))
    {
        Error = TEXT("DDS mip inventory disagrees with metadata.");
        return nullptr;
    }
    if (!ParentMaterial)
        ParentMaterial =
            LoadObject<UMaterialInterface>(nullptr, TEXT("/NNPBRRuntime/Materials/M_NNPBR.M_NNPBR"));
    if (!ParentMaterial)
    {
        Error = TEXT("Generate and cook the M_NNPBR master material first.");
        return nullptr;
    }
    TArray<FMaterialParameterInfo> Infos;
    TArray<FGuid> Ids;
    ParentMaterial->GetAllTextureParameterInfo(Infos, Ids);
    TSet<FName> TextureNames;
    for (const auto &Info : Infos)
        TextureNames.Add(Info.Name);
    Infos.Reset();
    Ids.Reset();
    ParentMaterial->GetAllVectorParameterInfo(Infos, Ids);
    TSet<FName> WeightNames;
    for (const auto &Info : Infos)
        WeightNames.Add(Info.Name);
    for (int32 I = 0; I < 86; ++I)
    {
        if (!WeightNames.Contains(FName(*FString::Printf(TEXT("NNW%02d"), I))) ||
            (I < 4 && !TextureNames.Contains(FName(*FString::Printf(TEXT("NNLatent%d"), I)))))
        {
            Error = TEXT("Parent material does not declare the NN-PBR texture and weight parameters.");
            return nullptr;
        }
    }
    UMaterialInstanceDynamic *MID = UMaterialInstanceDynamic::Create(ParentMaterial, WorldContextObject);
    if (!MID)
    {
        Error = TEXT("Could not create the neural material instance.");
        return nullptr;
    }
    for (int32 I = 0; I < 4; ++I)
        MID->SetTextureParameterValue(FName(*FString::Printf(TEXT("NNLatent%d"), I)), Upload(MID, Layers[I]));
    for (int32 I = 0; I < 86; ++I)
        MID->SetVectorParameterValue(
            FName(*FString::Printf(TEXT("NNW%02d"), I)),
            FLinearColor(Weights[I * 4], Weights[I * 4 + 1], Weights[I * 4 + 2], Weights[I * 4 + 3]));
    return MID;
}
