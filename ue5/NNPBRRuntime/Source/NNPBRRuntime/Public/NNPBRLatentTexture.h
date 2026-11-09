#pragma once

#include "Engine/Texture.h"
#include "MaterialShared.h"
#include "NNPBRLatentTexture.generated.h"

struct FNNPBRLayer;
class UTexture2D;

// Store the exported DDS verbatim in the material package. Texture DDC would recompress its blocks.
UCLASS()
class NNPBRRUNTIME_API UNNPBRLatentTexture : public UTexture
{
    GENERATED_BODY()
  public:
    UNNPBRLatentTexture();
    void Initialize(FNNPBRLayer &&Layer);
    void PostLoad() override;
    FTextureResource *CreateResource() override;
    EMaterialValueType GetMaterialType() const override
    {
        return MCT_Texture2D;
    }
    ETextureClass GetTextureClass() const override
    {
        return ETextureClass::Other2DNoSource;
    }
    float GetSurfaceWidth() const override
    {
        return Width;
    }
    float GetSurfaceHeight() const override
    {
        return Width;
    }
    float GetSurfaceDepth() const override
    {
        return 0;
    }
    uint32 GetSurfaceArraySize() const override
    {
        return 0;
    }
    int32 GetMipCount() const
    {
        return MipCount;
    }
#if WITH_EDITOR
    TConstArrayView<uint8> GetDDS() const { return DDS; }
    UTexture2D *CreatePreview(UObject *Owner) const;
#endif
  private:
    UPROPERTY()
    TArray<uint8> DDS;
    UPROPERTY()
    int32 Width = 0;
    UPROPERTY()
    int32 MipCount = 0;
};
