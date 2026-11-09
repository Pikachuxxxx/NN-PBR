#include "NNPBRLatentTexture.h"
#include "NNPBRExport.h"
#include "PixelFormat.h"
#include "Misc/App.h"
#include "RHICommandList.h"
#include "RHIStaticStates.h"
#include "TextureResource.h"

namespace
{
class FNNPBRTextureResource final : public FTextureResource
{
  public:
    FNNPBRTextureResource(const TArray<uint8> &InBytes, const NNPBR::Pyramid &InLayout,
                          FTextureReference &InReference)
        : Bytes(InBytes), Layout(InLayout), Reference(InReference)
    {
    }

    uint32 GetSizeX() const override
    {
        return Layout.Levels[0].Width;
    }
    uint32 GetSizeY() const override
    {
        return GetSizeX();
    }

    void InitRHI(FRHICommandListBase &Commands) override
    {
        const auto Desc = FRHITextureCreateDesc::Create2D(TEXT("NNPBR latent"), GetSizeX(), GetSizeY(), PF_BC6H)
                              .SetNumMips(Layout.Count)
                              .SetFlags(ETextureCreateFlags::ShaderResource);
        TextureRHI = Commands.CreateTexture(Desc);
        for (uint32 Index = 0; Index < Layout.Count; ++Index)
        {
            const auto &Mip = Layout.Levels[Index];
            const FUpdateTextureRegion2D Region(0, 0, 0, 0, Mip.Width, Mip.Width);
            Commands.UpdateTexture2D(TextureRHI, Index, Region, (Mip.Width / 4) * 16,
                                     Bytes.GetData() + Mip.Offset);
        }
        SamplerStateRHI = TStaticSamplerState<SF_Trilinear, AM_Clamp, AM_Clamp, AM_Clamp>::GetRHI();
        Commands.UpdateTextureReference(Reference.TextureReferenceRHI, TextureRHI);
    }
    void ReleaseRHI() override
    {
        FRHICommandListExecutor::GetImmediateCommandList().UpdateTextureReference(Reference.TextureReferenceRHI,
                                                                                  nullptr);
        FTextureResource::ReleaseRHI();
    }

  private:
    TArray<uint8> Bytes;
    NNPBR::Pyramid Layout;
    FTextureReference &Reference;
};
} // namespace

UNNPBRLatentTexture::UNNPBRLatentTexture()
{
    SRGB = false;
    NeverStream = true;
    Filter = TF_Trilinear;
}
void UNNPBRLatentTexture::Initialize(FNNPBRLayer &&Layer)
{
    Width = Layer.Layout.Levels[0].Width;
    MipCount = Layer.Layout.Count;
    DDS = MoveTemp(Layer.Bytes);
    UpdateResource();
}
void UNNPBRLatentTexture::PostLoad()
{
    Super::PostLoad();
    if (FApp::CanEverRender() && !GetResource())
        UpdateResource();
}
FTextureResource *UNNPBRLatentTexture::CreateResource()
{
    NNPBR::Pyramid Layout{};
    if (!GPixelFormats[PF_BC6H].Supported || !NNPBR::ParseDDS(DDS.GetData(), DDS.Num(), Layout))
        return nullptr;
    return new FNNPBRTextureResource(DDS, Layout, TextureReference);
}
#if WITH_EDITOR
UTexture2D *UNNPBRLatentTexture::CreatePreview(UObject *Owner) const
{
    FNNPBRLayer Layer;
    if (!NNPBR::ParseDDS(DDS.GetData(), DDS.Num(), Layer.Layout))
        return nullptr;
    Layer.Bytes = DDS;
    return NNPBR::Upload(Owner, Layer);
}
#endif
