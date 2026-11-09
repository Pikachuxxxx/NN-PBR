#include "NNPBRCreateMaterialCommandlet.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Engine/Texture2D.h"
#include "HAL/FileManager.h"
#include "MaterialEditingLibrary.h"
#include "Materials/Material.h"
#include "Materials/MaterialExpressionAppendVector.h"
#include "Materials/MaterialExpressionCustom.h"
#include "Materials/MaterialExpressionComponentMask.h"
#include "Materials/MaterialExpressionScalarParameter.h"
#include "Materials/MaterialExpressionTextureSampleParameter2D.h"
#include "Materials/MaterialExpressionVectorParameter.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "UObject/SavePackage.h"
#include "UObject/UnrealType.h"

namespace
{
const TCHAR *Folder = TEXT("/NNPBRRuntime/Materials/");

bool Save(UObject *Asset)
{
    UPackage *Package = Asset->GetOutermost();
    const FString File =
        FPackageName::LongPackageNameToFilename(Package->GetName(), FPackageName::GetAssetPackageExtension());
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(File), true);
    FAssetRegistryModule::AssetCreated(Asset);
    Asset->MarkPackageDirty();
    FSavePackageArgs Args;
    Args.TopLevelFlags = RF_Public | RF_Standalone;
    return UPackage::SavePackage(Package, Asset, *File, Args);
}

template <typename T> T *Node(UMaterial *Mat, int32 X, int32 Y)
{
    return CastChecked<T>(UMaterialEditingLibrary::CreateMaterialExpression(Mat, T::StaticClass(), X, Y));
}

void AddInput(UMaterialExpressionCustom *Custom, FName Name, UMaterialExpression *Expression)
{
    FCustomInput &Input = Custom->Inputs.AddDefaulted_GetRef();
    Input.InputName = Name;
    Input.Input.Connect(0, Expression);
}

UTexture2D *LinearPlaceholder()
{
    const FString Name = TEXT("T_NNPBR_Linear");
    const FString PackageName = FString(Folder) + Name;
    if (FPackageName::DoesPackageExist(PackageName))
        return LoadObject<UTexture2D>(nullptr, *(PackageName + TEXT(".") + Name));
    auto *Texture = NewObject<UTexture2D>(CreatePackage(*PackageName), *Name, RF_Public | RF_Standalone);
    uint8 Pixels[4 * 4 * 4] = {};
    Texture->Source.Init(4, 4, 1, 1, TSF_BGRA8, Pixels);
    Texture->SRGB = false;
    Texture->CompressionSettings = TC_VectorDisplacementmap;
    Texture->AddressX = Texture->AddressY = TA_Clamp;
    Texture->PostEditChange();
    return Save(Texture) ? Texture : nullptr;
}
} // namespace

UNNPBRCreateMaterialCommandlet::UNNPBRCreateMaterialCommandlet()
{
    IsClient = false;
    IsServer = false;
    IsEditor = true;
    LogToConsole = true;
}

int32 UNNPBRCreateMaterialCommandlet::Main(const FString &Params)
{
    const FString Name = TEXT("M_NNPBR");
    const FString PackageName = FString(Folder) + Name;
    if (FPackageName::DoesPackageExist(PackageName))
    {
        UE_LOG(LogTemp, Display, TEXT("NN-PBR master already exists: %s"), *PackageName);
        return 0;
    }
    UTexture2D *Placeholder = LinearPlaceholder();
    if (!Placeholder)
        return 1;
    auto *Mat = NewObject<UMaterial>(CreatePackage(*PackageName), *Name, RF_Public | RF_Standalone);
    Mat->SetShadingModel(MSM_DefaultLit);
    Mat->bTangentSpaceNormal = true;
    auto *Custom = Node<UMaterialExpressionCustom>(Mat, 0, 0);
    Custom->Description = TEXT("NN-PBR decoder");
    Custom->OutputType = CMOT_Float3;
    Custom->IncludeFilePaths.Add(TEXT("/Plugin/NNPBRRuntime/Private/NNPBRMaterial.ush"));
    Custom->Inputs.Reset();
    for (int32 I = 0; I < 4; ++I)
    {
        auto *Tex = Node<UMaterialExpressionTextureSampleParameter2D>(Mat, -480, I * 180);
        Tex->ParameterName = *FString::Printf(TEXT("NNLatent%d"), I);
        Tex->Texture = Placeholder;
        Tex->SamplerType = SAMPLERTYPE_LinearColor;
        AddInput(Custom, *FString::Printf(TEXT("L%d"), I), Tex);
    }
    auto *Sign = Node<UMaterialExpressionScalarParameter>(Mat, -480, 720);
    Sign->ParameterName = TEXT("NNNormalYSign");
    Sign->DefaultValue = -1.0f;
    AddInput(Custom, TEXT("NormalSign"), Sign);
    FString Code = TEXT("float4 w[86];\n");
    for (int32 I = 0; I < 86; ++I)
    {
        const FName Param(*FString::Printf(TEXT("NNW%02d"), I));
        auto *Weight = Node<UMaterialExpressionVectorParameter>(Mat, -960, 900 + I * 140);
        Weight->ParameterName = Param;
        Weight->DefaultValue = FLinearColor(0, 0, 0, 0);
        // VectorParameter output 0 is RGB; output 4 is its alpha component.
        auto *RGBA = Node<UMaterialExpressionAppendVector>(Mat, -480, 900 + I * 140);
        RGBA->A.Connect(0, Weight);
        RGBA->B.Connect(4, Weight);
        AddInput(Custom, Param, RGBA);
        Code += FString::Printf(TEXT("w[%d] = %s;\n"), I, *Param.ToString());
    }
    for (const TCHAR *Output : {TEXT("NormalTS"), TEXT("ORM")})
    {
        auto &Pin = Custom->AdditionalOutputs.AddDefaulted_GetRef();
        Pin.OutputName = Output;
        Pin.OutputType = CMOT_Float3;
    }
    Custom->Code = Code + TEXT("return NNPBRDecode(L0,L1,L2,L3,w,NormalSign,NormalTS,ORM);");
    // PostEditChange refreshes the custom node's additional output pins.
    FPropertyChangedEvent PinsChanged(
        FindFPropertyChecked<FProperty>(UMaterialExpressionCustom::StaticClass(),
                                        GET_MEMBER_NAME_CHECKED(UMaterialExpressionCustom, AdditionalOutputs)));
    static_cast<UObject *>(Custom)->PostEditChangeProperty(PinsChanged);
    bool Connected = UMaterialEditingLibrary::ConnectMaterialProperty(Custom, TEXT(""), MP_BaseColor) &&
                     UMaterialEditingLibrary::ConnectMaterialProperty(Custom, TEXT("NormalTS"), MP_Normal);
    const EMaterialProperty Properties[] = {MP_AmbientOcclusion, MP_Roughness, MP_Metallic};
    for (int32 I = 0; I < 3; ++I)
    {
        auto *Mask = Node<UMaterialExpressionComponentMask>(Mat, 400, I * 120);
        Mask->R = I == 0;
        Mask->G = I == 1;
        Mask->B = I == 2;
        Mask->A = false;
        Connected &= UMaterialEditingLibrary::ConnectMaterialExpressions(Custom, TEXT("ORM"), Mask, TEXT(""));
        Connected &= UMaterialEditingLibrary::ConnectMaterialProperty(Mask, TEXT(""), Properties[I]);
    }
    if (!Connected)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to wire the NN-PBR material outputs."));
        return 1;
    }
    Mat->PostEditChange();
    return Save(Mat) ? 0 : 1;
}
