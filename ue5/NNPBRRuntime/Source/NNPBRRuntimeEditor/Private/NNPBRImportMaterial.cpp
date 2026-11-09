#include "NNPBREditor.h"
#include "NNPBRCreateMaterialCommandlet.h"
#include "NNPBRExport.h"
#include "NNPBRLatentTexture.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "MaterialEditingLibrary.h"
#include "Materials/MaterialInstanceConstant.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "UObject/SavePackage.h"
#include "HAL/FileManager.h"

UMaterialInstanceConstant *NNPBREditor::ImportMaterial(const FString &MetadataFile, const FString &PackageName,
                                                       FString &Error)
{
    if (!FPaths::GetCleanFilename(MetadataFile).Equals(TEXT("metadata.json"), ESearchCase::CaseSensitive))
    {
        Error = TEXT("Select the metadata.json at the root of an NN-PBR v5 export.");
        return nullptr;
    }
    if (!PackageName.StartsWith(TEXT("/Game/")) || !FPackageName::IsValidLongPackageName(PackageName) ||
        FPackageName::DoesPackageExist(PackageName) || FindPackage(nullptr, *PackageName))
    {
        Error = TEXT("Choose a new asset name under /Game.");
        return nullptr;
    }
    FNNPBRExport Export;
    if (!NNPBR::ReadExport(FPaths::GetPath(MetadataFile), Export, Error))
        return nullptr;
    if (!FPackageName::DoesPackageExist(TEXT("/NNPBRRuntime/Materials/M_NNPBR")) &&
        NewObject<UNNPBRCreateMaterialCommandlet>()->Main(TEXT("")) != 0)
    {
        Error = TEXT("Could not generate the NN-PBR master. Check that the plugin Content folder is writable "
                     "and see the Output Log.");
        return nullptr;
    }
    UMaterialInterface *Parent = NNPBR::ResolveParent(nullptr, Error);
    if (!Parent)
        return nullptr;
    UPackage *Package = CreatePackage(*PackageName);
    const FString Name = FPackageName::GetLongPackageAssetName(PackageName);
    auto *Instance =
        NewObject<UMaterialInstanceConstant>(Package, *Name, RF_Public | RF_Standalone | RF_Transactional);
    UMaterialEditingLibrary::SetMaterialInstanceParent(Instance, Parent);
    for (int32 I = 0; I < 4; ++I)
    {
        const FName Parameter(*FString::Printf(TEXT("NNLatent%d"), I));
        auto *Texture = NewObject<UNNPBRLatentTexture>(Instance, Parameter);
        Texture->Initialize(MoveTemp(Export.Layers[I]));
        Instance->SetTextureParameterValueEditorOnly(FMaterialParameterInfo(Parameter), Texture);
    }
    const auto &W = Export.Weights;
    for (int32 I = 0; I < 86; ++I)
    {
        const FName Parameter(*FString::Printf(TEXT("NNW%02d"), I));
        Instance->SetVectorParameterValueEditorOnly(
            FMaterialParameterInfo(Parameter),
            FLinearColor(W[4 * I], W[4 * I + 1], W[4 * I + 2], W[4 * I + 3]));
    }
    Instance->PostEditChange();
    const FString File =
        FPackageName::LongPackageNameToFilename(PackageName, FPackageName::GetAssetPackageExtension());
    FSavePackageArgs Args;
    Args.TopLevelFlags = RF_Public | RF_Standalone;
    if (!IFileManager::Get().MakeDirectory(*FPaths::GetPath(File), true) ||
        !UPackage::SavePackage(Package, Instance, *File, Args))
    {
        Error = TEXT("Could not save material package: ") + File;
        Instance->ClearFlags(RF_Public | RF_Standalone);
        Instance->MarkAsGarbage();
        Package->SetDirtyFlag(false);
        Package->Rename(nullptr, nullptr, REN_DontCreateRedirectors | REN_NonTransactional);
        return nullptr;
    }
    FAssetRegistryModule::AssetCreated(Instance);
    return Instance;
}
