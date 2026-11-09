#include "Modules/ModuleManager.h"
#include "NNPBREditor.h"
#include "ContentBrowserModule.h"
#include "IContentBrowserSingleton.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Framework/Application/SlateApplication.h"
#include "Materials/MaterialInstanceConstant.h"
#include "Misc/MessageDialog.h"
#include "Misc/Paths.h"
#include "Misc/PackageName.h"
#include "ObjectTools.h"
#include "ToolMenus.h"

namespace
{
void Import()
{
    TArray<FString> Files;
    IDesktopPlatform *Desktop = FDesktopPlatformModule::Get();
    if (!Desktop ||
        !Desktop->OpenFileDialog(FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr),
                                 TEXT("Choose NN-PBR metadata.json"), TEXT(""), TEXT("metadata.json"),
                                 TEXT("NN-PBR metadata (*.json)|*.json"), EFileDialogFlags::None, Files) ||
        Files.IsEmpty())
        return;
    auto &Browser = FModuleManager::LoadModuleChecked<FContentBrowserModule>("ContentBrowser").Get();
    FString Folder = FPaths::GetPath(Files[0]);
    if (FPaths::GetCleanFilename(Folder) == TEXT("export"))
        Folder = FPaths::GetPath(Folder);
    FSaveAssetDialogConfig Config;
    Config.DialogTitleOverride = FText::FromString(TEXT("Save NN-PBR material instance"));
    Config.DefaultPath = TEXT("/Game");
    Config.DefaultAssetName = TEXT("MI_") + ObjectTools::SanitizeObjectName(FPaths::GetCleanFilename(Folder));
    Config.ExistingAssetPolicy = ESaveAssetDialogExistingAssetPolicy::Disallow;
    const FString Path = Browser.CreateModalSaveAssetDialog(Config);
    if (Path.IsEmpty())
        return;
    FString Error;
    auto *Instance = NNPBREditor::ImportMaterial(Files[0], FPackageName::ObjectPathToPackageName(Path), Error);
    if (!Instance)
    {
        FMessageDialog::Open(EAppMsgType::Ok, FText::FromString(Error));
        return;
    }
    Browser.SyncBrowserToAssets(TArray<FAssetData>{FAssetData(Instance)});
    NNPBREditor::InspectMaterial(Instance);
}

void InspectSelected()
{
    TArray<FAssetData> Assets;
    FModuleManager::LoadModuleChecked<FContentBrowserModule>("ContentBrowser").Get().GetSelectedAssets(Assets);
    if (Assets.Num() == 1)
        if (auto *Instance = Cast<UMaterialInstanceConstant>(Assets[0].GetAsset()))
        {
            NNPBREditor::InspectMaterial(Instance);
            return;
        }
    FMessageDialog::Open(
        EAppMsgType::Ok,
        FText::FromString(TEXT("Select one imported NN-PBR material instance in the Content Browser.")));
}
} // namespace
class FNNPBRRuntimeEditorModule final : public IModuleInterface
{
    void StartupModule() override
    {
        UToolMenus::RegisterStartupCallback(
            FSimpleMulticastDelegate::FDelegate::CreateRaw(this, &FNNPBRRuntimeEditorModule::RegisterMenus));
    }
    void ShutdownModule() override
    {
        UToolMenus::UnRegisterStartupCallback(this);
        UToolMenus::UnregisterOwner(this);
    }
    void RegisterMenus()
    {
        FToolMenuOwnerScoped Owner(this);
        auto &Section = UToolMenus::Get()
                            ->ExtendMenu("LevelEditor.MainMenu.Tools")
                            ->FindOrAddSection("NNPBR", FText::FromString(TEXT("NN-PBR")));
        Section.AddMenuEntry("ImportNNPBR", FText::FromString(TEXT("Import NN-PBR Material…")),
                             FText::FromString(TEXT("Create a material from metadata.json.")), FSlateIcon(),
                             FUIAction(FExecuteAction::CreateStatic(&Import)));
        Section.AddMenuEntry(
            "InspectNNPBR", FText::FromString(TEXT("Inspect NN-PBR Material")),
            FText::FromString(TEXT("Inspect the selected material's latents and decoder weights.")),
            FSlateIcon(), FUIAction(FExecuteAction::CreateStatic(&InspectSelected)));
    }
};
IMPLEMENT_MODULE(FNNPBRRuntimeEditorModule, NNPBRRuntimeEditor)
