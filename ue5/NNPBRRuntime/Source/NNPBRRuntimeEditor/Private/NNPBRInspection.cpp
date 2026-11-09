#include "NNPBREditor.h"
#include "NNPBRInspection.h"
#include "NNPBRLatentTexture.h"
#include "Engine/Texture2D.h"
#include "Materials/MaterialInstanceConstant.h"
#include "Framework/Application/SlateApplication.h"
#include "IDetailsView.h"
#include "PropertyEditorModule.h"
#include "Misc/MessageDialog.h"
#include "Modules/ModuleManager.h"
#include "UObject/StrongObjectPtr.h"
#include "Widgets/Images/SImage.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/SWindow.h"
#include "Widgets/Text/STextBlock.h"

void NNPBREditor::InspectMaterial(UMaterialInstanceConstant *Instance)
{
    TStrongObjectPtr<UNNPBRInspection> Model(NewObject<UNNPBRInspection>());
    Model->Material = Instance->GetPathName();
    const auto Invalid = [] {
        FMessageDialog::Open(
            EAppMsgType::Ok,
            FText::FromString(TEXT("This instance does not contain a complete imported NN-PBR material.")));
    };
    float Weights[344];
    for (int32 I = 0; I < 86; ++I)
    {
        FLinearColor V;
        if (!Instance->GetVectorParameterValue(FMaterialParameterInfo(*FString::Printf(TEXT("NNW%02d"), I)), V))
        {
            Invalid();
            return;
        }
        Weights[I * 4] = V.R;
        Weights[I * 4 + 1] = V.G;
        Weights[I * 4 + 2] = V.B;
        Weights[I * 4 + 3] = V.A;
    }
    for (int32 J = 0; J < 16; ++J)
    {
        auto &Row = Model->HiddenLayer.AddDefaulted_GetRef();
        Row.Output = FString::Printf(TEXT("Hidden %02d"), J);
        Row.InputWeights.Append(Weights + J * 12, 12);
        Row.Bias = Weights[192 + J];
    }
    const TCHAR *Names[] = {TEXT("Albedo R"), TEXT("Albedo G"), TEXT("Albedo B"),  TEXT("Normal X"),
                            TEXT("Normal Y"), TEXT("AO"),       TEXT("Roughness"), TEXT("Metallic")};
    for (int32 J = 0; J < 8; ++J)
    {
        auto &Row = Model->OutputLayer.AddDefaulted_GetRef();
        Row.Output = Names[J];
        Row.InputWeights.Append(Weights + 208 + J * 16, 16);
        Row.Bias = Weights[336 + J];
    }

    auto Strip = SNew(SHorizontalBox);
    for (int32 I = 0; I < 4; ++I)
    {
        UTexture *Bound = nullptr;
        Instance->GetTextureParameterValue(FMaterialParameterInfo(*FString::Printf(TEXT("NNLatent%d"), I)),
                                           Bound);
        auto *Texture = Cast<UNNPBRLatentTexture>(Bound);
        auto *Preview = Texture ? Texture->CreatePreview(Model.Get()) : nullptr;
        if (!Preview)
        {
            Invalid();
            return;
        }
        Model->Previews.Add(Preview);
        auto Brush = MakeShared<FSlateBrush>();
        Brush->DrawAs = ESlateBrushDrawType::Image;
        Brush->SetResourceObject(Preview);
        const FString Caption =
            FString::Printf(TEXT("L%d  ·  %d²  ·  %d mips"), I, Preview->GetSizeX(), Texture->GetMipCount());
        Strip->AddSlot().FillWidth(1).Padding(
            4)[SNew(SVerticalBox) +
               SVerticalBox::Slot().AutoHeight().HAlign(
                   HAlign_Center)[SNew(SBox).WidthOverride(112).HeightOverride(
                   112)[SNew(SImage).Image_Lambda([Brush] { return Brush.Get(); })]] +
               SVerticalBox::Slot().AutoHeight().HAlign(
                   HAlign_Center)[SNew(STextBlock).Text(FText::FromString(Caption))]];
    }
    FDetailsViewArgs Args;
    Args.bAllowSearch = true;
    auto Details =
        FModuleManager::LoadModuleChecked<FPropertyEditorModule>("PropertyEditor").CreateDetailView(Args);
    Details->SetObject(Model.Get());
    auto Window =
        SNew(SWindow)
            .Title(FText::FromString(TEXT("NN-PBR Details · ") + Instance->GetName()))
            .ClientSize(FVector2D(680, 600))[SNew(SVerticalBox) + SVerticalBox::Slot().AutoHeight()[Strip] +
                                             SVerticalBox::Slot().FillHeight(1)[Details]];
    // The property view holds a weak object pointer. Keep the inspected snapshot and thumbnails alive until
    // close.
    Window->SetOnWindowClosed(FOnWindowClosed::CreateLambda([Model](const TSharedRef<SWindow> &) {}));
    FSlateApplication::Get().AddWindow(Window);
}
