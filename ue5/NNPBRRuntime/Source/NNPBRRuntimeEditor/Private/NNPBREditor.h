#pragma once
#include "CoreMinimal.h"

class UMaterialInstanceConstant;
namespace NNPBREditor
{
UMaterialInstanceConstant *ImportMaterial(const FString &MetadataFile, const FString &PackageName,
                                          FString &Error);
void InspectMaterial(UMaterialInstanceConstant *Instance);
} // namespace NNPBREditor
