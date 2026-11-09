#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "NNPBRMaterialLibrary.generated.h"

class UMaterialInstanceDynamic;
class UMaterialInterface;

UCLASS()
class NNPBRRUNTIME_API UNNPBRMaterialLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
  public:
    /** Relative paths start at the project's Content directory. Use the generated M_NNPBR parent. */
    UFUNCTION(BlueprintCallable, Category = "NNPBR", meta = (WorldContext = "WorldContextObject"))
    static UMaterialInstanceDynamic *LoadNeuralMaterial(UObject *WorldContextObject,
                                                        const FString &ExportDirectory,
                                                        UMaterialInterface *ParentMaterial, FString &Error);
};
