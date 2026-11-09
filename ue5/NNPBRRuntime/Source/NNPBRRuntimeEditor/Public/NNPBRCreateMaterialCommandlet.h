#pragma once
#include "Commandlets/Commandlet.h"
#include "NNPBRCreateMaterialCommandlet.generated.h"

UCLASS()
class UNNPBRCreateMaterialCommandlet : public UCommandlet
{
    GENERATED_BODY()
  public:
    UNNPBRCreateMaterialCommandlet();
    virtual int32 Main(const FString &Params) override;
};
