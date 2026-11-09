#pragma once
#include "Commandlets/Commandlet.h"
#include "NNPBRImportCheckCommandlet.generated.h"

// Invoke twice to test serialization across separate editor processes.
UCLASS()
class UNNPBRImportCheckCommandlet : public UCommandlet
{
    GENERATED_BODY()
  public:
    UNNPBRImportCheckCommandlet();
    int32 Main(const FString &Params) override;
};
