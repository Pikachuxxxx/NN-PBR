#pragma once
#include "CoreMinimal.h"
#include "NNPBRInspection.generated.h"

class UTexture2D;

USTRUCT()
struct FNNPBRNeuron
{
    GENERATED_BODY()
    UPROPERTY(VisibleAnywhere, Category = "Decoder")
    FString Output;
    UPROPERTY(VisibleAnywhere, Category = "Decoder")
    float Bias = 0;
    UPROPERTY(VisibleAnywhere, Category = "Decoder")
    TArray<float> InputWeights;
};

UCLASS()
class UNNPBRInspection : public UObject
{
    GENERATED_BODY()
  public:
    UPROPERTY(VisibleAnywhere, Category = "Material")
    FString Material;
    UPROPERTY(VisibleAnywhere, Category = "Decoder", meta = (TitleProperty = "Output"))
    TArray<FNNPBRNeuron> HiddenLayer;
    UPROPERTY(VisibleAnywhere, Category = "Decoder", meta = (TitleProperty = "Output"))
    TArray<FNNPBRNeuron> OutputLayer;
    UPROPERTY(Transient)
    TArray<TObjectPtr<UTexture2D>> Previews;
};
