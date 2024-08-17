// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "BPFileIO.generated.h"

/**
 * 
 */
UCLASS()
class DATASETGENERATOR_API UBPFileIO : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:

    // Writes an array of bytes to a file.
    // Returns true if successful, false otherwise.
    UFUNCTION(BlueprintCallable, Category = "File IO")
    static bool WriteBytesToFile(const FString& FilePath, const TArray<uint8>& Data);

    // Converts float to array of 4 bytes
    UFUNCTION(BlueprintCallable, Category = "File IO")
    static void ConvertFloatToByteArray(const float Value, TArray<uint8>& OutData);
	
};
