// Fill out your copyright notice in the Description page of Project Settings.


#include "BPFileIO.h"

#include "CoreMinimal.h"
#include "HAL/UnrealMemory.h"
#include "Containers/Array.h"
#include "Serialization/BufferArchive.h"

bool UBPFileIO::WriteBytesToFile(const FString& FilePath, const TArray<uint8>& Data)
{
    // Create a binary writer
    IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
    FBufferArchive BufferArchive;

    // Serialize the data into the buffer archive
    BufferArchive.Serialize((void*)Data.GetData(), Data.Num());

    // Write the data to the file
    if (FFileHelper::SaveArrayToFile(BufferArchive, *FilePath))
    {
        return true;
    }
    else
    {
        // Handle error
        UE_LOG(LogTemp, Error, TEXT("Failed to write bytes to file: %s"), *FilePath);
        return false;
    }
}

void UBPFileIO::ConvertFloatToByteArray(const float Value, TArray<uint8>& OutData)
{
    // Ensure that the output array is empty before starting
    OutData.Empty();

    // Resize the output array to hold 4 bytes
    OutData.SetNumUninitialized(4);

    // Copy the memory contents of the float into the output array
    FMemory::Memcpy((void *)OutData.GetData(), (const void *)&Value, sizeof(float));
}