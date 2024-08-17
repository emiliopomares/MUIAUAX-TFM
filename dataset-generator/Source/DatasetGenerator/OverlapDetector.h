// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/BoxComponent.h"
#include "OverlapDetector.generated.h"

UCLASS()
class DATASETGENERATOR_API AOverlapDetector : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AOverlapDetector();

	// Expose this function to Blueprints
	UFUNCTION(BlueprintCallable)
	void UpdateOverlaps();

protected:
	// Collision component
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Collision")
	class UBoxComponent* MyCollisionComponent;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

};
