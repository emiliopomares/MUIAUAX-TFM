// Fill out your copyright notice in the Description page of Project Settings.


#include "OverlapDetector.h"

// Sets default values
AOverlapDetector::AOverlapDetector()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;
	// Create collision component
	MyCollisionComponent = CreateDefaultSubobject<UBoxComponent>(TEXT("MyCollisionComponent"));
	//RootComponent = MyCollisionComponent;
	MyCollisionComponent->SetupAttachment(GetRootComponent());

}

// Called when the game starts or when spawned
void AOverlapDetector::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void AOverlapDetector::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void AOverlapDetector::UpdateOverlaps()
{
	// Call UpdateOverlaps
	MyCollisionComponent->UpdateOverlaps();
}


