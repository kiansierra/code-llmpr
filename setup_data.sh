#!/bin/bash

function download_and_extract_competiton(){
    local COMPETITION=$1
    kaggle competitions download -c $COMPETITION
    mkdir -p ../input/$COMPETITION
    unzip -qq $COMPETITION.zip -d ../input/$COMPETITION
    echo "Extracted $COMPETITION"
    rm $COMPETITION.zip
}

function download_and_extract_dataset(){
    local USER=$1
    local DATASET=$2
    kaggle datasets download -d $USER/$DATASET
    mkdir -p ../input/$DATASET
    unzip -qq $DATASET.zip -d ../input/$DATASET
    echo "Extracted $DATASET"
    rm $DATASET.zip
}

download_and_extract_competiton "llm-prompt-recovery"
download_and_extract_dataset "nbroad" "gemma-rewrite-nbroad"
download_and_extract_dataset "thedrcat" "llm-prompt-recovery-data"
download_and_extract_dataset "newtonbaba12345" "llm-prompt-recovery-data-gemini-and-gemma"
download_and_extract_dataset "dipamc77" "3000-rewritten-texts-prompt-recovery-challenge"
download_and_extract_dataset "alexxxsem" "llm-recovery"
