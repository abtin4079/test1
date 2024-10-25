#!/bin/bash

# Define the base image path
BASE_IMAGE_PATH="F:/UNIVERCITY/sharifian/DATASETS_FOR_COLAB/COVID_19_DATASET_INFECTION/colab/test/img"

# Run the command 1000 times
for i in $(seq 1 460)
do
    # Construct the image file name
    IMAGE_FILE="${BASE_IMAGE_PATH}/${i}.png"
    
    # Run the Python script with the constructed file name and image number
    python getting_images.py --image_path "$IMAGE_FILE" --image_number "$i"
    if (( i % 20 == 0 )); then
    echo "${i}.png have been saved"
    fi

done
