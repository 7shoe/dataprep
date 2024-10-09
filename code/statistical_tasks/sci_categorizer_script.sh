#!/bin/bash

# Check if an argument is passed
if [ -z "$1" ]; then
  echo "Please provide an integer for chunk_index."
  exit 1
fi

# Assign the input argument to chunk_index
chunk_index=$1

# Loop through the model_index values (0 to 4)
for model_index in {0..4}
do
  # Run the Python script with the current chunk_index and model_index
  python predict_scientific_category.py --chunk_index $chunk_index --model_index $model_index
done
