#!/bin/bash

# Loop through different configurations
for huggingface_model in "bert-base-uncased" "distilbert/distilbert-base-uncased"; do
    for parser in "pymupdf" "nougat"; do
        for max_number_chars in 12800; do
            for learning_rate in 1e-5 1e-4; do
                for lora_r in 2 8 16; do

                    echo "Running experiment with huggingface_model=$huggingface_model, parser=$parser, max_number_chars=$max_number_chars, learning_rate=$learning_rate, lora_r=$lora_r"
                    
                    # Modify the YAML file in place
                    sed -i "s|huggingface_model: .*|huggingface_model: $huggingface_model|" ./configs/journal_cls.yaml
                    sed -i "s|parser: .*|parser: $parser|" ./configs/journal_cls.yaml
                    sed -i "s|max_number_chars: .*|max_number_chars: $max_number_chars|" ./configs/journal_cls.yaml
                    sed -i "s|learning_rate: .*|learning_rate: $learning_rate|" ./configs/journal_cls.yaml
                    sed -i "s|lora_r: .*|lora_r: $lora_r|" ./configs/journal_cls.yaml

                    # Run the classification script with the updated YAML config
                    python run_classification.py -c ./configs/journal_cls.yaml

                done
            done
        done
    done
done
