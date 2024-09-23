from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from transformers import DataCollatorWithPadding
from datasets import load_from_disk, Dataset

from data_utils import compile_DatasetFrames

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Convert labels to float for regression and ensure they have the correct shape
        labels = labels.float().view_as(logits)  # Reshape labels to match logits

        # Use Mean Squared Error Loss for regression
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

def main():
    # load frames
    df_train, df_test, df_val = compile_DatasetFrames()

    # convert to datasets
    dset_train = Dataset.from_pandas(df_train)
    dset_val = Dataset.from_pandas(df_val)

    # Step 1: Load the tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess the dataset
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)

    # Step 2: Tokenize the dataset
    dset_train_encoded = dset_train.map(preprocess_function, batched=True)
    dset_val_encoded = dset_val.map(preprocess_function, batched=True)

    # Remove unnecessary columns, keep the BLEU score targets
    dset_train_encoded = dset_train_encoded.remove_columns(['text', 'embeddings', 'path', 'abstract', 'firstpage', '__index_level_0__'])
    dset_val_encoded = dset_val_encoded.remove_columns(['text', 'embeddings', 'path', 'abstract', 'firstpage', '__index_level_0__'])

    # Rename columns to 'labels' for each BLEU score (multivariate regression)
    dset_train_encoded = dset_train_encoded.rename_column('bleu_pymupdf', 'labels')
    dset_val_encoded = dset_val_encoded.rename_column('bleu_pymupdf', 'labels')
    
    # Set format for PyTorch
    dset_train_encoded.set_format("torch")
    dset_val_encoded.set_format("torch")

    # Step 3: Load the model (adapt for regression)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # 3 target variables

    # Step 4: Set up LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Multivariate regression is treated similarly to SEQ_CLS
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # Step 5: Apply LoRA to the base model using PEFT
    model = get_peft_model(base_model, lora_config)

    # Data collator (for dynamic padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Step 6: Define metrics for regression (RMSE, MSE, R²)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.tensor(logits)
        labels = torch.tensor(labels)

        # Convert to numpy arrays
        predictions = predictions.detach().numpy()
        labels = labels.detach().numpy()

        # Compute RMSE, MSE, and R² for each output variable
        rmse = np.sqrt(mean_squared_error(labels, predictions, multioutput='raw_values'))
        mse = mean_squared_error(labels, predictions, multioutput='raw_values')
        r2 = r2_score(labels, predictions, multioutput='raw_values')

        return {
            "rmse_pymupdf": rmse[0],
            "mse_pymupdf": mse[0],
            "r2_pymupdf": r2[0],
        }

    # Step 7: Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Set to "epoch"
        save_strategy="epoch",  # Match with evaluation_strategy
        learning_rate=1e-6,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,  # Load the best model
    )

    # Step 8: Initialize Trainer
    trainer = RegressionTrainer(model=model, 
                                args=training_args, 
                                train_dataset=dset_train_encoded,
                                eval_dataset=dset_val_encoded,
                                tokenizer=tokenizer,
                                data_collator=data_collator,
                                compute_metrics=compute_metrics)

    # Step 9: Train the model with LoRA applied
    trainer.train()

# entry point
if __name__ == '__main__':
    main()

