from __future__ import annotations

from typing import Any, Literal
from pathlib import Path
import torch
import time

from base import BaseConfig, BasePredictionModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model

from data_utils import compile_DatasetFrames

__all__ = [
    'ClassifierConfig',
    'ClassifierModel',
]

class ClassifierConfig(BaseConfig):
    """Configuration for the text classification model."""

    # Name of the HF model used for classification based on textual input
    huggingface_model: str = "bert-base-uncased"
    # Exogenous variable (`X`) on which the classification is based on
    exo_var: str = "text"
    # Endogenous variable (`y`) that is to be predicted
    endo_var: str = "journal_cls"
    # Number of class labels
    num_labels: int = 4
    # Parser from which the text is predicted
    parser: str = ''
    # Maximum number of characters to use in the 'text' feature
    max_number_chars: int = -1
    # Flag indicating if inference is conducted onto the test set
    test_flag: bool = False
    # (Initial) learning rate for training
    learning_rate: float = 1e-5
    # Batch size for training
    batch_size: int = 32
    # Number of epochs
    num_epochs: int = 3
    # Weight decay for optimizer
    weight_decay: float = 0.01
    # Rank of the update matrix in the Low-rank Adaptation (LoRA) scheme (smaller -> fewer trainable paras)
    lora_r: int = 8
    # LoRA scaling factor
    lora_alpha: float = 0.0
    # Flag indicating if rank-stabilized LoRA is used
    use_rslora: bool = False
    # Dropout probability for LoRA layers
    lora_dropout: float = 0.1
    # Number of steps after which results are logged
    logging_steps: int = 10
    # Indicates if the best model is to be loaded
    load_best_model_at_end: bool = False
    # Path to which results are stored
    results_dir: Path = Path("/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/DPO/cls_page_results")
    # Path to which logged output is stored
    logs_dir: Path = Path("/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/DPO/cls_logs")

class ClassifierModel(BasePredictionModel):
    """Text classification model based on HF's sequence classification task"""

    def __init__(self, config: ClassifierConfig):
        """Initialize the classifier with the provided configuration."""
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.huggingface_model)
        self.model = self._load_model()

    def _load_model(self):
        """Load and return the model with LoRA applied."""
    
        # k-class classifier
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.huggingface_model, 
            num_labels=self.config.num_labels,
            trust_remote_code=True
        )
    
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Seq. classification task
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            use_rslora=self.config.use_rslora,
        )
    
        return get_peft_model(base_model, lora_config)


    def preprocess_function(self, examples):
        """Preprocess the dataset for tokenization."""
    
        # Truncate after the first `max_number_chars` number of characters
        text = examples[self.config.exo_var][:self.config.max_number_chars]  # Truncate text based on max_number_chars
        tokenized_inputs = self.tokenizer(text, truncation=True)  # Removed 'clean_up_tokenization_spaces' argument
    
        # Ensure that the labels are added
        tokenized_inputs["labels"] = examples[self.config.endo_var]
    
        return tokenized_inputs

    def preprocess_and_filter_data(self, dset: Dataset) -> Dataset:
        """Preprocess the dataset and filter invalid labels."""
        
        # Preprocess the text (truncate based on max_number_chars)
        dset_processed = dset.map(self.preprocess_function, batched=True)
        
        # Read-out number of classes
        num_labels = int(self.config.num_labels)
        
        # Filter invalid labels after preprocessing
        dset_filtered = dset_processed.filter(lambda example: example['labels'] >= 0 and example['labels'] < num_labels)
        
        # Set format for PyTorch tensors
        dset_filtered.set_format("torch")
        
        return dset_filtered

    def train(self, train_data: Dataset, eval_data: Dataset = None):
        """Train the classification model."""
        # - train
        train_data_encoded = self.preprocess_and_filter_data(train_data)
        # - val
        eval_data_encoded = self.preprocess_and_filter_data(eval_data)

        # unqiue name for run
        run_name = f"{self.config.huggingface_model}-run-{time.strftime('%Y%m%d-%H%M%S')}"
        
        # training arguments
        training_args = TrainingArguments(
            output_dir=Path(self.config.results_dir) / run_name,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.logs_dir,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=self.config.load_best_model_at_end,
            run_name=run_name
        )

        # Trainer setup
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data_encoded,
            eval_dataset=eval_data_encoded,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        # run training
        trainer.train()

    def evaluate(self, eval_data: Dataset) -> dict:
        """Evaluate the classification model."""
        # preprocess/filter
        dset_eval_encoded = self.preprocess_and_filter_data(eval_data)

        # setup trainer
        trainer = Trainer(
            model=self.model,
            eval_dataset=dset_eval_encoded,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        # conduct evaluation
        results = trainer.evaluate()
        print(f"Validation Results: {results}")
        
        return results

    def compute_metrics(self, eval_pred) -> dict:
        """Compute accuracy for classification."""
        logits, labels = eval_pred
        predictions = torch.argmax(torch.from_numpy(logits), dim=-1)
        labels = torch.tensor(labels)

        # Compute accuracy
        acc = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        return {"accuracy": acc}