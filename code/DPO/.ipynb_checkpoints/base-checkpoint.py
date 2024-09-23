from pydantic import BaseModel
from typing import Literal, Optional
from pathlib import Path
import uuid
from abc import ABC, abstractmethod
from datasets import Dataset

class BaseConfig(BaseModel):
    """
    Base configuration class containing dataset-specific settings.
    Contains attributes of dataset-specific attributes that should be shared 
    independent of the statistical task (classification, regression, etc.).
    """
    
    p_embeddings_root_dir: Path = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/embeddings/emb_by_model')
    p_response_csv_path: Path = Path('/lus/eagle/projects/argonne_tpc/siebenschuh/aurora_gpt/database/parser_metrics_without_text_output.csv')
    predefined_split: bool = True
    p_split_yaml_path: Path = Path('/home/siebenschuh/Projects/dataprep/code/DPO/meta_split/pymupdf.yaml')
    frequency_train: float = 0.85
    seed_val: int = 18
    normalized: bool = False

    @property
    def unique_id(self) -> str:
        """Get the unique identifier for the config."""
        if not hasattr(self, '_unique_id'):
            self._unique_id = str(uuid.uuid4())
        return self._unique_id


class BasePredictionModel(ABC):
    """Base model class from which all prediction models inherit."""
    
    @abstractmethod
    def train(self, train_data: Dataset):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def evaluate(self, eval_data: Dataset):
        """Evaluate the model on the validation data."""
        pass

    @abstractmethod
    def compute_metrics(self):
        """Get the evaluation metrics for the model."""
        pass

    def restrict_text(self, dataset: Dataset, max_number_chars: int) -> Dataset:
        """
        Restrict the text column in the dataset to a maximum number of (first) characters.
        
        Parameters:
        - dataset (Dataset): The Hugging Face Dataset object that contains the text data.
        - max_number_chars (int): The maximum number of characters to keep for each text entry.
        
        Returns:
        Dataset: The dataset with the text column truncated to the specified maximum number of characters.
        """
        def truncate_text(examples):
            examples['text'] = [text[:max_number_chars] for text in examples['text']]
            return examples
        
        return dataset.map(truncate_text, batched=True)