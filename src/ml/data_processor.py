"""Data loading and preprocessing for ticket classification."""

import pandas as pd
from datasets import load_dataset, Dataset
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
from typing import Tuple
import torch

from src.config import settings


LABEL_MAP = {
    "billing": 0,
    "technical": 1,
    "account": 2,
    "feature_request": 3,
    "general": 4,
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


class TicketDataProcessor:
    """Handles all data loading, preprocessing, and tokenization."""

    def __init__(self, model_name: str = None, max_length: int = None):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name or settings.model_name)
        self.max_length = max_length or settings.max_token_length

    def load_and_prepare(self, data_path: str = None) -> Tuple[Dataset, Dataset, Dataset]:
        """Load data, split into train/val/test, tokenize.

        If no data_path provided, uses a public dataset and maps it
        to our label schema.
        """
        if data_path:
            df = pd.read_csv(data_path)
        else:
            # Use a public dataset as starting point
            # We'll remap categories to our support ticket labels
            dataset = load_dataset("ag_news", split="train[:10000]")
            df = pd.DataFrame(dataset)
            # Map ag_news labels (0-3) to our labels
            ag_to_ticket = {0: "general", 1: "technical", 2: "billing", 3: "feature_request"}
            df["label_name"] = df["label"].map(ag_to_ticket)
            df["label"] = df["label_name"].map(LABEL_MAP)
            df = df.rename(columns={"text": "ticket_text"})

        # Split: 70/15/15
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

        # Tokenize
        train_dataset = self._tokenize_df(train_df)
        val_dataset = self._tokenize_df(val_df)
        test_dataset = self._tokenize_df(test_df)

        return train_dataset, val_dataset, test_dataset

    def _tokenize_df(self, df: pd.DataFrame) -> Dataset:
        """Tokenize a DataFrame into a HuggingFace Dataset."""
        dataset = Dataset.from_pandas(df[["ticket_text", "label"]].reset_index(drop=True))
        dataset = dataset.map(
            lambda batch: self.tokenizer(
                batch["ticket_text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            ),
            batched=True,
        )
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        return dataset

    def tokenize_single(self, text: str) -> dict:
        """Tokenize a single input for inference."""
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoded
