"""
Amazon Reviews Sentiment Classifier
Handles sentiment classification of Amazon product reviews using DistilRoBERTa
"""

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    AdamW,
)
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class StreamingDataset(Dataset):
    """Custom dataset for streaming Amazon reviews"""

    def __init__(self, df, tokenizer, max_length, cache_size):
        self.data = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            row["processed_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
        label = sentiment_map[row["sentiment"]]

        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

        if len(self.cache) < self.cache_size:
            self.cache[idx] = item

        return item


class SentimentClassifier:
    """Handles training and evaluation of the sentiment classifier"""

    def __init__(self, data_path: str, config: dict = None):
        """Initialize the classifier with data path and optional config

        Args:
            data_path (str): Path to the data directory
            config (dict, optional): Configuration overrides. Defaults to None.
        """
        self.data_path = Path(data_path)
        self.config = {
            "MODEL_NAME": "distilroberta-base",
            "MAX_LENGTH": 256,
            "BATCH_SIZE": 64,
            "CACHE_SIZE": 50000,
            "NUM_EPOCHS": 2,
            "LEARNING_RATE": 2e-5,
            "WARMUP_RATIO": 0.1,
            "WEIGHT_DECAY": 0.01,
            "GRAD_ACCUMULATION": 2,
            "ADAM_EPSILON": 1e-8,
            "MAX_GRAD_NORM": 1.0,
            "DROPOUT": 0.1,
            "LABEL_SMOOTHING": 0.1,
            "PATIENCE": 3,
            "MIN_DELTA": 0.001,
        }
        if config:
            self.config.update(config)

    def set_seed(self, seed=42):
        """Set seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_data(self):
        """Load and prepare balanced datasets"""
        print("Loading full dataset...")
        try:
            train_df = pd.read_parquet(self.data_path / "train_processed.parquet")
            val_df = pd.read_parquet(self.data_path / "val_processed.parquet")
            test_df = pd.read_parquet(self.data_path / "test_processed.parquet")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise

        print("\nOriginal class distributions:")
        for name, df in [
            ("Train", train_df),
            ("Validation", val_df),
            ("Test", test_df),
        ]:
            print(f"\n{name} set:")
            print(df["sentiment"].value_counts())

        # Create balanced datasets using all available data
        balanced_train_df, balanced_val_df, balanced_test_df = (
            self._create_balanced_datasets(
                pd.concat(
                    [train_df, val_df, test_df], ignore_index=True
                )  # Combine all data
            )
        )
        return balanced_train_df, balanced_val_df, balanced_test_df

    def _create_balanced_datasets(
        self,
        train_df,
        samples_per_class=200_000,
        val_size=50_000,
        test_size=50_000,
        random_state=42,
    ):
        """Create balanced datasets for training, validation, and testing"""
        balanced_train, balanced_val, balanced_test = [], [], []

        for sentiment in ["Negative", "Neutral", "Positive"]:
            print(f"\nProcessing {sentiment} class...")
            # Get all samples for this class
            class_samples = train_df[train_df["sentiment"] == sentiment].copy()
            total_samples = len(class_samples)
            print(f"Total {sentiment} samples available: {total_samples:,}")

            # Shuffle the samples
            class_samples = shuffle(class_samples, random_state=random_state)

            # Split the data
            train_samples = class_samples.iloc[:samples_per_class]
            val_samples = class_samples.iloc[
                samples_per_class : samples_per_class + val_size // 3
            ]
            test_samples = class_samples.iloc[
                samples_per_class
                + val_size // 3 : samples_per_class
                + val_size // 3
                + test_size // 3
            ]

            # Append to respective lists
            balanced_train.append(train_samples)
            balanced_val.append(val_samples)
            balanced_test.append(test_samples)

            print(f"Selected {len(train_samples):,} for training")
            print(f"Selected {len(val_samples):,} for validation")
            print(f"Selected {len(test_samples):,} for testing")

        # Combine and shuffle the final datasets
        train_df = shuffle(
            pd.concat(balanced_train, ignore_index=True), random_state=random_state
        )
        val_df = shuffle(
            pd.concat(balanced_val, ignore_index=True), random_state=random_state
        )
        test_df = shuffle(
            pd.concat(balanced_test, ignore_index=True), random_state=random_state
        )

        # Print final distributions
        print("\nFinal class distributions:")
        print("\nTraining set:")
        print(train_df["sentiment"].value_counts())
        print("\nValidation set:")
        print(val_df["sentiment"].value_counts())
        print("\nTest set:")
        print(test_df["sentiment"].value_counts())

        # Save balanced datasets
        print("\nSaving balanced datasets...")
        train_df.to_parquet(self.data_path / "balanced_train.parquet")
        val_df.to_parquet(self.data_path / "balanced_val.parquet")
        test_df.to_parquet(self.data_path / "balanced_test.parquet")

        return train_df, val_df, test_df

    def setup_data_loaders(self, train_df, val_df, test_df):
        """Create datasets and dataloaders"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["MODEL_NAME"])

        train_dataset = StreamingDataset(
            train_df,
            self.tokenizer,
            self.config["MAX_LENGTH"],
            self.config["CACHE_SIZE"],
        )
        val_dataset = StreamingDataset(
            val_df, self.tokenizer, self.config["MAX_LENGTH"], self.config["CACHE_SIZE"]
        )
        test_dataset = StreamingDataset(
            test_df,
            self.tokenizer,
            self.config["MAX_LENGTH"],
            self.config["CACHE_SIZE"],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["BATCH_SIZE"],
            num_workers=0,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config["BATCH_SIZE"],
            num_workers=0,
            pin_memory=True,
        )

        print("\nDatasets created successfully!")
        print(f"Training samples: {len(train_df):,}")
        print(f"Validation samples: {len(val_df):,}")
        print(f"Test samples: {len(test_df):,}")

        return train_loader, val_loader, test_loader

    def setup_model(self):
        """Initialize model, optimizer, scheduler, and criterion"""
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create checkpoint directory
        self.checkpoint_dir = self.data_path / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["MODEL_NAME"],
            num_labels=3,
            hidden_dropout_prob=self.config["DROPOUT"],
            attention_probs_dropout_prob=self.config["DROPOUT"],
        ).to(self.device)

        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()

        # Setup optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config["WEIGHT_DECAY"],
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config["LEARNING_RATE"],
            eps=self.config["ADAM_EPSILON"],
        )

    def setup_training(self, train_loader, train_df):
        """Setup scheduler and loss criterion"""
        # Learning rate scheduler
        num_training_steps = len(train_loader) * self.config["NUM_EPOCHS"]
        warmup_steps = int(num_training_steps * self.config["WARMUP_RATIO"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Calculate class weights
        print("\nCalculating class weights...")
        train_labels_array = (
            train_df["sentiment"]
            .map({"Negative": 0, "Neutral": 1, "Positive": 2})
            .values
        )

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels_array),
            y=train_labels_array,
        )
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=self.config["LABEL_SMOOTHING"]
        )
        print("✓ Class weights calculated:", class_weights.cpu().numpy())

    def save_checkpoint(self, epoch, train_loss, val_loss, val_accuracy, val_f1):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "config": self.config,
        }
        torch.save(checkpoint, self.checkpoint_dir / "checkpoint.pt")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"]

    def train(self, train_loader, val_loader):
        """Training loop with validation"""
        best_val_f1 = 0
        early_stopping_counter = 0
        best_val_loss = float("inf")

        print("Starting training...")

        for epoch in range(self.config["NUM_EPOCHS"]):
            # Training phase
            self.model.train()
            total_loss = 0
            progress = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}"
            )

            self.optimizer.zero_grad()
            for i, batch in enumerate(progress):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = (
                    self.criterion(outputs.logits, labels)
                    / self.config["GRAD_ACCUMULATION"]
                )
                loss.backward()

                if (i + 1) % self.config["GRAD_ACCUMULATION"] == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["MAX_GRAD_NORM"]
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss = total_loss / len(train_loader)

            # Validation phase
            val_loss, val_accuracy, val_f1 = self.validate(val_loader)

            # Print metrics
            print(f"\nEpoch {epoch+1}/{self.config['NUM_EPOCHS']}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"Val F1: {val_f1:.4f}")

            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss, val_accuracy, val_f1)

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(
                    self.model.state_dict(), self.checkpoint_dir / "best_model.pt"
                )
                print("✓ Saved new best model!")

            # Early stopping
            if val_loss < best_val_loss - self.config["MIN_DELTA"]:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.config["PATIENCE"]:
                    print("\nEarly stopping triggered!")
                    break

    def validate(self, val_loader):
        """Run validation and return metrics"""
        self.model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Get model outputs (without passing labels)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Calculate loss with our criterion
                loss = self.criterion(outputs.logits, labels)
                val_loss += loss.item()

                # Get predictions from logits (unchanged)
                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_f1 = precision_recall_fscore_support(
            val_labels, val_preds, average="weighted"
        )[2]

        return val_loss, val_accuracy, val_f1

    def evaluate(self, test_loader):
        """Evaluate model on test set and return metrics"""
        self.model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0

        print("Evaluating model...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Get model outputs (without passing labels)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Calculate loss with our criterion
                loss = self.criterion(outputs.logits, labels)
                test_loss += loss.item()

                # Get predictions from logits (unchanged)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        test_loss = test_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )

        # Store results
        results = {
            "loss": test_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": all_preds,
            "labels": all_labels,
        }

        return results

    def print_metrics(self, results):
        """Print evaluation metrics"""
        print("\nTest Metrics:")
        print(f"Loss: {results['loss']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")

        # Per-class metrics
        print("\nPer-class Metrics:")
        precision, recall, f1, support = precision_recall_fscore_support(
            results["labels"], results["predictions"]
        )

        for i, label in enumerate(["Negative", "Neutral", "Positive"]):
            print(f"\n{label}:")
            print(f"Precision: {precision[i]:.4f}")
            print(f"Recall: {recall[i]:.4f}")
            print(f"F1: {f1[i]:.4f}")
            print(f"Support: {support[i]}")

    def plot_confusion_matrix(self, results, save_path=None):
        """Plot and optionally save confusion matrix"""
        cm = confusion_matrix(results["labels"], results["predictions"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Neutral", "Positive"],
            yticklabels=["Negative", "Neutral", "Positive"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def run_full_evaluation(self, test_loader, save_plots=True):
        """Run complete evaluation with metrics and visualizations"""
        results = self.evaluate(test_loader)
        self.print_metrics(results)

        if save_plots:
            # Use the correct static/images directory from project root
            image_dir = Path("static/images")

            # Save confusion matrix
            self.plot_confusion_matrix(
                results, save_path=image_dir / "confusion_matrix.png"
            )
        else:
            self.plot_confusion_matrix(results)

        return results


def main():
    """Example usage of the SentimentClassifier"""
    classifier = SentimentClassifier(
        data_path="data/amazon_reviews_backup/processed/sampled"
    )

    # Set seed for reproducibility
    classifier.set_seed(42)

    # Load and prepare data
    train_df, val_df, test_df = classifier.load_data()
    train_loader, val_loader, test_loader = classifier.setup_data_loaders(
        train_df, val_df, test_df
    )

    # Setup model and training
    classifier.setup_model()
    classifier.setup_training(train_loader, train_df)

    # Train model
    classifier.train(train_loader, val_loader)

    # Evaluate model
    results = classifier.run_full_evaluation(test_loader)

    return results


if __name__ == "__main__":
    main()
