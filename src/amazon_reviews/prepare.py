"""
Amazon Reviews Data Preparation Module

This module handles the preprocessing of Amazon review data, including:
- Sampling and combining data from multiple categories
- Creating train/validation/test splits
- Text preprocessing and cleaning
- Progress tracking and recovery
- Data validation and verification

The module maintains the same functionality as the original notebook while adding
type safety and better error handling.
"""

import os
import re
import json
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple, Set

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Enable tqdm for pandas operations
tqdm.pandas()

# Type aliases
PathLike = Union[str, Path]
DataSplits = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

# Constants
DEFAULT_SAMPLE_FRACTION = 0.03
DEFAULT_TEST_SIZE = 0.1
DEFAULT_VAL_SIZE = 0.1
DEFAULT_RANDOM_STATE = 42
DATA_PATH = os.getenv("DATA_PATH", "data/amazon_reviews_backup")


def save_sampling_progress(
    processed_files: Set[str], backup_dir: PathLike = None
) -> None:
    """
    Save progress of sampling operation to avoid reprocessing files.

    Args:
        processed_files: Set of filenames that have been processed
        backup_dir: Directory to save progress file (default: current directory)
    """
    progress = {
        "processed_files": list(processed_files),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_path = Path(backup_dir) if backup_dir else Path.cwd()
    with open(save_path / "sampling_progress.json", "w") as f:
        json.dump(progress, f)


def load_sampling_progress(backup_dir: PathLike = None) -> Set[str]:
    """
    Load progress of sampling operation.

    Args:
        backup_dir: Directory containing progress file

    Returns:
        Set of processed filenames
    """
    save_path = Path(backup_dir) if backup_dir else Path.cwd()
    progress_file = save_path / "sampling_progress.json"

    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
        return set(progress.get("processed_files", []))
    return set()


def save_combination_progress(
    processed_files: Set[str], total_rows: int, backup_dir: PathLike = None
) -> None:
    """
    Save progress of file combination operation.

    Args:
        processed_files: Set of filenames that have been combined
        total_rows: Total number of rows processed
        backup_dir: Directory to save progress file
    """
    progress = {
        "processed_files": list(processed_files),
        "total_rows": total_rows,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_path = Path(backup_dir) if backup_dir else Path.cwd()
    with open(save_path / "combination_progress.json", "w") as f:
        json.dump(progress, f)


def load_combination_progress(backup_dir: PathLike = None) -> Tuple[Set[str], int]:
    """
    Load progress of file combination operation.

    Args:
        backup_dir: Directory containing progress file

    Returns:
        Tuple of (processed files set, total rows processed)
    """
    save_path = Path(backup_dir) if backup_dir else Path.cwd()
    progress_file = save_path / "combination_progress.json"

    if progress_file.exists():
        with open(progress_file, "r") as f:
            progress = json.load(f)
        return set(progress.get("processed_files", [])), progress.get("total_rows", 0)
    return set(), 0


def check_existing_splits(data_dir: PathLike = None) -> bool:
    """
    Check if train/val/test splits already exist.

    Args:
        data_dir: Directory to check for split files

    Returns:
        bool: True if all splits exist
    """
    data_dir = Path(data_dir) if data_dir else Path.cwd()
    required_files = ["train.parquet", "val.parquet", "test.parquet"]

    return all((data_dir / file).exists() for file in required_files)


def create_sample_dataset(
    input_dir: PathLike,
    output_dir: PathLike,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> None:
    """
    Create sampled datasets from original parquet files.

    Args:
        input_dir: Directory containing original parquet files
        output_dir: Directory to save sampled files
        sample_fraction: Fraction of data to sample
        random_state: Random seed for reproducibility
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load progress
    processed_files = load_sampling_progress(output_dir)

    # Process each parquet file
    parquet_files = [f for f in input_dir.glob("*.parquet")]
    for file_path in tqdm(parquet_files, desc="Sampling files"):
        if file_path.name in processed_files:
            print(f"Skipping {file_path.name} - already processed")
            continue

        try:
            # Read and sample data
            df = pd.read_parquet(file_path)
            sampled_df = df.sample(frac=sample_fraction, random_state=random_state)

            # Save sampled data
            output_path = output_dir / file_path.name
            sampled_df.to_parquet(output_path)

            # Update progress
            processed_files.add(file_path.name)
            save_sampling_progress(processed_files, output_dir)

            print(f"✓ Sampled {file_path.name}: {len(df):,} → {len(sampled_df):,}")

        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {str(e)}")

        # Clear memory
        gc.collect()


def combine_sampled_files(
    input_dir: PathLike, output_file: PathLike, min_rating: int = 1, max_rating: int = 5
) -> pd.DataFrame:
    """
    Combine sampled parquet files into a single dataset.

    Args:
        input_dir: Directory containing sampled parquet files
        output_file: Path to save combined dataset
        min_rating: Minimum rating to include
        max_rating: Maximum rating to include

    Returns:
        Combined DataFrame
    """
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    # Load progress
    processed_files, total_rows = load_combination_progress(input_dir)

    # Process each sampled file
    dfs = []
    parquet_files = [f for f in input_dir.glob("*.parquet")]

    for file_path in tqdm(parquet_files, desc="Combining files"):
        if file_path.name in processed_files:
            print(f"Skipping {file_path.name} - already processed")
            continue

        try:
            # Read and filter data
            df = pd.read_parquet(file_path)
            df = df[
                (df["rating"].between(min_rating, max_rating))
                & (df["text"].notna())
                & (df["text"].str.len() > 0)
            ]

            dfs.append(df)
            processed_files.add(file_path.name)
            total_rows += len(df)

            # Save progress
            save_combination_progress(processed_files, total_rows, input_dir)
            print(f"✓ Added {file_path.name}: {len(df):,} reviews")

        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {str(e)}")

        # Clear memory
        gc.collect()

    # Combine all dataframes
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_parquet(output_file)
        print(f"\nTotal reviews combined: {len(final_df):,}")
        return final_df

    raise ValueError("No valid data to combine")


def create_data_splits(
    df: pd.DataFrame,
    output_dir: PathLike,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DataSplits:
    """Create train/validation/test splits from the dataset."""
    if df is None or len(df) == 0:
        raise ValueError("Input DataFrame cannot be None or empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # For small datasets, ensure minimum size per class
    min_samples_per_class = df["rating"].value_counts().min()
    if min_samples_per_class < 3:
        raise ValueError(
            f"Each rating class needs at least 3 samples. "
            f"Minimum found: {min_samples_per_class}"
        )

    # Adjust split sizes if necessary
    min_test_size = 3 / len(df)  # Ensure at least 3 samples per class
    test_size = max(test_size, min_test_size)
    val_size = max(val_size, min_test_size)

    # Create splits
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["rating"]
    )

    val_adjusted_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_adjusted_size,
        random_state=random_state,
        stratify=train_val_df["rating"],
    )

    # Save splits
    train_df.to_parquet(output_dir / "train.parquet")
    val_df.to_parquet(output_dir / "val.parquet")
    test_df.to_parquet(output_dir / "test.parquet")

    return train_df, val_df, test_df


def preprocess_text(text: str) -> str:
    """
    Clean and normalize review text.

    Args:
        text: Raw review text

    Returns:
        Cleaned and normalized text

    Raises:
        ValueError: If input is not a string
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text.strip()


def preprocess_dataset(
    df: pd.DataFrame, text_column: str = "text", min_words: int = 3
) -> pd.DataFrame:
    """
    Preprocess the entire dataset.

    Args:
        df: Input DataFrame
        text_column: Name of text column to process
        min_words: Minimum number of words required

    Returns:
        Preprocessed DataFrame
    """
    # Create copy to avoid modifying original
    df = df.copy()

    # Only process text if processed_text column doesn't exist
    if "processed_text" not in df.columns:
        print("Cleaning review texts...")
        df["processed_text"] = df[text_column].progress_apply(preprocess_text)

    # Filter by length
    word_counts = df["processed_text"].str.split().str.len()
    df = df[word_counts >= min_words]

    print(f"\nPreprocessing complete:")
    print(f"Final reviews: {len(df):,}")

    return df


def verify_splits(data_dir: PathLike) -> bool:
    """
    Verify the integrity of data splits.

    Args:
        data_dir: Directory containing split files

    Returns:
        bool: True if verification passes
    """
    data_dir = Path(data_dir)
    split_files = ["train.parquet", "val.parquet", "test.parquet"]

    try:
        splits = {}
        total_rows = 0

        for file in split_files:
            df = pd.read_parquet(data_dir / file)
            splits[file] = len(df)
            total_rows += len(df)

            # Check for required columns
            required_columns = {"text", "processed_text", "rating", "category"}
            if not all(col in df.columns for col in required_columns):
                print(f"✗ Missing required columns in {file}")
                return False

            # Check for empty texts
            if df["processed_text"].isna().any() or (df["processed_text"] == "").any():
                print(f"✗ Found empty texts in {file}")
                return False

        print("\nSplit Verification:")
        for file, count in splits.items():
            print(f"{file}: {count:,} reviews ({count/total_rows:.1%})")

        return True

    except Exception as e:
        print(f"✗ Verification failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Example usage
    try:
        input_dir = Path(DATA_PATH) / "raw"
        output_dir = Path(DATA_PATH) / "processed"

        # Create sample dataset
        create_sample_dataset(input_dir, output_dir / "sampled")

        # Combine samples
        df = combine_sampled_files(
            output_dir / "sampled", output_dir / "combined.parquet"
        )

        # Preprocess
        df = preprocess_dataset(df)

        # Create splits
        create_data_splits(df, output_dir)

        # Verify
        if verify_splits(output_dir):
            print("\n✓ Data preparation complete!")
        else:
            print("\n✗ Data preparation failed verification")

    except Exception as e:
        print(f"\n✗ Error during data preparation: {str(e)}")
