"""
Amazon Reviews Download Module

This module handles the downloading and processing of Amazon review data from the 
McAuley-Lab/Amazon-Reviews-2023 dataset. It includes functionality for:
- Downloading reviews from multiple categories
- Processing and converting to parquet format
- Backing up to local storage
- Progress tracking and recovery
- Storage management and cleanup
"""

import os
import gc
import sys
import time
import json
import shutil
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Tuple, Dict, Union, List

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, load_from_disk

# Type aliases
PathLike = Union[str, Path]

# Constants
DEFAULT_MIN_DISK_SPACE = 20  #
DEFAULT_COLUMNS = {"text", "rating", "asin", "title", "helpful_vote"}
DATA_PATH = "data/amazon_reviews_backup"

# Global variables
TOTAL_SPACE_FREED = 0


def setup_data_directory() -> str:
    """Set up local data directory."""
    os.makedirs(DATA_PATH, exist_ok=True)
    print("\n" + "=" * 50)
    print("✓ Local backup directory created")
    print("✓ Backup directory:", DATA_PATH)
    print("=" * 50 + "\n")
    return DATA_PATH


def clear_memory() -> None:
    """Clear memory and run garbage collection."""
    gc.collect()
    if "torch" in sys.modules:
        torch.cuda.empty_cache()


def check_disk_space() -> float:
    """
    Check available disk space.

    Returns:
        float: Available space in GB
    """
    total, used, free = shutil.disk_usage("/")
    return free // (2**30)


def check_storage_usage() -> Tuple[float, float]:
    """
    Check current storage usage.

    Returns:
        Tuple[float, float]: (dataset_size, parquet_size) in GB
    """
    dataset_size = 0
    if os.path.exists("amazon_reviews"):
        dataset_size = sum(
            sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for filename in filenames
            )
            for dirpath, dirnames, filenames in os.walk("amazon_reviews")
        ) / (1024**3)

    parquet_size = 0
    if os.path.exists("amazon_reviews_processed"):
        parquet_size = sum(
            os.path.getsize(os.path.join("amazon_reviews_processed", f))
            for f in os.listdir("amazon_reviews_processed")
        ) / (1024**3)

    print("\nStorage Status:")
    print(f"Original datasets: {dataset_size:.1f}GB")
    print(f"Processed parquet files: {parquet_size:.1f}GB")
    print(f"Total space freed: {TOTAL_SPACE_FREED:.1f}GB")

    return dataset_size, parquet_size


def load_progress() -> Dict:
    """
    Load download progress from local storage.

    Returns:
        Dict containing:
        {
            'completed': List[str],  # List of completed categories
            'total_reviews': int     # Total reviews processed
        }
    """
    if not DATA_PATH:
        return {"completed": [], "total_reviews": 0}

    progress_file = f"{DATA_PATH}/download_progress.json"
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return json.load(f)
    return {"completed": [], "total_reviews": 0}


def save_progress(completed_categories: List[str], total_reviews: int) -> None:
    """
    Save download progress to local storage.

    Args:
        completed_categories: List of completed category names
        total_reviews: Total number of reviews processed
    """
    if not DATA_PATH:
        return

    progress_file = f"{DATA_PATH}/download_progress.json"
    with open(progress_file, "w") as f:
        json.dump(
            {
                "completed": completed_categories,
                "total_reviews": total_reviews,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
        )


def check_progress() -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Check current progress across all locations.

    Returns:
        Tuple of Sets containing:
        - backup_files: Categories backed up to local storage
        - local_files: Categories in local processed directory
        - dataset_files: Categories in original dataset directory
        - remaining: Categories still to be processed
    """
    # Load saved progress
    progress = load_progress()
    completed_categories = set(progress["completed"])

    # Check local backup
    backup_files = set()
    if DATA_PATH and os.path.exists(DATA_PATH):
        backup_files = {
            f.replace(".parquet", "")
            for f in os.listdir(DATA_PATH)
            if f.endswith(".parquet") and not f.startswith("temp_")
        }
    # Check local processed files
    local_files = set()
    if os.path.exists("amazon_reviews_processed"):
        local_files = {
            f.replace(".parquet", "")
            for f in os.listdir("amazon_reviews_processed")
            if f.endswith(".parquet")
        }

    # Check original datasets
    dataset_files = set()
    if os.path.exists("amazon_reviews"):
        dataset_files = {
            f.replace(".dataset", "")
            for f in os.listdir("amazon_reviews")
            if f.endswith(".dataset")
        }

    # Get remaining categories
    configs = get_dataset_config_names("McAuley-Lab/Amazon-Reviews-2023")
    all_categories = {
        config.replace("raw_review_", "")
        for config in configs
        if config.startswith("raw_review_")
    }
    remaining = all_categories - backup_files

    # Print status
    print("\nPROGRESS STATUS")
    print("=" * 50)
    print(f"Total categories: {len(all_categories)}")
    print(f"Completed: {len(backup_files)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Total reviews processed: {progress['total_reviews']:,}")

    return backup_files, local_files, dataset_files, remaining


def verify_parquet_integrity(file_path: PathLike) -> bool:
    try:
        df = pd.read_parquet(file_path)
        if len(df) == 0:
            return False

        required_columns = {"text", "rating", "category"}
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns in {file_path}")
            return False

        if df["text"].isna().any() or (df["text"] == "").any():
            print(f"Found empty texts in {file_path}")
            return False

        return True
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
        print(f"Parquet integrity check failed: {str(e)}")
        return False


def backup_to_local(category: str, is_success: bool = True) -> bool:
    try:
        if is_success:
            source = f"amazon_reviews_processed/{category}.parquet"
            temp_dest = f"{DATA_PATH}/temp_{category}.parquet"
            final_dest = f"{DATA_PATH}/{category}.parquet"

            shutil.copy2(source, temp_dest)

            if verify_parquet_integrity(temp_dest):
                if os.path.exists(final_dest):
                    os.remove(final_dest)
                os.rename(temp_dest, final_dest)
                print(f"✓ Backed up {category} to local storage")
            else:
                if os.path.exists(temp_dest):
                    os.remove(temp_dest)
                raise ValueError("Backup verification failed")

        # Always backup the log file
        log_files = [f for f in os.listdir(".") if f.startswith("download_log_")]
        if log_files:
            latest_log = max(log_files)
            shutil.copy2(latest_log, f"{DATA_PATH}/{latest_log}")

    except (IOError, OSError, ValueError) as e:
        print(f"✗ Error backing up to local storage: {str(e)}")
        return False
    return True


def cleanup_category(category: str) -> Dict[str, float]:
    """
    Clean up local files for a category after successful backup.

    Args:
        category: Category name to clean up

    Returns:
        Dict containing cleanup information:
        {
            'dataset_deleted': bool,
            'dataset_size': float (in GB),
            'parquet_deleted': bool,
            'parquet_size': float (in GB)
        }
    """
    global TOTAL_SPACE_FREED
    cleanup_info = {
        "dataset_deleted": False,
        "dataset_size": 0,
        "parquet_deleted": False,
        "parquet_size": 0,
    }

    try:
        # Remove original dataset if it exists
        dataset_path = f"amazon_reviews/{category}.dataset"
        if os.path.exists(dataset_path):
            if os.path.isdir(dataset_path):
                cleanup_info["dataset_size"] = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(dataset_path)
                ) / (1024**3)
                shutil.rmtree(dataset_path)
            else:
                cleanup_info["dataset_size"] = os.path.getsize(dataset_path) / (1024**3)
                os.remove(dataset_path)
            cleanup_info["dataset_deleted"] = True
            TOTAL_SPACE_FREED += cleanup_info["dataset_size"]
            print(f"✓ Removed original dataset for {category}")

        # Remove local parquet if it exists
        parquet_path = f"amazon_reviews_processed/{category}.parquet"
        if os.path.exists(parquet_path):
            cleanup_info["parquet_size"] = os.path.getsize(parquet_path) / (1024**3)
            os.remove(parquet_path)
            cleanup_info["parquet_deleted"] = True
            TOTAL_SPACE_FREED += cleanup_info["parquet_size"]
            print(f"✓ Removed local parquet for {category}")

        return cleanup_info

    except Exception as e:
        print(f"✗ Error during cleanup for {category}: {str(e)}")
        return cleanup_info


def process_category(
    category: str, needed_columns: Set[str], is_new_download: bool = False
) -> int:
    """
    Process a single category of reviews.

    Args:
        category: Category name to process
        needed_columns: Set of column names to keep
        is_new_download: Whether to download new data or use existing

    Returns:
        int: Number of reviews processed, 0 if failed

    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Check disk space
            free_gb = check_disk_space()
            if free_gb < DEFAULT_MIN_DISK_SPACE:
                print("\nWARNING: Low disk space!")
                print(f"Only {free_gb}GB remaining")
                response = input("Continue anyway? (yes/no): ")
                if response.lower() != "yes":
                    return 0

            print(f"\nProcessing {category} (Attempt {attempt + 1}/{max_retries})")

            # Load or download data
            if is_new_download:
                config_name = f"raw_review_{category}"
                dataset = load_dataset(
                    "McAuley-Lab/Amazon-Reviews-2023",
                    config_name,
                    trust_remote_code=True,
                )
                ds = dataset["full"]
            else:
                ds = load_from_disk(f"amazon_reviews/{category}.dataset")["full"]

            # Convert to DataFrame
            df = pd.DataFrame(
                {col: ds[col] for col in needed_columns if col in ds.column_names}
            )
            df["category"] = category

            # Save as parquet
            os.makedirs("amazon_reviews_processed", exist_ok=True)
            output_file = f"amazon_reviews_processed/{category}.parquet"
            df.to_parquet(output_file)

            num_reviews = len(df)
            print(f"Processed {num_reviews:,} reviews")

            # Cleanup
            del ds, df
            clear_memory()

            # Backup to local storage
            if backup_to_local(category):
                cleanup_category(category)
                return num_reviews

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"All attempts failed for {category}")
                backup_to_local(category, is_success=False)
                return 0

    return 0


def download_amazon_reviews(
    categories: Optional[List[str]] = None, needed_columns: Optional[Set[str]] = None
) -> int:
    """
    Download and process Amazon reviews.

    Args:
        categories: List of categories to process (None for all)
        needed_columns: Set of columns to keep (None for default set)

    Returns:
        int: Total number of reviews processed

    Example:
        >>> total_reviews = download_amazon_reviews(
        ...     categories=["Books", "Electronics"],
        ...     needed_columns={'text', 'rating', 'asin', 'title'}
        ... )
    """
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"✓ Created local storage directory: {DATA_PATH}")

    if needed_columns is None:
        needed_columns = DEFAULT_COLUMNS

    # Get all categories if none specified
    if categories is None:
        configs = get_dataset_config_names("McAuley-Lab/Amazon-Reviews-2023")
        categories = [
            config.replace("raw_review_", "")
            for config in configs
            if config.startswith("raw_review_")
        ]

    # Load progress
    progress = load_progress()
    total_reviews = progress["total_reviews"]
    completed = set(progress["completed"])

    try:
        for category in tqdm(categories, desc="Processing categories"):
            if category in completed:
                print(f"Skipping {category} - already completed")
                continue

            num_reviews = process_category(
                category, needed_columns, is_new_download=True
            )
            if num_reviews > 0:
                completed.add(category)
                total_reviews += num_reviews
                save_progress(list(completed), total_reviews)

            clear_memory()

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nProcess failed: {str(e)}")
    finally:
        check_storage_usage()
        print(f"\nTotal space freed: {TOTAL_SPACE_FREED:.1f}GB")

    return total_reviews


def print_status(message: str, is_header: bool = False) -> None:
    """
    Print formatted status messages.

    Args:
        message: Message to print
        is_header: Whether to format as header with separators
    """
    if is_header:
        print("\n" + "=" * 50)
        print(message)
        print("=" * 50)
    else:
        print(message)


def check_data_exists() -> bool:
    """
    Check if raw data files exist.

    Looks for parquet files in:
    - data/amazon_reviews_backup/raw/*.parquet

    Returns:
        bool: True if data exists, False otherwise
    """
    raw_dir = Path(DATA_PATH) / "raw"

    # Check directory exists
    if not raw_dir.exists():
        print(f"✗ Raw data directory not found at: {raw_dir}")
        return False

    # Check for parquet files
    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"✗ No parquet files found in: {raw_dir}")
        return False

    print(f"✓ Found {len(parquet_files)} parquet files in: {raw_dir}")
    for file in parquet_files:
        print(f"  - {file.name}")

    return True


def ensure_data_ready():
    """
    Ensure data is downloaded and ready for processing.

    Expected directory structure:
    amazon_reviews/
    ├── data/
    │   └── amazon_reviews_backup/
    │       ├── raw/                  # Original parquet files
    │       └── processed/            # Will be created during processing
    """
    if not check_data_exists():
        print("\nRaw data not found. Initiating download...")
        download_amazon_reviews(
            categories=["Books", "Electronics"],  # Specific categories or None for all
            needed_columns={"text", "rating", "asin", "title", "helpful_vote"},
        )
    else:
        print("\n✓ Raw data is ready for processing")


if __name__ == "__main__":
    print_status("AMAZON REVIEWS DOWNLOADER", is_header=True)

    try:
        # First ensure data exists
        ensure_data_ready()

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Download failed: {str(e)}")
    finally:
        # Show final storage status
        check_storage_usage()
        print(f"\nTotal space freed: {TOTAL_SPACE_FREED:.1f}GB")
