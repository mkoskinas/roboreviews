# tests/test_prepare.py
"""Tests for the Amazon Reviews Data Preparation Module"""

import os
import shutil
from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import patch

from amazon_reviews.prepare import (
    preprocess_text,
    preprocess_dataset,
    create_data_splits,
    verify_splits,
    create_sample_dataset,
    combine_sampled_files,
    save_sampling_progress,
    load_sampling_progress,
    save_combination_progress,
    load_combination_progress,
    check_existing_splits
)

# Test fixtures
@pytest.fixture
def mock_data_path(tmp_path):
    """Create a temporary directory for testing"""
    data_path = tmp_path / "test_data"
    data_path.mkdir()
    return data_path

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    data = {
        'text': [
            'Sample review 1', 
            'Sample review 2',
            'Sample review 3',
            'Sample review 4',
            'Sample review 5',
            'Sample review 6',
            'Sample review 7',
            'Sample review 8',
            'Sample review 9'
        ],
        'processed_text': [   
            'sample review one', 
            'sample review two',
            'sample review three',
            'sample review four',
            'sample review five',
            'sample review six',
            'sample review seven',
            'sample review eight',
            'sample review nine'
        ],
        'rating': [
            4, 4, 4,   
            5, 5, 5,   
            3, 3, 3   
        ],
        'category': [
            'Books', 'Electronics', 'Books',
            'Electronics', 'Books', 'Electronics',
            'Books', 'Electronics', 'Books'
        ]
    }
    return pd.DataFrame(data)

# Test functions
def test_preprocess_text():
    """Test text preprocessing function"""
    text = "This is a <b>sample</b> review! With http://example.com link."
    cleaned = preprocess_text(text)
    assert "sample" in cleaned
    assert "http" not in cleaned
    assert "<b>" not in cleaned
    assert cleaned.islower()

def test_preprocess_dataset(sample_df):
    """Test dataset preprocessing"""
    processed_df = preprocess_dataset(sample_df)
    assert 'processed_text' in processed_df.columns
    assert len(processed_df) <= len(sample_df)
    assert not processed_df['processed_text'].str.contains('http').any()

def test_create_data_splits(sample_df, mock_data_path):
    """Test creation of train/val/test splits"""
    train_df, val_df, test_df = create_data_splits(
        sample_df, 
        mock_data_path,
        test_size=0.2,
        val_size=0.2
    )
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_df)
    assert all(f.exists() for f in [
        mock_data_path / 'train.parquet',
        mock_data_path / 'val.parquet',
        mock_data_path / 'test.parquet'
    ])

def test_verify_splits(mock_data_path, sample_df):
    """Test split verification"""
    # Create test splits directly since data is already processed
    train_df, val_df, test_df = create_data_splits(sample_df, mock_data_path)
    
    # Verify splits
    assert verify_splits(mock_data_path)

def test_create_sample_dataset(mock_data_path):
    """Test dataset sampling"""
    # Create test input file
    input_dir = mock_data_path / "input"
    input_dir.mkdir()
    sample_df = pd.DataFrame({
        'text': ['Test'] * 100,
        'rating': [5] * 100
    })
    sample_df.to_parquet(input_dir / "test.parquet")
    
    output_dir = mock_data_path / "output"
    create_sample_dataset(input_dir, output_dir, sample_fraction=0.1)
    
    assert (output_dir / "test.parquet").exists()
    sampled = pd.read_parquet(output_dir / "test.parquet")
    assert len(sampled) < len(sample_df)

def test_progress_tracking(mock_data_path):
    """Test progress saving and loading"""
    # Test sampling progress
    processed_files = {'file1.parquet', 'file2.parquet'}
    save_sampling_progress(processed_files, mock_data_path)
    loaded = load_sampling_progress(mock_data_path)
    assert loaded == processed_files
    
    # Test combination progress
    save_combination_progress(processed_files, 1000, mock_data_path)
    loaded_files, total_rows = load_combination_progress(mock_data_path)
    assert loaded_files == processed_files
    assert total_rows == 1000

def test_check_existing_splits(mock_data_path):
    """Test split existence checking"""
    assert not check_existing_splits(mock_data_path)
    
    # Create dummy split files
    for split in ['train', 'val', 'test']:
        (mock_data_path / f'{split}.parquet').touch()
    
    assert check_existing_splits(mock_data_path)

def test_combine_sampled_files(mock_data_path):
    """Test combining sampled files"""
    input_dir = mock_data_path / "sampled"
    input_dir.mkdir()
    
    # Create test files
    for i in range(2):
        df = pd.DataFrame({
            'text': [f'Review {j}' for j in range(5)],
            'rating': [3] * 5,
            'category': ['Books'] * 5
        })
        df.to_parquet(input_dir / f'category_{i}.parquet')
    
    output_file = mock_data_path / "combined.parquet"
    combined_df = combine_sampled_files(input_dir, output_file)
    
    assert output_file.exists()
    assert len(combined_df) == 10  # 2 files * 5 reviews

def test_invalid_inputs():
    """Test handling of invalid inputs"""
    with pytest.raises(ValueError):
        preprocess_text(None)
    
    with pytest.raises(ValueError):
        create_data_splits(None, "invalid_path")