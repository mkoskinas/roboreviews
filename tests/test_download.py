"""
Tests for the Amazon Reviews Download Module
"""

import os
import shutil
from pathlib import Path
from unittest.mock import patch  # Remove unused Mock

# Third party imports
import pandas as pd  # Move to top level
import pytest

# Local imports
from amazon_reviews.download import (
    check_disk_space,
    verify_parquet_integrity,
    load_progress,
    save_progress,
    check_progress,
    setup_data_directory,
    check_data_exists,
    ensure_data_ready    
)

# Test fixtures
@pytest.fixture
def mock_data_path(tmp_path):
    """Create a temporary directory to simulate local storage"""
    data_path = tmp_path / "amazon_reviews_backup"
    data_path.mkdir()
    return data_path

@pytest.fixture
def sample_parquet(tmp_path):
    """Create a sample parquet file for testing"""
    import pandas as pd
    
    # Create sample data
    data = {
        'text': ['Sample review 1', 'Sample review 2'],
        'rating': [4, 5],
        'category': ['Books', 'Books']
    }
    df = pd.DataFrame(data)
    
    # Save as parquet
    file_path = tmp_path / "test.parquet"
    df.to_parquet(file_path)
    return file_path

# Test functions
def test_setup_data_directory(tmp_path):    
    """Test local storage setup"""
    with patch('amazon_reviews.download.DATA_PATH', tmp_path):
        path = setup_data_directory()      
        assert path is not None
        assert os.path.exists(path)

def test_verify_parquet_integrity(sample_parquet):
    """Test parquet file verification"""
    assert verify_parquet_integrity(sample_parquet) == True

def test_load_save_progress(mock_data_path):
    """Test progress tracking functions"""
    with patch('amazon_reviews.download.DATA_PATH', mock_data_path):
        # Test initial load with no file
        progress = load_progress()
        assert progress == {'completed': [], 'total_reviews': 0}
        
        # Test saving and loading progress
        completed = ['Books', 'Electronics']
        total_reviews = 1000
        save_progress(completed, total_reviews)
        
        loaded = load_progress()
        assert loaded['completed'] == completed
        assert loaded['total_reviews'] == total_reviews

def test_check_disk_space():
    """Test disk space checking"""
    space = check_disk_space()
    assert isinstance(space, (int, float))
    assert space > 0

def test_check_progress(mock_data_path):
    """Test progress checking across locations"""
    with patch('amazon_reviews.download.DATA_PATH', mock_data_path):
        # Create some test files
        (mock_data_path / "Books.parquet").touch()
        os.makedirs("amazon_reviews_processed", exist_ok=True)
        Path("amazon_reviews_processed/Electronics.parquet").touch()
        os.makedirs("amazon_reviews", exist_ok=True)
        Path("amazon_reviews/Movies.dataset").touch()
        
        backup_files, local_files, dataset_files, remaining = check_progress()  
        
        assert 'Books' in backup_files  
        assert 'Electronics' in local_files
        assert 'Movies' in dataset_files
        assert len(remaining) > 0

    # Cleanup
    shutil.rmtree("amazon_reviews_processed", ignore_errors=True)
    shutil.rmtree("amazon_reviews", ignore_errors=True)

def test_check_data_exists(tmp_path):
    """Test data existence checking"""
    # Setup test directory
    raw_dir = tmp_path / "raw"  # Simplified path
    raw_dir.mkdir(parents=True)
    
    with patch('amazon_reviews.download.DATA_PATH', str(tmp_path)):
        # Test empty directory
        assert not check_data_exists()
        print(f"\nTesting with empty directory: {raw_dir}")
        
        # Add a parquet file
        (raw_dir / "test.parquet").touch()
        print(f"Added test file: {raw_dir}/test.parquet")
        
        assert check_data_exists()

def test_ensure_data_ready(mocker):
    """Test data readiness check"""
    # Mock check_data_exists
    mock_check = mocker.patch('amazon_reviews.download.check_data_exists')
    mock_download = mocker.patch('amazon_reviews.download.download_amazon_reviews')
    
    # Test when data exists
    mock_check.return_value = True
    ensure_data_ready()
    mock_download.assert_not_called()
    
    # Test when data doesn't exist
    mock_check.return_value = False
    ensure_data_ready()
    mock_download.assert_called_once()