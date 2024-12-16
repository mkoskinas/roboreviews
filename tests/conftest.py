import pytest
import torch
import pandas as pd
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="session")
def device():
    """Get PyTorch device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def test_data_path():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def mock_product_names():
    """Create mock product names dictionary"""
    return {
        'B001': 'Test Product A',
        'B002': 'Test Product B',
        'B003': 'Test Product C',
        'B004': 'Test Book A',
        'B005': 'Test Book B'
    }