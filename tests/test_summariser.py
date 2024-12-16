import pytest
import pandas as pd
import torch
from pathlib import Path
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from amazon_reviews.summariser import ProductSummariser, ProductComparisonEvaluator

@pytest.fixture
def mock_data():
    """Create mock training data with sufficient reviews per product"""
    data = {
        'meta_category': [0] * 20 + [1] * 5,  # 20 reviews in category 0
        'category': ['Electronics'] * 20 + ['Books'] * 5,
        'asin': ['B001'] * 10 + ['B002'] * 10 + ['B003'] * 5,  
        'rating': [5.0] * 10 +   
                 [1.0] * 10 +    
                 [3.0] * 5,      
        'helpful_vote': list(range(10, 0, -1)) * 2 + [5] * 5,   
        'text': [f"Review {i+1}" for i in range(25)]   
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def summariser(temp_dir, mock_data):
    """Initialize ProductSummariser with mock data"""
    # Save mock data as parquet files
    for suffix in ['train', 'val', 'test']:
        mock_data.to_parquet(temp_dir / f"{suffix}_clustered_run_5.parquet")
    
    return ProductSummariser(
        data_path=str(temp_dir),
        cache_dir=str(temp_dir / "cache")
    )

class TestProductSummariser:
    """Test ProductSummariser class"""
    
    def test_initialization(self, summariser):
        """Test proper initialization of summariser"""
        assert summariser.data_path.exists()
        assert summariser.cache_dir.exists()
        assert summariser.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert summariser.model is None
        assert summariser.tokenizer is None
    
    def test_load_configs(self, summariser):
        """Test configuration loading"""
        assert summariser.generation_config is not None
        assert summariser.bnb_config is not None
        assert summariser.lora_config is not None
        assert summariser.generation_config["max_new_tokens"] == 1024
        
    def test_load_data(self, summariser):
        """Test data loading functionality"""
        summariser.load_data()
        assert summariser.train_df is not None
        assert summariser.val_df is not None
        assert summariser.test_df is not None
        assert 0 in summariser.meta_category_names
        
    def test_validate_data(self, summariser):
        """Test data validation"""
        summariser.load_data()
        summariser._validate_data()  # Should not raise exception
        
        # Test with missing column
        with pytest.raises(ValueError):
            bad_df = summariser.train_df.drop('meta_category', axis=1)
            summariser.train_df = bad_df
            summariser._validate_data()
            
    @patch('requests.get')
    def test_fetch_product_name(self, mock_get, summariser):
        """Test product name fetching"""
        mock_response = Mock()
        mock_response.text = '<span id="productTitle">Test Product</span>'
        mock_get.return_value = mock_response
        
        name = summariser._fetch_product_name('B001')
        assert name == 'Test Product'
        
    def test_prepare_review_data(self, summariser):
        """Test review data preparation"""
        summariser.load_data()
        result = summariser.prepare_review_data(0)
        
        assert 'review_data' in result
        assert 'category' in result
        assert 'TOP RATED PRODUCTS:' in result['review_data']
        assert 'WORST RATED PRODUCT WARNING:' in result['review_data']