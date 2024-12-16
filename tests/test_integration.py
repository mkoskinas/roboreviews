import pytest
import torch
from pathlib import Path
import tempfile
import shutil
import pandas as pd
from unittest.mock import Mock, patch

from amazon_reviews.summariser import ProductSummariser, ProductComparisonEvaluator

@pytest.fixture(scope="class")
def mock_data():
    """Create mock training data"""
    data = {
        'meta_category': [0] * 20 + [1] * 5,
        'category': ['Electronics'] * 20 + ['Books'] * 5,
        'asin': ['B001'] * 10 + ['B002'] * 10 + ['B003'] * 5,
        'rating': [5.0] * 10 + [1.0] * 10 + [3.0] * 5,
        'helpful_vote': list(range(10, 0, -1)) * 2 + [5] * 5,
        'text': [f"Review {i+1}" for i in range(25)]
    }
    return pd.DataFrame(data)

class TestIntegration:
    """Integration tests for the complete workflow"""
    
    @pytest.fixture(scope="class")
    def setup(self, mock_data):
        """Set up test environment"""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Save mock data as parquet files
        for suffix in ['train', 'val', 'test']:
            mock_data.to_parquet(temp_path / f"{suffix}_clustered_run_5.parquet")
        
        # Create mock model and tokenizer
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_model.to.return_value = mock_model  # Handle device movement
        
        # Create a proper mock tokenizer that returns a dictionary
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "This is a mock summary"
        
        # Create a mock tensor dictionary with to() method
        class MockTensorDict(dict):
            def to(self, device):
                return self
        
        # Set up the tokenizer call return value correctly
        def tokenizer_call(*args, **kwargs):
            return MockTensorDict({
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            })
        mock_tokenizer.side_effect = tokenizer_call
        
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model_init, \
             patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_init:
            
            mock_model_init.return_value = mock_model
            mock_tokenizer_init.return_value = mock_tokenizer
            
            summariser = ProductSummariser(
                data_path=str(temp_path),
                cache_dir=str(temp_path / "cache"),
                hf_token="mock_token"
            )
            
            # Mock the device to avoid CUDA/CPU issues
            summariser.device = 'cpu'
            
            yield summariser
        
        shutil.rmtree(temp_dir)
    
    def test_complete_workflow(self, setup):
        """Test the complete workflow from data loading to evaluation"""
        summariser = setup
        
        # 1. Load data
        summariser.load_data()
        assert summariser.train_df is not None
        assert len(summariser.train_df) > 0
        
        # 2. Initialize model
        summariser.initialize_model()
        assert summariser.model is not None
        assert summariser.tokenizer is not None
        
        # 3. Generate summary
        summary = summariser.generate_summary(meta_category=0)
        assert isinstance(summary, str)
        assert len(summary) > 0
        
        # 4. Evaluate results
        evaluator = ProductComparisonEvaluator(
            summariser.model,
            summariser.tokenizer,
            summariser.meta_category_names
        )
        
        review_data = summariser.prepare_review_data(0)
        results = {
            'structure': evaluator.evaluate_section_presence(summary),
            'products': evaluator.evaluate_product_accuracy(summary, review_data['review_data'])
        }
        
        assert all(metric in results for metric in ['structure', 'products'])
        assert all(0 <= score <= 1 for metric in results.values() 
                  for score in metric.values())
    
    def test_invalid_category(self, setup):
        """Test handling of invalid category"""
        summariser = setup
        summariser.load_data()
        
        with pytest.raises(ValueError, match="Category .* not found in training data"):
            summariser.generate_summary(meta_category=999)  # Invalid category
    
    def test_missing_model(self, setup):
        """Test handling of uninitialized model"""
        summariser = setup
        summariser.load_data()
        
        # Reset model and tokenizer without initialization
        summariser.model = None
        summariser.tokenizer = None
        
        with pytest.raises(RuntimeError, match="Model not initialized"):
            # This should fail before trying to generate
            summariser.generate_summary(meta_category=0)