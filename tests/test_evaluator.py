from typing import Dict, List, Optional, Union  # Add this line

import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from amazon_reviews.summariser import ProductComparisonEvaluator

@pytest.fixture
def evaluator():
    """Initialize ProductComparisonEvaluator with mock data"""
    mock_model = None  # We don't need actual model for most tests
    mock_tokenizer = None
    meta_category_names = {0: 'Electronics', 1: 'Books'}
    
    return ProductComparisonEvaluator(mock_model, mock_tokenizer, meta_category_names)

@pytest.fixture
def sample_text():
    """Create sample generated text"""
    return """
    Title: Compare Top 3 Wireless Earbuds

    Introduction: Brief analysis of wireless earbuds.

    TOP 3 RECOMMENDED PRODUCTS:
    1. Product A
    Key Strengths: Good battery life
    
    2. Product B
    Key Strengths: Excellent sound
    
    3. Product C
    Key Strengths: Comfortable fit

    WORST PRODUCT WARNING:
    Product D is the worst due to poor quality.

    Final Verdict:
    Choose based on your needs.
    """

class TestProductComparisonEvaluator:
    """Test ProductComparisonEvaluator class"""
    
    def test_initialization(self, evaluator):
        """Test proper initialization"""
        assert evaluator.required_sections == [
            "Title:",
            "Introduction:",
            "TOP 3 RECOMMENDED PRODUCTS:",
            "WORST PRODUCT WARNING:",
            "Final Verdict:"
        ]
        assert evaluator.category_results == {}
        
    def evaluate_section_presence(self, text: str) -> Dict[str, float]:
        """
        Evaluate if all required sections are present in the text
        """
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check which sections are present
        sections_present = [section.lower() in text_lower for section in self.required_sections]
        
        # Calculate metrics
        true_positives = sum(sections_present)
        total_required = len(self.required_sections)
        
        # Calculate precision, recall, and F1 score
        precision = true_positives / total_required if total_required > 0 else 0
        recall = true_positives / total_required if total_required > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': sum(sections_present) / len(sections_present),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def test_evaluate_product_accuracy(self, evaluator):
        """Test product accuracy evaluation"""
        source = "Product: Test Product A\nProduct: Test Product B"
        generated = "Product: Test Product A\nProduct: Test Product C"  
        
        results = evaluator.evaluate_product_accuracy(generated, source)
        
        print(f"Debug - Product accuracy results: {results}")   
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert results['precision'] == 0.5  
        
    def test_extract_products(self, evaluator):
        """Test product name extraction"""
        text = """
        1. Product A Key Strengths: Good
        Product: Product B
        • Product C Key Strengths: Nice
        """
        
        products = evaluator._extract_products(text)
        assert len(products) == 3
        assert 'Product A' in products
        assert 'Product B' in products
        assert 'Product C' in products
        