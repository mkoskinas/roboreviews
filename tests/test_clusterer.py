import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from amazon_reviews.clusterer import CategoryClusterer

@pytest.fixture
def sample_data(tmp_path):
    """Create sample data for testing with all 33 categories"""
    categories = [
        "Books & Literature", "Electronics", "Home & Kitchen", "Fashion", "Beauty",
        "Sports & Outdoors", "Toys & Games", "Automotive", "Health & Personal Care",
        "Tools & Home Improvement", "Pet Supplies", "Office Products", "Grocery",
        "Baby", "Movies & TV", "Music", "Garden & Outdoor", "Arts & Crafts",
        "Video Games", "Industrial & Scientific", "Cell Phones", "Computers",
        "Clothing", "Shoes", "Jewelry", "Watches", "Home & Decor", "Appliances",
        "Camera & Photo", "Musical Instruments", "Software", "Collectibles", "Antiques"
    ]
    
    # Create sample data with all categories
    sample_data = {
        'category': categories * 3,  # Repeat each category 3 times
        'text': [f'sample review text {i}' for i in range(len(categories) * 3)],
        'asin': [f'A{i}' for i in range(len(categories) * 3)]
    }
    df = pd.DataFrame(sample_data)
    
    # Create test data directory
    data_dir = tmp_path / "data/amazon_reviews_backup/processed/sampled"
    data_dir.mkdir(parents=True)
    
    # Save as parquet files
    df.to_parquet(data_dir / "train_processed.parquet")
    df.to_parquet(data_dir / "val_processed.parquet")
    df.to_parquet(data_dir / "test_processed.parquet")
    
    return data_dir

@pytest.fixture
def clusterer(sample_data):
    """Create CategoryClusterer instance with sample data"""
    return CategoryClusterer(data_path=str(sample_data))

def test_initialization(clusterer):
    """Test clusterer initialization"""
    assert clusterer is not None
    assert clusterer.data_path.exists()
    assert clusterer.image_dir.exists()

def test_load_data(clusterer):
    """Test data loading"""
    clusterer.load_data()
    assert clusterer.df is not None
    assert len(clusterer.unique_categories) == 33
    assert isinstance(clusterer.df, pd.DataFrame)

def test_generate_embeddings(clusterer):
    """Test embedding generation"""
    clusterer.load_data()
    clusterer.generate_embeddings()
    assert clusterer.category_embeddings is not None
    assert isinstance(clusterer.category_embeddings, np.ndarray)
    assert len(clusterer.category_embeddings) == 33

def test_reduce_dimensions(clusterer):
    """Test dimension reduction"""
    clusterer.load_data()
    clusterer.generate_embeddings()
    clusterer.reduce_dimensions()
    assert clusterer.umap_embeddings is not None
    assert clusterer.umap_embeddings.shape == (33, 3)

def test_perform_clustering(clusterer):
    """Test clustering"""
    clusterer.load_data()
    clusterer.generate_embeddings()
    clusterer.reduce_dimensions()
    clusterer.perform_clustering()
    assert clusterer.cluster_labels is not None
    assert len(clusterer.cluster_labels) == 33

def test_evaluate_clustering(clusterer):
    """Test clustering evaluation"""
    clusterer.load_data()
    clusterer.generate_embeddings()
    clusterer.reduce_dimensions()
    clusterer.perform_clustering()
    metrics = clusterer.evaluate_clustering()
    assert isinstance(metrics, dict)
    assert 'silhouette' in metrics

def test_save_results(clusterer):
    """Test saving results"""
    clusterer.load_data()
    clusterer.generate_embeddings()
    clusterer.reduce_dimensions()
    clusterer.perform_clustering()
    clusterer.save_results(run_id="test")
    assert (clusterer.data_path / "amazon_reviews_clustered_test.parquet").exists()

def test_visualization_methods(clusterer):
    """Test visualization methods"""
    meta_category_names = {
        0: "Entertainment & General Retail",
        1: "Technology & Automotive",
        2: "Industrial & DIY",
        3: "Health & Beauty",
        4: "Home & Garden"
    }
    
    clusterer.load_data()
    clusterer.generate_embeddings()
    clusterer.reduce_dimensions()
    clusterer.perform_clustering()
    
    # Test all visualization methods
    clusterer.plot_category_clusters(meta_category_names)
    assert (clusterer.image_dir / "amazon_category_clusters.png").exists()
    
    clusterer.plot_treemap(meta_category_names)
    assert (clusterer.image_dir / "category_treemap.html").exists()
    
    clusterer.plot_silhouette_analysis(meta_category_names)
    assert (clusterer.image_dir / "silhouette_analysis.png").exists()
    
    clusterer.plot_probability_distribution(meta_category_names)
    assert (clusterer.image_dir / "probability_distribution.png").exists()
    
    clusterer.plot_cluster_comparison(meta_category_names)
    assert (clusterer.image_dir / "cluster_comparison.png").exists()

def test_full_pipeline(clusterer):
    """Test the complete clustering pipeline"""
    clusterer.run_clustering_pipeline()
    assert (clusterer.data_path / "amazon_reviews_clustered_latest.parquet").exists()
    assert (clusterer.image_dir / "amazon_category_clusters.png").exists()
    assert (clusterer.image_dir / "category_treemap.html").exists()