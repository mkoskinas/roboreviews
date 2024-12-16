"""
Pipeline script for running the Amazon Reviews analysis system.
Handles data preparation, sentiment classification, category clustering,
and review summarization tasks.
"""

import logging
import argparse
from pathlib import Path
from amazon_reviews.download import (
    setup_data_directory,
    download_amazon_reviews,
    check_storage_usage,
)
from amazon_reviews.prepare import (
    create_sample_dataset,
    combine_sampled_files,
    preprocess_dataset,
    create_data_splits,
)
from amazon_reviews.classifier import SentimentClassifier
from amazon_reviews.clusterer import CategoryClusterer
from amazon_reviews.summariser import ProductSummariser


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def run_data_pipeline(data_dir: Path):
    """Run the data preparation pipeline"""
    try:
        # Download and process reviews
        categories = ["Books", "Electronics"]  # Example categories
        total_reviews = download_amazon_reviews(
            categories=categories,
            needed_columns={"text", "rating", "asin", "title", "helpful_vote"},
        )
        logging.info(f"Successfully processed {total_reviews:,} reviews")

        # Create sample dataset
        create_sample_dataset(
            input_dir=data_dir / "raw", output_dir=data_dir / "processed/sampled"
        )
        logging.info("Created sampled dataset")

        # Combine samples
        df = combine_sampled_files(
            input_dir=data_dir / "processed/sampled",
            output_file=data_dir / "processed/combined.parquet",
        )
        logging.info(f"Combined {len(df):,} reviews")

        # Preprocess
        df = preprocess_dataset(df)
        logging.info("Dataset preprocessing completed")

        # Create splits
        train_df, val_df, test_df = create_data_splits(df, data_dir / "processed")
        logging.info(
            f"Created data splits - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}"
        )

    except Exception as e:
        logging.error(f"Data pipeline failed: {str(e)}")
        raise


def run_classifier_pipeline(data_dir: Path):
    """Run the sentiment classification pipeline"""
    try:
        logging.info("Running sentiment classifier...")
        classifier = SentimentClassifier(data_dir)

        # Set seed for reproducibility
        classifier.set_seed(42)

        # Load and prepare data
        logging.info("Loading and preparing data...")
        train_df, val_df, test_df = classifier.load_data()
        train_loader, val_loader, test_loader = classifier.setup_data_loaders(
            train_df, val_df, test_df
        )

        # Setup model and training
        logging.info("Setting up model and training...")
        classifier.setup_model()
        classifier.setup_training(train_loader, train_df)

        # Train model
        logging.info("Starting training...")
        classifier.train(train_loader, val_loader)

        # Evaluate model
        logging.info("Evaluating model...")
        results = classifier.run_full_evaluation(test_loader)
        logging.info("Sentiment classification completed!")

    except Exception as e:
        logging.error(f"Classifier pipeline failed: {str(e)}")
        raise


def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description="Run Amazon Reviews Pipeline")
    parser.add_argument(
        "--task",
        type=str,
        choices=["data", "classifier", "clusterer", "summariser", "all"],
        default="all",
        help="Pipeline task to run",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="HuggingFace API token for accessing models",
        default=None,
    )
    args = parser.parse_args()

    # Setup
    setup_logging()
    logging.info("Starting Amazon Reviews Pipeline")
    data_dir = setup_data_directory()
    logging.info(f"Data directory set up at: {data_dir}")

    try:
        # Run selected pipeline(s)
        if args.task in ["data", "all"]:
            logging.info("Running data preparation pipeline...")
            run_data_pipeline(data_dir)

        if args.task in ["classifier", "all"]:
            run_classifier_pipeline(data_dir)

        if args.task in ["clusterer", "all"]:
            logging.info("Running category clusterer...")
            clusterer = CategoryClusterer(data_dir)
            clusterer.run_clustering_pipeline()

        if args.task in ["summariser", "all"]:
            logging.info("Running product summariser...")
            summariser = ProductSummariser(
                data_path=data_dir, cache_dir=data_dir / "cache", hf_token=args.hf_token
            )
            summariser.load_data()
            summariser.initialize_model()

            # Generate summaries for each meta-category
            for meta_category in summariser.meta_category_names:
                logging.info(
                    f"Generating summary for {summariser.meta_category_names[meta_category]}..."
                )
                summary = summariser.generate_summary(meta_category)

                # Save summary
                output_file = data_dir / "summaries" / f"{meta_category}_summary.txt"
                output_file.parent.mkdir(exist_ok=True)
                output_file.write_text(summary)
                logging.info(f"Summary saved to: {output_file}")

    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
    finally:
        check_storage_usage()


if __name__ == "__main__":
    main()
