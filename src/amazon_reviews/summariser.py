"""
Product Summariser module for generating comparative summaries of Amazon products
using the Mistral-7B model with optional fine-tuning capabilities.
"""
import re
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import Dataset
from tqdm.auto import tqdm
import os
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from bert_score import score

# Default prompt template
PROMPT_TEMPLATE = """<s>[INST] Based on customer reviews, write a short article, like a blogpost reviewer would write, about {meta_category} products to help customers choose the best one.
Your response should follow this format:

1. Title: Compare the 3 specific products in an engaging way

2. Introduction: Brief overview of the analysis

3. TOP 3 RECOMMENDED PRODUCTS:
- The top 3 most recommended products and their key differences
- Top complaints for each of those products

4. WORST PRODUCT WARNING:
   - What is the worst product in the category {category} and why you should never buy it

5. Final Verdict:
   - Clear recommendation for each top product
   - Final warning about the worst product

Use this review data:
{review_data}[/INST]"""


class ProductSummariser:
    """
    Main class for generating comparative product summaries using the Mistral model.
    Supports both zero-shot inference and fine-tuning capabilities.
    """

    def __init__(
        self,
        data_path: str,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        prompt_template: str = PROMPT_TEMPLATE,
    ):
        """
        Initialize the summarizer with paths and configurations.

        Args:
            data_path: Path to the data directory containing parquet files
            model_path: Optional path to a saved model
            cache_dir: Optional path to cache directory
            hf_token: Optional Hugging Face token
            prompt_template: Optional custom prompt template
        """
        self.data_path = Path(data_path)
        self.model_path = Path(model_path) if model_path else None
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.data_path / "cached_model"
        )
        self.prompt_template = prompt_template

        # Set up Hugging Face token
        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        # Initialize empty attributes
        self.model = None
        self.tokenizer = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.meta_category_names = None
        self.product_names = {}

        # Create necessary directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configurations
        self._load_configs()

    def _load_configs(self) -> None:
        """Load all configurations"""
        self.generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "num_return_sequences": 1,
        }

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _validate_data(self) -> None:
        """Validate loaded data"""
        required_columns = [
            "meta_category",
            "category",
            "asin",
            "rating",
            "helpful_vote",
            "text",
        ]
        for df_name, df in [
            ("train", self.train_df),
            ("val", self.val_df),
            ("test", self.test_df),
        ]:
            if df is None:
                raise ValueError(f"{df_name} dataset not loaded")
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing columns in {df_name} dataset: {missing_cols}"
                )

    def load_data(self) -> None:
        """Load and prepare the dataset"""
        try:
            print("\nLoading datasets...")
            self.train_df = pd.read_parquet(
                self.data_path / "train_clustered_run_5.parquet"
            )
            self.val_df = pd.read_parquet(
                self.data_path / "val_clustered_run_5.parquet"
            )
            self.test_df = pd.read_parquet(
                self.data_path / "test_clustered_run_5.parquet"
            )

            print(f"✓ Loaded train set: {len(self.train_df):,} reviews")
            print(f"✓ Loaded validation set: {len(self.val_df):,} reviews")
            print(f"✓ Loaded test set: {len(self.test_df):,} reviews")

            # Extract meta-categories
            if "meta_category" in self.train_df.columns:
                self.meta_category_names = (
                    self.train_df.groupby("meta_category")["meta_category"]
                    .first()
                    .to_dict()
                )
                print(f"\n✓ Found {len(self.meta_category_names)} meta-categories")
            else:
                raise KeyError("meta_category column not found")

        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise

    def initialize_model(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        force_download: bool = False,
    ) -> None:
        """
        Initialize or load cached model and tokenizer

        Args:
            model_name: Name of the model to load from Hugging Face
            force_download: If True, ignore cache and download new model
        """
        try:
            print(f"\nInitializing {model_name}...")

            # Check for cached model unless force_download is True
            if not force_download and self._load_from_cache():
                return

            # Download and initialize new model
            self._download_and_cache_model(model_name)

            if torch.cuda.is_available():
                print(
                    f"✓ GPU Memory After Loading: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB"
                )

        except Exception as e:
            print(f"✗ Error initializing model: {str(e)}")
            raise

    def _load_from_cache(self) -> bool:
        """Try to load model from cache"""
        tokenizer_path = self.cache_dir / "tokenizer"
        model_path = self.cache_dir / "model"

        if tokenizer_path.exists() and model_path.exists():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_4bit=True,
                )
                print("✓ Loaded model from cache successfully")
                return True
            except Exception as e:
                print(f"✗ Error loading from cache: {str(e)}")
                return False
        return False

    def _download_and_cache_model(self, model_name: str) -> None:
        """Download and cache the model and tokenizer"""
        try:
            print("Downloading model and tokenizer...")

            # Download and initialize
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=self.bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Save to cache
            tokenizer_path = self.cache_dir / "tokenizer"
            model_path = self.cache_dir / "model"

            print("Saving to cache...")
            self.tokenizer.save_pretrained(tokenizer_path)
            self.model.save_pretrained(model_path)
            print("✓ Model cached successfully")

        except Exception as e:
            print(f"✗ Error downloading model: {str(e)}")
            raise

    def load_product_names(self) -> None:
        """Load product names from cache or build new cache"""
        cache_file = self.cache_dir / "product_names_cache.json"

        try:
            print("\nTrying to load product names from cache...")
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    self.product_names = json.load(f)
                print(f"✓ Loaded {len(self.product_names)} product names from cache")
            else:
                print("Cache not found, building new cache...")
                self._build_product_names_cache()

        except Exception as e:
            print(f"✗ Error loading product names: {str(e)}")
            raise

    def _build_product_names_cache(self) -> None:
        """Build cache of product names from training data"""
        unique_asins = self.train_df["asin"].unique()
        print(f"\nBuilding cache for {len(unique_asins)} products...")

        for i, asin in enumerate(tqdm(unique_asins)):
            try:
                if i % 50 == 0:  # Save progress periodically
                    self._save_product_names_cache()

                if asin in self.product_names:
                    continue

                name = self._fetch_product_name(asin)
                self.product_names[asin] = name

                time.sleep(2)  # Respect rate limits

            except Exception as e:
                print(f"\n✗ Couldn't fetch name for {asin}: {str(e)}")
                self.product_names[asin] = f"Product {asin}"
                continue

        self._save_product_names_cache()
        print(f"✓ Cache built with {len(self.product_names)} products")

    def _fetch_product_name(self, asin: str) -> str:
        """Fetch product name from Amazon"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

        url = f"https://www.amazon.com/dp/{asin}"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find("span", {"id": "productTitle"}) or soup.find(
            "h1", {"id": "title"}
        )
        return title.text.strip() if title and title.text.strip() else f"Product {asin}"

    def _save_product_names_cache(self) -> None:
        """Save product names cache to file"""
        cache_file = self.cache_dir / "product_names_cache.json"
        with open(cache_file, "w") as f:
            json.dump(self.product_names, f)

    def prepare_review_data(self, meta_category: int) -> Dict[str, str]:
        """
        Prepare review data for a specific meta-category
        """
        category_name = self.meta_category_names[meta_category]
        category_reviews = self.train_df[
            self.train_df["meta_category"] == meta_category
        ]

        # Get main subcategory
        subcategories = (
            category_reviews.groupby("category").size().sort_values(ascending=False)
        )
        main_subcategory = subcategories.index[0]

        if main_subcategory == "[UNKNOWN]" and len(subcategories) > 1:
            main_subcategory = subcategories.index[1]

        # Filter for main subcategory
        subcategory_reviews = category_reviews[
            category_reviews["category"] == main_subcategory
        ]

        # Get product stats
        product_stats = subcategory_reviews.groupby("asin").agg(
            {"rating": ["count", "mean"], "helpful_vote": "sum", "category": "first"}
        )

        # Filter for products with minimum reviews
        min_reviews = 10
        product_stats = product_stats[product_stats[("rating", "count")] >= min_reviews]

        # Get top and worst products
        top_products = product_stats.sort_values(
            [("rating", "mean"), ("rating", "count")], ascending=[False, False]
        ).head(3)

        worst_product = product_stats.sort_values(
            [("rating", "mean"), ("helpful_vote", "sum")], ascending=[True, False]
        ).head(1)

        # Prepare review text
        selected_reviews = []

        # Add top products
        selected_reviews.append("\nTOP RATED PRODUCTS:")
        for asin in top_products.index:
            product_reviews = subcategory_reviews[subcategory_reviews["asin"] == asin]
            product_name = f"Product {asin}"  # Simplified for testing

            selected_reviews.append(
                f"\nProduct: {product_name}\n"
                f"Average Rating: {product_stats.loc[asin, ('rating', 'mean')]:.1f}/5\n"
                f"Total Reviews: {len(product_reviews)}\n"
                f"Sample Reviews:\n"
                + "\n".join(
                    product_reviews.sort_values("helpful_vote", ascending=False)
                    .head(2)["text"]
                    .tolist()
                )
            )

        # Add worst product
        selected_reviews.append("\nWORST RATED PRODUCT WARNING:")
        asin = worst_product.index[0]
        product_reviews = subcategory_reviews[subcategory_reviews["asin"] == asin]
        product_name = f"Product {asin}"  # Simplified for testing

        selected_reviews.append(
            f"\nProduct: {product_name}\n"
            f"Average Rating: {product_stats.loc[asin, ('rating', 'mean')]:.1f}/5\n"
            f"Total Reviews: {len(product_reviews)}\n"
            f"Sample Negative Reviews:\n"
            + "\n".join(
                product_reviews.sort_values(
                    ["rating", "helpful_vote"], ascending=[True, False]
                )
                .head(2)["text"]
                .tolist()
            )
        )

        # Return formatted data
        return {
            "review_data": "\n".join(selected_reviews),
            "category": main_subcategory,
        }

    def generate_summary(
        self, meta_category: int, generation_config: Optional[Dict] = None
    ) -> str:
        """
        Generate a product comparison summary for a category

        Args:
            meta_category: ID of the meta-category to summarize
            generation_config: Optional custom generation parameters

        Returns:
            Generated summary text
        """
        # Check if model is initialized
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        if generation_config is None:
            generation_config = self.generation_config

        if meta_category not in self.train_df["meta_category"].unique():
            raise ValueError(f"Category {meta_category} not found in training data")

        # Prepare data and format prompt
        result = self.prepare_review_data(meta_category)
        formatted_prompt = self.prompt_template.format(
            meta_category=self.meta_category_names[meta_category],
            category=result["category"],
            review_data=result["review_data"],
        )

        # Generate summary
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, **generation_config)
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summary

    def prepare_training_data(self, examples: List[Dict]) -> Dataset:
        """
        Prepare dataset for fine-tuning

        Args:
            examples: List of training examples with meta_category and output

        Returns:
            HuggingFace Dataset ready for training
        """
        texts = []

        for example in examples:
            try:
                result = self.prepare_review_data(example["meta_category"])

                if not result["review_data"].strip():
                    print(
                        f"Skipping example for category {example['meta_category']}: No valid review data"
                    )
                    continue

                prompt = self.prompt_template.format(
                    meta_category=self.meta_category_names[example["meta_category"]],
                    category=result["category"],
                    review_data=result["review_data"],
                )

                text = prompt + example["output"] + "</s>"
                texts.append({"text": text})

            except Exception as e:
                print(
                    f"Error processing example for category {example['meta_category']}: {str(e)}"
                )
                continue

        if not texts:
            raise ValueError("No valid training examples were created")

        return Dataset.from_list(texts)

    def fine_tune(
        self,
        training_examples: List[Dict],
        output_dir: str,
        training_args: Optional[Dict] = None,
    ) -> None:
        """
        Fine-tune the model on custom examples

        Args:
            training_examples: List of training examples
            output_dir: Directory to save the fine-tuned model
            training_args: Optional custom training arguments
        """
        # Prepare dataset
        dataset = self.prepare_training_data(training_examples)

        # Tokenize dataset
        def preprocess_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
                padding="max_length",
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            preprocess_function, batched=True, remove_columns=dataset.column_names
        )

        # Set up training arguments
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=output_dir,
                max_steps=200,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                save_strategy="steps",
                save_steps=50,
                save_total_limit=5,
                logging_steps=20,
                learning_rate=1e-5,
                weight_decay=0.01,
                fp16=True,
                max_grad_norm=0.5,
                warmup_steps=50,
                logging_first_step=True,
                report_to=["none"],
            )

        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)

        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            max_seq_length=2048,
            dataset_text_field="text",
            packing=False,
        )

        # Train and save
        trainer.train()
        trainer.save_model(output_dir)

        print(f"\n✓ Model fine-tuned and saved to {output_dir}")


class ProductComparisonEvaluator:
    """Evaluation utilities for product comparison summaries"""

    def __init__(self, model, tokenizer, meta_category_names: Dict[int, str]):
        self.model = model
        self.tokenizer = tokenizer
        self.meta_category_names = meta_category_names

        self.required_sections = [
            "Title:",
            "Introduction:",
            "TOP 3 RECOMMENDED PRODUCTS:",
            "WORST PRODUCT WARNING:",
            "Final Verdict:",
        ]

        self.category_results = {}

    def evaluate_section_presence(self, generated_text: str) -> Dict[str, float]:
        """Evaluate if all required sections are present"""
        sections_found = [
            section
            for section in self.required_sections
            if section.lower() in generated_text.lower()
        ]

        tp = len(sections_found)
        fn = len(self.required_sections) - tp
        fp = len([s for s in generated_text.split("\n") if s.endswith(":")]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = tp / len(self.required_sections)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate_product_accuracy(
        self, generated_text: str, source_data: str
    ) -> Dict[str, float]:
        """Evaluate accuracy of product mentions"""
        source_products = self._extract_products(source_data)
        generated_products = self._extract_products(generated_text)

        tp = len(set(source_products) & set(generated_products))
        fp = len(set(generated_products) - set(source_products))
        fn = len(set(source_products) - set(generated_products))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = tp / len(source_products) if source_products else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _extract_products(self, text: str) -> List[str]:
        """Extract product names from text"""
        patterns = [
            r"\d\.\s*([^\.]+?)(?=\s*Key Strengths:)",
            r"Product:\s*([^\n]+)",
            r"•\s*([^\.]+?)(?=\s*Key Strengths:)",
        ]

        products = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            products.extend([match.group(1).strip() for match in matches])

        return list(set(products))

    def evaluate_bert_score(
        self, generated_text: str, source_text: str
    ) -> Dict[str, float]:
        """Calculate BERTScore metrics"""
        try:
            P, R, F1 = score([generated_text], [source_text], lang="en", verbose=False)
            return {"precision": P.item(), "recall": R.item(), "f1": F1.item()}
        except Exception as e:
            print(f"Warning: BERTScore calculation failed: {str(e)}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def visualize_results(self, results: Dict) -> None:
        """Create visualizations of evaluation results"""
        sns.set_theme(style="whitegrid", palette="deep")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))

        # Plot 1: Structure Metrics
        metrics_df = pd.DataFrame(
            {
                "Structure": [
                    results["structure"][m]
                    for m in ["accuracy", "precision", "recall", "f1"]
                ],
                "Products": [
                    results["products"][m]
                    for m in ["accuracy", "precision", "recall", "f1"]
                ],
            },
            index=["Accuracy", "Precision", "Recall", "F1"],
        )

        metrics_df.plot(kind="bar", ax=axes[0][0])
        axes[0][0].set_title("Structure vs Product Metrics")
        axes[0][0].set_ylim(0, 1)

        # Plot 2: BERTScore
        if "bert_score" in results:
            bert_df = pd.DataFrame(
                {
                    "Score": [
                        results["bert_score"][m] for m in ["precision", "recall", "f1"]
                    ]
                },
                index=["Precision", "Recall", "F1"],
            )

            bert_df.plot(kind="bar", ax=axes[0][1], color="green")
            axes[0][1].set_title("BERTScore Metrics")
            axes[0][1].set_ylim(0, 1)

        # Plot 3: Heatmap
        sns.heatmap(
            metrics_df, annot=True, cmap="YlOrRd", vmin=0, vmax=1, ax=axes[1][0]
        )
        axes[1][0].set_title("Metrics Heatmap")

        # Plot 4: Category Performance
        if self.category_results:
            category_df = pd.DataFrame(self.category_results)
            sns.boxplot(data=category_df.melt(), x="variable", y="value", ax=axes[1][1])
            axes[1][1].set_title("Performance by Category")
            axes[1][1].set_xticklabels(axes[1][1].get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def save_evaluation_results(self, results: Dict, output_dir: str) -> None:
        """Save evaluation results and plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metrics
        with open(output_path / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save plots
        plt.figure()
        self.visualize_results(results)
        plt.savefig(output_path / "evaluation_plots.png")
        plt.close()


def main():
    """Main function for demonstration"""
    # Initialize summarizer
    summarizer = ProductSummariser(
        data_path="path/to/data", cache_dir="path/to/cache", hf_token="your_token_here"
    )

    # Load data and initialize model
    summarizer.load_data()
    summarizer.initialize_model()
    summarizer.load_product_names()

    # Generate a summary
    summary = summarizer.generate_summary(meta_category=0)
    print("\nGenerated Summary:")
    print("=" * 50)
    print(summary)

    # Initialize evaluator
    evaluator = ProductComparisonEvaluator(
        summarizer.model, summarizer.tokenizer, summarizer.meta_category_names
    )

    # Evaluate the summary
    review_data = summarizer.prepare_review_data(0)
    results = {
        "structure": evaluator.evaluate_section_presence(summary),
        "products": evaluator.evaluate_product_accuracy(
            summary, review_data["review_data"]
        ),
        "bert_score": evaluator.evaluate_bert_score(
            summary, review_data["review_data"]
        ),
    }

    # Visualize results
    evaluator.visualize_results(results)


if __name__ == "__main__":
    main()
