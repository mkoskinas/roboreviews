"""
Amazon Category Clusterer
Handles clustering of Amazon product categories using HDBSCAN and UMAP
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import gc
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)
from typing import Dict


class CategoryClusterer:
    """Handles clustering of Amazon product categories"""

    def __init__(self, data_path: str = "data/amazon_reviews_backup/processed/sampled"):
        self.data_path = Path(data_path)
        self.config = {
            "MODEL_NAME": "sentence-transformers/all-mpnet-base-v2",
            "BATCH_SIZE": 32,
            "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        # Create necessary directories
        self.image_dir = Path("static/images")
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # Initialize instance variables
        self.df = None
        self.unique_categories = None
        self.model = None
        self.category_embeddings = None
        self.umap_reducer = None
        self.umap_embeddings = None
        self.clusterer = None
        self.cluster_labels = None

        print(f"Using device: {self.config['DEVICE']}")

    def load_data(self) -> None:
        """Load and combine datasets"""
        print("\nLoading datasets...")
        try:
            train_df = pd.read_parquet(self.data_path / "train_processed.parquet")
            val_df = pd.read_parquet(self.data_path / "val_processed.parquet")
            test_df = pd.read_parquet(self.data_path / "test_processed.parquet")

            # Combine for category analysis
            self.df = pd.concat([train_df, val_df, test_df])
            self.unique_categories = self.df["category"].unique().tolist()

            print(f"✓ Loaded {len(self.df):,} total reviews")
            print(f"✓ Found {len(self.unique_categories):,} unique categories")

            # Print info before clustering
            print("\n" + "=" * 50)
            print("Before Clustering:")
            print(f"Total Reviews: {len(self.df):,}")
            print(f"Total Products: {self.df['asin'].nunique():,}")
            print(f"Unique Categories: {len(self.unique_categories):,}")
            print("=" * 50 + "\n")

            # Memory cleanup
            del train_df, val_df, test_df
            gc.collect()

        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise

    def generate_embeddings(self) -> None:
        """Generate embeddings using SentenceTransformer"""
        print("\nGenerating embeddings...")
        self.model = SentenceTransformer(self.config["MODEL_NAME"])
        self.category_embeddings = self.model.encode(
            self.unique_categories,
            batch_size=self.config["BATCH_SIZE"],
            show_progress_bar=True,
            device=self.config["DEVICE"],
        )

    def reduce_dimensions(self) -> None:
        """Reduce dimensions with UMAP"""
        print("\nReducing dimensions with UMAP...")
        self.umap_reducer = umap.UMAP(
            n_neighbors=5,
            n_components=3,
            min_dist=0.01,
            metric="cosine",
            random_state=42,
        )
        self.umap_embeddings = self.umap_reducer.fit_transform(self.category_embeddings)

        # Save the UMAP reducer
        with open(self.data_path / "umap_reducer.pkl", "wb") as f:
            pickle.dump(self.umap_reducer, f)

    def perform_clustering(self) -> None:
        """Perform HDBSCAN clustering"""
        print("\nPerforming HDBSCAN clustering...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            cluster_selection_epsilon=0.5,
            alpha=0.5,
        )
        self.cluster_labels = self.clusterer.fit_predict(self.umap_embeddings)

        # Save the HDBSCAN clusterer
        with open(self.data_path / "hdbscan_clusterer.pkl", "wb") as f:
            pickle.dump(self.clusterer, f)

        # Create mapping and add to DataFrame
        category_to_cluster = dict(zip(self.unique_categories, self.cluster_labels))
        self.df["meta_category"] = self.df["category"].map(category_to_cluster)

        print("\n" + "=" * 50)
        print("After Clustering:")
        print(f"Total Reviews: {len(self.df):,}")
        print(f"Total Products: {self.df['asin'].nunique():,}")
        print(f"Unique Categories: {len(self.unique_categories):,}")
        print(
            f"Number of Clusters (excluding noise): {len(set(self.cluster_labels)) - 1:,}"
        )
        print("=" * 50 + "\n")

    def evaluate_clustering(self) -> Dict:
        """Comprehensive evaluation of HDBSCAN clustering"""
        print("\nClustering Evaluation")
        print("=" * 50)

        # 1. Basic Clustering Statistics
        n_clusters = len(set(self.cluster_labels)) - (
            1 if -1 in self.cluster_labels else 0
        )
        n_noise = np.sum(self.cluster_labels == -1)
        noise_ratio = n_noise / len(self.cluster_labels)

        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise} ({noise_ratio:.2%})")

        # 2. Cluster Sizes and Distribution
        cluster_sizes = Counter(self.cluster_labels)
        if -1 in cluster_sizes:
            del cluster_sizes[-1]

        print("\nCluster Size Distribution:")
        for cluster, size in cluster_sizes.most_common():
            print(
                f"Cluster {cluster}: {size} points ({size/len(self.cluster_labels):.2%})"
            )

        # 3. Cluster Validity Metrics
        valid_mask = self.cluster_labels != -1
        valid_embeddings = self.umap_embeddings[valid_mask]
        valid_labels = self.cluster_labels[valid_mask]

        metrics = {}
        if len(set(valid_labels)) > 1:
            metrics["silhouette"] = silhouette_score(valid_embeddings, valid_labels)
            metrics["calinski_harabasz"] = calinski_harabasz_score(
                valid_embeddings, valid_labels
            )
            metrics["davies_bouldin"] = davies_bouldin_score(
                valid_embeddings, valid_labels
            )

            print("\nValidity Metrics:")
            print(f"Silhouette Score: {metrics['silhouette']:.3f}")
            print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.3f}")
            print(f"Davies-Bouldin Score: {metrics['davies_bouldin']:.3f}")

        # Save evaluation metrics
        pd.DataFrame([metrics]).to_csv(
            self.data_path / "clustering_evaluation_metrics.csv"
        )
        print(
            f"\n✓ Saved evaluation metrics to: {self.data_path}/clustering_evaluation_metrics.csv"
        )

        return metrics

    def plot_metrics(self, metrics: Dict) -> None:
        """Plot clustering metrics"""
        plt.figure(figsize=(12, 6))
        metrics_to_plot = {
            k: v
            for k, v in metrics.items()
            if v is not None and isinstance(v, (int, float))
        }
        plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
        plt.xticks(rotation=45)
        plt.title("Clustering Quality Metrics")
        plt.tight_layout()
        plt.savefig(self.image_dir / "clustering_metrics.png")
        plt.close()

    def plot_category_clusters(self, meta_category_names: Dict[int, str]) -> None:
        """Plot category clusters as a network graph"""
        G = nx.Graph()

        # Create cluster contents
        cluster_contents = {}
        for cluster in range(len(meta_category_names)):
            mask = self.cluster_labels == cluster
            cluster_contents[cluster] = [
                self.unique_categories[i] for i in np.where(mask)[0]
            ]

        # Add nodes and edges
        for cluster_id, categories in cluster_contents.items():
            meta_name = meta_category_names[cluster_id]
            G.add_node(meta_name, node_type="meta", size=2000)
            for category in categories:
                G.add_node(category, node_type="sub", size=1000)
                G.add_edge(meta_name, category)

        plt.figure(figsize=(20, 12))
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes and edges
        meta_nodes = [
            node for node, attr in G.nodes(data=True) if attr.get("node_type") == "meta"
        ]
        sub_nodes = [
            node for node, attr in G.nodes(data=True) if attr.get("node_type") == "sub"
        ]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=meta_nodes,
            node_color="lightblue",
            node_size=2000,
            alpha=0.6,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sub_nodes,
            node_color="lightgreen",
            node_size=1000,
            alpha=0.4,
        )
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("Amazon Category Clusters", fontsize=16, pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(self.image_dir / "amazon_category_clusters.png")
        plt.close()

    def plot_treemap(self, meta_category_names: Dict[int, str]) -> None:
        """Plot category hierarchy as a treemap with membership probabilities"""
        data = []
        for cluster_id, categories in self.get_cluster_contents(
            meta_category_names
        ).items():
            meta_name = meta_category_names[cluster_id]
            for category in categories:
                mask = self.cluster_labels == cluster_id
                prob = self.clusterer.probabilities_[mask][
                    list(categories).index(category)
                ]
                data.append(
                    {
                        "Meta-Category": meta_name,
                        "Category": category,
                        "Probability": prob,
                    }
                )

        df_treemap = pd.DataFrame(data)
        fig = px.treemap(
            df_treemap,
            path=["Meta-Category", "Category"],
            values="Probability",
            color="Probability",
            title="Category Hierarchy with Membership Probabilities",
            color_continuous_scale="RdYlBu",
        )
        fig.write_html(self.image_dir / "category_treemap.html")

    def plot_silhouette_analysis(self, meta_category_names: Dict[int, str]) -> None:
        """Plot silhouette analysis visualization"""
        silhouette_vals = silhouette_samples(self.umap_embeddings, self.cluster_labels)

        plt.figure(figsize=(15, 10))
        y_lower = 10

        for cluster_id in range(len(meta_category_names)):
            cluster_silhouette_vals = silhouette_vals[self.cluster_labels == cluster_id]
            cluster_silhouette_vals.sort()

            size_cluster = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster

            color = plt.cm.nipy_spectral(float(cluster_id) / len(meta_category_names))
            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            plt.text(
                -0.05, y_lower + 0.5 * size_cluster, meta_category_names[cluster_id]
            )
            y_lower = y_upper + 10

        plt.title("Silhouette Analysis of Clusters")
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster")
        plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
        plt.savefig(self.image_dir / "silhouette_analysis.png")
        plt.close()

    def get_cluster_contents(self, meta_category_names: Dict[int, str]) -> Dict:
        """Helper method to get cluster contents"""
        cluster_contents = {}
        for cluster in range(len(meta_category_names)):
            mask = self.cluster_labels == cluster
            cluster_contents[cluster] = [
                self.unique_categories[i] for i in np.where(mask)[0]
            ]
        return cluster_contents

    def save_results(self, run_id: str = "latest") -> None:
        """Save clustered data and print statistics"""
        try:
            # Save clustered DataFrame
            output_path = self.data_path / f"amazon_reviews_clustered_{run_id}.parquet"
            self.df.to_parquet(output_path)
            print(f"\n✓ Saved clustered data to: {output_path}")

            # Print clustering statistics
            noise_mask = self.cluster_labels == -1
            noise_ratio = noise_mask.sum() / len(self.cluster_labels)
            print(f"Noise ratio: {noise_ratio:.2%}")

            # Print categories marked as noise
            noise_categories = [
                cat
                for cat, label in zip(self.unique_categories, self.cluster_labels)
                if label == -1
            ]
            print("\nCategories marked as noise:")
            for cat in noise_categories:
                print(f"- {cat}")

            # Cluster size distribution
            cluster_sizes = Counter(self.cluster_labels[self.cluster_labels != -1])
            print("\nCluster size distribution:")
            for cluster_id, size in cluster_sizes.most_common():
                print(f"Cluster {cluster_id}: {size} categories")
        except Exception as e:
            print(f"Error saving results: {e}")
            raise

    def plot_probability_distribution(
        self, meta_category_names: Dict[int, str]
    ) -> None:
        """Plot cluster membership probability distributions"""
        plt.figure(figsize=(15, 6))

        for cluster_id in range(len(meta_category_names)):
            mask = self.cluster_labels == cluster_id
            cluster_probs = self.clusterer.probabilities_[mask]
            sns.kdeplot(data=cluster_probs, label=meta_category_names[cluster_id])

        plt.title("Cluster Membership Probability Distribution")
        plt.xlabel("Probability")
        plt.ylabel("Density")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(self.image_dir / "probability_distribution.png")
        plt.close()

    def plot_cluster_comparison(self, meta_category_names: Dict[int, str]) -> None:
        """Plot cluster size and average probability comparison"""
        metrics_df = pd.DataFrame()

        for cluster_id in range(len(meta_category_names)):
            mask = self.cluster_labels == cluster_id
            cluster_probs = self.clusterer.probabilities_[mask]
            cluster_categories = [self.unique_categories[i] for i in np.where(mask)[0]]

            metrics = {
                "Meta-Category": meta_category_names[cluster_id],
                "Size": len(cluster_categories),
                "Avg Probability": np.mean(cluster_probs),
            }
            metrics_df = pd.concat(
                [metrics_df, pd.DataFrame([metrics])], ignore_index=True
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.barplot(data=metrics_df, x="Meta-Category", y="Size", ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.set_title("Cluster Sizes")

        sns.barplot(data=metrics_df, x="Meta-Category", y="Avg Probability", ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.set_title("Average Cluster Probabilities")

        plt.tight_layout()
        plt.savefig(self.image_dir / "cluster_comparison.png")
        plt.close()

    def run_clustering_pipeline(self) -> None:
        """Run the complete clustering pipeline"""
        print("\nStarting clustering pipeline...")

        # Define meta-category names
        meta_category_names = {
            0: "Entertainment & General Retail",
            1: "Technology & Automotive",
            2: "Industrial & DIY",
            3: "Health & Beauty",
            4: "Home & Garden",
        }

        # Run pipeline
        self.load_data()
        self.generate_embeddings()
        self.reduce_dimensions()
        self.perform_clustering()

        print("\nStarting evaluation and visualization...")
        # Evaluate and visualize
        metrics = self.evaluate_clustering()
        self.plot_metrics(metrics)
        self.plot_category_clusters(meta_category_names)
        self.plot_treemap(meta_category_names)
        self.plot_silhouette_analysis(meta_category_names)
        self.plot_probability_distribution(meta_category_names)
        self.plot_cluster_comparison(meta_category_names)

        # Save final results
        self.save_results(run_id="latest")

        print("\nClustering pipeline completed successfully!")


def main():
    """Example usage of the CategoryClusterer"""
    # Initialize and run clusterer
    clusterer = CategoryClusterer()
    clusterer.run_clustering_pipeline()
    return clusterer


if __name__ == "__main__":
    main()
