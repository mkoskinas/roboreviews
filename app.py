from flask import Flask, make_response
import plotly
import json

app = Flask(__name__)

@app.route("/")
def blog():
    fig_data = [{
        "branchvalues": "total",
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "ids": ["Entertainment & General Retail/All_Beauty", "Entertainment & General Retail/Amazon_Fashion", "Technology & Automotive/Appliances", "Industrial & DIY/Arts_Crafts_and_Sewing", "Technology & Automotive/Automotive", "Entertainment & General Retail/Baby_Products", "Health & Beauty/Beauty_and_Personal_Care", "Entertainment & General Retail/Books", "Entertainment & General Retail/CDs_and_Vinyl", "Entertainment & General Retail/Cell_Phones_and_Accessories", "Entertainment & General Retail/Digital_Music", "Technology & Automotive/Electronics", "Entertainment & General Retail/Gift_Cards", "Home & Garden/Grocery_and_Gourmet_Food", "Entertainment & General Retail/Handmade_Products", "Health & Beauty/Health_and_Household", "Health & Beauty/Health_and_Personal_Care", "Home & Garden/Home_and_Kitchen", "Industrial & DIY/Industrial_and_Scientific", "Entertainment & General Retail/Kindle_Store", "Entertainment & General Retail/Magazine_Subscriptions", "Entertainment & General Retail/Movies_and_TV", "Entertainment & General Retail/Musical_Instruments", "Entertainment & General Retail/Office_Products", "Home & Garden/Patio_Lawn_and_Garden", "Entertainment & General Retail/Pet_Supplies", "Technology & Automotive/Software", "Entertainment & General Retail/Sports_and_Outdoors", "Entertainment & General Retail/Subscription_Boxes", "Industrial & DIY/Tools_and_Home_Improvement", "Entertainment & General Retail/Toys_and_Games", "Entertainment & General Retail/Video_Games", "Entertainment & General Retail/[UNKNOWN]", "Entertainment & General Retail", "Health & Beauty", "Home & Garden", "Industrial & DIY", "Technology & Automotive"],
        "labels": ["All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books", "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food", "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store", "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Software", "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games", "[UNKNOWN]", "Entertainment & General Retail", "Health & Beauty", "Home & Garden", "Industrial & DIY", "Technology & Automotive"],
        "parents": ["Entertainment & General Retail", "Entertainment & General Retail", "Technology & Automotive", "Industrial & DIY", "Technology & Automotive", "Entertainment & General Retail", "Health & Beauty", "Entertainment & General Retail", "Entertainment & General Retail", "Entertainment & General Retail", "Entertainment & General Retail", "Technology & Automotive", "Entertainment & General Retail", "Home & Garden", "Entertainment & General Retail", "Health & Beauty", "Health & Beauty", "Home & Garden", "Industrial & DIY", "Entertainment & General Retail", "Entertainment & General Retail", "Entertainment & General Retail", "Entertainment & General Retail", "Entertainment & General Retail", "Home & Garden", "Entertainment & General Retail", "Technology & Automotive", "Entertainment & General Retail", "Entertainment & General Retail", "Industrial & DIY", "Entertainment & General Retail", "Entertainment & General Retail", "Entertainment & General Retail", "", "", "", "", ""],
        "values": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 20, 3, 3, 3, 4],
        "type": "sunburst"
    }]
    
    graphJSON = json.dumps(fig_data, cls=plotly.utils.PlotlyJSONEncoder)
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>RoboReviews: The New Product Reviews Aggregator</title>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <style>
        body {
            font-family: 'Source Serif Pro', serif;
            line-height: 1.8;
            margin: 0;
            padding: 0;
            background-color: #fff;
            color: rgba(0, 0, 0, 0.84);
            font-size: 18px;
        }
        .container {
            width: 80%%;
            margin: auto;
            max-width: 800px;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        }
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 30px;
        }
        .title-section {
            flex: 1;
            padding-right: 20px;
        }
        img {
            max-width: 100%%;
            height: auto;
            display: block;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }
        h1 {
            font-size: 42px;
            line-height: 1.2;
            margin-bottom: 8px;
            color: rgba(0, 0, 0, 0.84);
            font-weight: 700;
        }
        h2 {
            font-size: 32px;
            margin-top: 48px;
            margin-bottom: 16px;
            color: rgba(0, 0, 0, 0.84);
        }
        h3 {
            font-size: 24px;
            margin-top: 32px;
            margin-bottom: 12px;
            color: rgba(0, 0, 0, 0.84);
        }
        p {
            margin-bottom: 20px;
            font-size: 18px;
            line-height: 1.8;
        }
        ul {
            margin: 20px 0;
            padding-left: 24px;
        }
        li {
            margin-bottom: 12px;
            line-height: 1.7;
        }
        code {
            background: rgba(0, 0, 0, 0.05);
            padding: 3px 6px;
            border-radius: 3px;
            font-family: 'Menlo', monospace;
            font-size: 16px;
        }
        pre {
            background: rgba(0, 0, 0, 0.05);
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Menlo', monospace;
            font-size: 16px;
            line-height: 1.5;
            margin: 20px 0;
        }
        .metric-card {
            background: #fff;
            padding: 24px;
            border-radius: 8px;
            margin: 16px 0;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        }
        em {
            font-style: italic;
        }
        strong {
            font-weight: 600;
            color: rgba(0, 0, 0, 0.84);
        }
        @media (max-width: 768px) {
            .container {
                width: 90%%;
                padding: 16px;
            }
            .header-container {
                flex-direction: column;
            }
            img {
                max-width: 100%%;
                margin: 20px auto;
            }
            .title-section {
                padding-right: 0;
                text-align: center;
            }
            h1 { font-size: 32px; }
            h2 { font-size: 28px; }
            h3 { font-size: 22px; }
            body { font-size: 16px; }
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <h1>RoboReviews: The New Product Review Aggregator</h1>
        
        <div class="header-container">
            <div class="title-section">
                <img src="/static/images/realistic_robot_workspace.png" alt="A realistic humanoid robot analyzing data" />
            </div>
        </div>

        <h2>Introduction</h2>
        <p>In today's digital age, customer reviews hold immense value for companies aiming to refine their products and services. However, the sheer volume of reviews can make extracting actionable insights a challenge. Enter <em>RoboReviews</em>, a project designed to demonstrate how businesses can analyse, cluster, and summarsze customer feedback using cutting-edge AI techniques. This blog will walk you through the project's objectives, methodologies, and detailed data pipeline.</p>

        <h2>Project Objectives</h2>
        <ul>
            <li><strong>Classifying Customer Reviews:</strong> Automatically categorize reviews as positive, negative, or neutral to provide insights into customer sentiment.</li>
            <li><strong>Clustering Product Categories:</strong> Group products into 4-6 meta-categories using clustering algorithms to simplify product analysis.</li>
            <li><strong>Summarizing Reviews:</strong> Generate concise summaries of reviews and recommend top products in each category, leveraging generative AI models.</li>
        </ul>

        <h2>The Data Pipeline</h2>
        <h3>Step 1: Data Acquisition</h3>
        <div class="metric-card">
            <ul>
                <li><strong>Script:</strong> <code>download.py</code></li>
                <li><strong>Task:</strong> Retrieve customer review datasets using the <code>datasets</code> library from Hugging Face.</li>
                <li><strong>Details:</strong>
                    <ul>
                        <li>Dataset source: <a href='https://amazon-reviews-2023.github.io/data_loading/huggingface.html' target='_blank'>Amazon Reviews 2023</a>, containing all 33 categories.</li>
                        <li>Storage:</li>
                        <ul>
                            <li><strong>Original Dataset:</strong> Stored in the local directory <code>amazon_reviews/{category}.dataset</code>.</li>
                            <li><strong>Processed Dataset:</strong> Converted to Parquet files and stored in <code>amazon_reviews_processed/{category}.parquet</code>.</li>
                            <li><strong>Google Drive Backup:</strong> Parquet files are backed up in <code>/content/drive/MyDrive/amazon_reviews_backup/{category}.parquet</code>.</li>
                        </ul>
                        <li>Key columns retained:
                            <ul>
                                <li><code>text</code>: The review content.</li>
                                <li><code>rating</code>: The numerical rating provided by the user.</li>
                                <li><code>asin</code>: Product identifier.</li>
                                <li><code>title</code>: The title of the review.</li>
                                <li><code>helpful_vote</code>: Count of helpful votes for the review.</li>
                            </ul>
                        </li>
                        <li>Process Features:
                            <ul>
                                <li>Google Drive was mounted to ensure datasets were securely backed up.</li>
                                <li>A JSON progress file tracked completed categories and the total number of processed reviews.</li>
                                <li>Integrity checks ensured the Parquet files were complete and readable after processing.</li>
                            </ul>
                        </li>
                    </ul>
                </li>
            </ul>
        </div>
        <h3>Step 2: Data Preparation</h3>
        <div class="metric-card">
            <ul>
                <li><strong>Script:</strong> <code>prepare.py</code></li>
                <li><strong>Task:</strong> Clean, preprocess, and structure data for analysis.</li>
                <li><strong>Details:</strong>
                    <ul>
                        <li>Created a 3%% sample dataset from the original data to enable faster iterations and testing.</li>
                        <li>Combined sampled files into a single dataset for consistency and ease of use.</li>
                        <li>Split data into training, validation, and test sets:
                            <ul>
                                <li><strong>Test:</strong> 10%% of the total dataset</li>
                                <li><strong>Validation:</strong> 10%% of the remaining 90%% (9%% of the total dataset)</li>
                                <li><strong>Train:</strong> The remaining 81%% of the total dataset</li>
                            </ul>
                        </li>
                        <li>Preprocessing steps applied:
                            <ul>
                                <li>Removed invalid data, including reviews with missing text or zero ratings.</li>
                                <li>Normalized text by removing HTML tags, URLs, emails, and extra whitespace.</li>
                                <li>Filtered very short reviews (fewer than 5 characters) and duplicates.</li>
                                <li>Added a sentiment label based on rating:
                                    <ul>
                                        <li><code>1-2:</code> Negative</li>
                                        <li><code>3:</code> Neutral</li>
                                        <li><code>4-5:</code> Positive</li>
                                    </ul>
                                </li>
                                <li>Processed and tokenized text for downstream tasks.</li>
                            </ul>
                        </li>
                        <li>Outputs:
                            <ul>
                                <li><strong>Processed Splits:</strong> Saved as Parquet files (<code>train_processed.parquet</code>, <code>val_processed.parquet</code>, <code>test_processed.parquet</code>).</li>
                                <li>Final dataset includes:
                                    <ul>
                                        <li><code>text</code>: The original review content.</li>
                                        <li><code>processed_text</code>: Cleaned and preprocessed review content.</li>
                                        <li><code>rating</code>: Original rating provided by the user (1-5).</li>
                                        <li><code>sentiment</code>: Derived sentiment label (Positive, Neutral, Negative).</li>
                                        <li><code>category</code>: Product category for each review.</li>
                                        <li><code>helpful_vote</code>: Count of helpful votes for the review.</li>
                                        <li><code>asin</code>: Product identifier.</li>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                    </ul>
                </li>
            </ul>
        </div>

        <h3>Step 3: Sentiment Classification</h3>
        <div class="metric-card">
            <ul>
                <li><strong>Script:</strong> <code>classifier.py</code></li>
                <li><strong>Task:</strong> Classify reviews into positive, negative, or neutral categories.</li>
                <li><strong>Details:</strong>
                    <ul>
                        <li><strong>Model:</strong> Fine-tuned <code>distilroberta-base</code> transformer model with three output labels (Positive, Neutral, Negative).</li>
                        <li><strong>Dataset Preparation:</strong>
                            <ul>
                                <li>Training, validation, and test datasets were balanced by sampling an equal number of reviews for each sentiment class.</li>
                                <li>Final dataset sizes after balancing:
                                    <ul>
                                        <li><strong>Training Set:</strong> Approximately 600,000 reviews (200,000 per class).</li>
                                        <li><strong>Validation Set:</strong> 50,000 reviews.</li>
                                        <li><strong>Test Set:</strong> 50,000 reviews.</li>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                        <li><strong>Training Configuration:</strong>
                            <pre>{
    'MODEL_NAME': 'distilroberta-base',
    'MAX_LENGTH': 256,
    'BATCH_SIZE': 64,
    'CACHE_SIZE': 50000,
    'NUM_EPOCHS': 2,
    'LEARNING_RATE': 2e-5,
    'WARMUP_RATIO': 0.1,
    'WEIGHT_DECAY': 0.01,
    'GRAD_ACCUMULATION': 2,
    'ADAM_EPSILON': 1e-8,
    'MAX_GRAD_NORM': 1.0,
    'DROPOUT': 0.1,
    'LABEL_SMOOTHING': 0.1,
    'PATIENCE': 3,
    'MIN_DELTA': 0.001
}</pre>
                        </li>
                        <li><strong>Evaluation Metrics:</strong>
                            <ul>
                                <li>Accuracy, Precision, Recall, and F1 Score (both weighted and per-class).</li>
                                <li>Confusion matrix visualization to assess classification performance.</li>
                            </ul>
                        </li>
                        <li><strong>Visualizations:</strong>
                            <ul>
                                <li>Confusion matrix before data balancing:
                                    <img src="/static/images/confusion_matrix_before.png" alt="Confusion Matrix" style="width:100%%; margin: 20px 0;" />
                                </li>
                                <li>Confusion matrix with the balanced dataset:
                                    <img src="/static/images/confusion_matrix.png" alt="Confusion Matrix" style="width:100%%; margin: 20px 0;" />
                                </li>
                            </ul>
                        </li>
                        <li><strong>Outputs:</strong>
                            <ul>
                                <li><strong>Negative:</strong>
                                    <ul>
                                        <li>Precision: 0.7860</li>
                                        <li>Recall: 0.7964</li>
                                        <li>F1: 0.7912</li>
                                    </ul>
                                </li>
                                <li><strong>Neutral:</strong>
                                    <ul>
                                        <li>Precision: 0.7121</li>
                                        <li>Recall: 0.7053</li>
                                        <li>F1: 0.7087</li>
                                    </ul>
                                </li>
                                <li><strong>Positive:</strong>
                                    <ul>
                                        <li>Precision: 0.8917</li>
                                        <li>Recall: 0.8884</li>
                                        <li>F1: 0.8901</li>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                    </ul>
                </li>
            </ul>
        </div>

        <h3>Step 4: Clustering</h3>
        <div class="metric-card">
            <ul>
                <li><strong>Script:</strong> <code>clusterer.py</code></li>
                <li><strong>Task:</strong> Cluster product categories based on customer reviews.</li>
                <li><strong>Details:</strong>
                    <ul>
                        <li><strong>Model:</strong> Sentence-transformers (all-mpnet-base-v2) for generating embeddings of unique product categories.</li>
                        <li><strong>Dimensionality Reduction:</strong>
                            <ul>
                                <li><strong>Technique:</strong> UMAP (Uniform Manifold Approximation and Projection).</li>
                                <li><strong>Parameters:</strong>
                                    <ul>
                                        <li><code>n_neighbors=5</code></li>
                                        <li><code>n_components=3</code></li>
                                        <li><code>min_dist=0.01</code></li>
                                        <li><code>metric='cosine'</code></li>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                        <li><strong>Clustering:</strong>
                            <ul>
                                <li><strong>Algorithm:</strong> HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).</li>
                                <li><strong>Parameters:</strong>
                                    <ul>
                                        <li><code>min_cluster_size=2</code></li>
                                        <li><code>min_samples=1</code></li>
                                        <li><code>metric='euclidean'</code></li>
                                        <li><code>cluster_selection_method='eom'</code></li>
                                        <li><code>cluster_selection_epsilon=0.5</code></li>
                                        <li><code>alpha=0.5</code></li>
                                    </ul>
                                </li>
                                <li><strong>Outputs:</strong>
                                    <ul>
                                        <li>Product Counts in Each Cluster:
                                            <ul>
                                                <li><strong>Meta-Category 0:</strong> 3,298,398 products</li>
                                                <li><strong>Meta-Category 1:</strong> 699,185 products</li>
                                                <li><strong>Meta-Category 2:</strong> 518,720 products</li>
                                                <li><strong>Meta-Category 3:</strong> 475,496 products</li>
                                                <li><strong>Meta-Category 4:</strong> 1,158,879 products</li>
                                            </ul>
                                        </li>
                                        <li>Validity Metrics:
                                            <ul>
                                                <li><strong>Silhouette Score:</strong> 0.537 (higher is better, range: [-1, 1])</li>
                                                <li><strong>Calinski-Harabasz Score:</strong> 34.346 (higher is better)</li>
                                                <li><strong>Davies-Bouldin Score:</strong> 0.538 (lower is better)</li>
                                                <li><strong>Mean cluster membership probability:</strong> 0.982</li>
                                                <li><strong>Std cluster membership probability:</strong> 0.063</li>
                                            </ul>
                                        </li>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                        <li><strong>Category Distribution in Clusters:</strong>
                            <ul>
                                <li><strong>Cluster 0:</strong> Movies_and_TV, All_Beauty, Baby_Products, Video_Games, Kindle_Store (Average probability: 0.999)</li>
                                <li><strong>Cluster 1:</strong> Electronics, Automotive, Software, Appliances (Average probability: 1.000)</li>
                                <li><strong>Cluster 2:</strong> Industrial_and_Scientific, Tools_and_Home_Improvement, Arts_Crafts_and_Sewing (Average probability: 0.948)</li>
                                <li><strong>Cluster 3:</strong> Health_and_Household, Beauty_and_Personal_Care, Health_and_Personal_Care (Average probability: 0.971)</li>
                                <li><strong>Cluster 4:</strong> Home_and_Kitchen, Patio_Lawn_and_Garden, Grocery_and_Gourmet_Food (Average probability: 0.890)</li>
                            </ul>
                        </li>
                        <li><strong>Visualizations:</strong>
                            <ul>
                                <li>Cluster size distribution and category overlap.</li>
                                <li>Sunburst and treemap visualizations of category hierarchies:
                                    <div id="category-chart" style="height:525px; width:100%%; margin: 20px 0; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);"></div>
                                </li>
                                <li>Silhouette analysis of clusters:
                                    <img src="/static/images/silhouette_analysis.png" alt="Silhouette Analysis" style="width:100%%; margin: 20px 0;" />
                                </li>
                                <li>Probability distribution of cluster memberships.</li>
                            </ul>
                        </li>
                    </ul>
                </li>
            </ul>
        </div>
        <h3>Step 5: Content generation/summarisation</h3>
        <div class="metric-card">
            <ul>
                <li><strong>Script:</strong> <code>summarizer.py</code></li>
                <li><strong>Task:</strong> Generate product summaries and comparisons for meta-categories using fine-tuned models and evaluation metrics.</li>
                <li><strong>Details:</strong>
                    <ul>
                        <li><strong>Model:</strong> Mistral-7B-Instruct-v0.2, fine-tuned for summarization tasks.</li>
                        <li><strong>Preprocessing:</strong>
                            <ul>
                                <li>Clustered data from Step 4 used as input.</li>
                                <li>Products grouped by meta-category and further filtered by popularity and review count.</li>
                            </ul>
                        </li>
                        <li><strong>Fine-tuning:</strong>
                            <ul>
                                <li><strong>Training:</strong> LoRA fine-tuning on product review data.</li>
                                <li><strong>Configuration:</strong>
                                    <ul>
                                        <li><code>max_steps=200</code></li>
                                        <li><code>learning_rate=1e-5</code></li>
                                        <li><code>per_device_train_batch_size=4</code></li>
                                        <li><code>gradient_accumulation_steps=8</code></li>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                        <li><strong>Prompt:</strong>
                            <pre style="background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin: 20px 0;">
&lt;s&gt;[INST] Based on customer reviews, write a short article, like a blogpost reviewer would write, about {meta_category} products to help customers choose the best one.

1. Title: Compare the 3 specific products in an engaging way.

2. Introduction: Brief overview of the analysis.

3. TOP 3 RECOMMENDED PRODUCTS:
- The top 3 most recommended products and their key differences.
- Top complaints for each of those products.

4. WORST PRODUCT WARNING:
- What is the worst product in the category {category} and why you should never buy it.

5. Final Verdict:
- Clear recommendation for each top product.
- Final warning about the worst product.

Use this review data:
{review_data}
[/INST]&lt;/s&gt;</pre>
                        </li>
                            <li><strong>Evaluation Metrics:</strong>
    <ul>
        <li><strong>1. BERTScore (Semantic Similarity):</strong>
            <ul>
                <li>Precision: 0.8987</li>
                <li>Recall: 0.9179</li>
                <li>F1: 0.9082</li>
            </ul>
        </li>
        <li><strong>2. Structure Evaluation:</strong>
            <ul>
                <li>Checks for required sections:
                    <ul>
                        <li>Title</li>
                        <li>Introduction</li>
                        <li>Top 3 recommended products</li>
                        <li>Worst product warning</li>
                        <li>Final Verdict</li>
                    </ul>
                </li>
                <li>Results:
                    <ul>
                        <li>Accuracy: 0.92 (sections present/total required)</li>
                        <li>Precision: 0.88 (correct sections/total sections found)</li>
                        <li>Recall: 0.94 (correct sections/required sections)</li>
                        <li>F1: 0.91</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li><strong>3. Product Mention Accuracy:</strong>
            <ul>
                <li>Compares products mentioned in generated text vs source data</li>
                <li>Results:
                    <ul>
                        <li>Accuracy: 0.89 (correct products/total products)</li>
                        <li>Precision: 0.87 (correct products/total mentioned)</li>
                        <li>Recall: 0.94 (mentioned products/source products)</li>
                        <li>F1: 0.90</li>
                    </ul>
                </li>
            </ul>
        </li>
    </ul>
</li>
            </ul>
        </div>
    </div>

    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
        var graphs = %s;
        Plotly.newPlot('category-chart', graphs, {
            title: {
                text: 'Amazon Category Hierarchy',
                font: {
                    family: 'Source Serif Pro, serif',
                    size: 16
                }
            },
            margin: {
                l: 0,
                r: 0,
                b: 0,
                t: 40
            },
            template: 'plotly',
            height: 525,
            width: null
        });
        
        window.onresize = function() {
            Plotly.Plots.resize('category-chart');
        };
    </script>
</body>
</html>
""" % graphJSON
    
    response = make_response(html_content)

    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    
    return response

if __name__ == "__main__":
    app.run(debug=True, port=5000)