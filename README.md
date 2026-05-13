# Flipkart Content-Based Recommendation Project

This repository contains a complete item-to-item content-based recommendation system for the uploaded Flipkart CSV dataset.

## Files
- `content_based_recommender.py`: full Python script
- `content_based_recommender.ipynb`: Jupyter notebook version
- `requirements.txt`: Python dependencies
- `IEEE_Report_Draft.md`: report draft content for IEEE paper writing

## Dataset
Expected input file:

- `flipkart_com-ecommerce_sample.csv`

## Install
```bash
pip install -r requirements.txt
```

## Run the Python Script
```bash
python content_based_recommender.py
```

## Outputs
The script creates:

- `outputs/cleaned_products.csv`
- `outputs/tables/dataset_summary.csv`
- `outputs/tables/sample_products.csv`
- `outputs/tables/per_item_evaluation.csv`
- `outputs/tables/evaluation_results.csv`
- `outputs/tables/example_recommendations.csv`
- `outputs/figures/top_categories.png`
- `outputs/figures/evaluation_metrics.png`
- `outputs/figures/recommendation_pipeline.png`

## Evaluation Strategy
The dataset does not contain user IDs or user-item interactions. Because of that, the project uses the instructor-approved fallback strategy for non-interaction datasets:

- item-to-item content-based recommendation
- relevance based on same leaf category
- fallback to same root category when leaf-category support is insufficient
- metrics: Precision@5, Recall, MAP
