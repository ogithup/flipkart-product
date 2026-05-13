"""
Content-based recommendation system for the Flipkart products dataset.

This project follows the instructor guide with an item-to-item content-based
setup because the dataset does not contain user-item interaction logs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TOP_K = 5
DATA_PATH = Path("flipkart_com-ecommerce_sample.csv")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def ensure_directories() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [token for token in text.split() if token not in ENGLISH_STOP_WORDS and len(token) > 1]
    return " ".join(tokens)


def parse_category_tree(raw_value: str) -> List[str]:
    if pd.isna(raw_value):
        return ["unknown"]
    text = str(raw_value).replace('["', "").replace('"]', "")
    parts = [part.strip().lower() for part in text.split(">>") if part.strip()]
    return parts or ["unknown"]


def parse_specifications(raw_value: str) -> str:
    if pd.isna(raw_value):
        return ""
    text = str(raw_value)
    text = re.sub(r'"\s*=>\s*"', " ", text)
    text = text.replace('{"product_specification"=>[', " ")
    text = text.replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
    text = text.replace('"', " ").replace("=>", " ").replace(",", " ")
    return normalize_text(text)


def extract_spec_pairs(raw_value: str) -> Dict[str, str]:
    if pd.isna(raw_value):
        return {}
    matches = re.findall(r'"key"\s*=>\s*"([^"]+)"\s*,\s*"value"\s*=>\s*"([^"]*)"', str(raw_value))
    spec_pairs: Dict[str, str] = {}
    for raw_key, raw_value_text in matches:
        key = normalize_text(raw_key)
        value = normalize_text(raw_value_text)
        if key and value:
            spec_pairs[key] = value
    return spec_pairs


def brand_tokens(brand: str) -> Set[str]:
    tokens = set(normalize_text(brand).split())
    return {token for token in tokens if token != "unknown"}


def strip_brand_from_text(text: str, brand: str) -> str:
    brand_token_set = brand_tokens(brand)
    tokens = [token for token in normalize_text(text).split() if token not in brand_token_set]
    return " ".join(tokens).strip()


def sanitize_category_parts(category_parts: List[str], brand: str) -> List[str]:
    cleaned_parts: List[str] = []
    for part in category_parts:
        cleaned = strip_brand_from_text(part, brand)
        if cleaned:
            cleaned_parts.append(cleaned)
    return cleaned_parts or ["unknown"]


def infer_leaf_category(category_parts: List[str]) -> str:
    if not category_parts:
        return "unknown"
    if len(category_parts) == 1:
        return category_parts[0]

    previous_part = category_parts[-2]
    leaf_part = category_parts[-1]
    previous_tokens = set(previous_part.split())
    leaf_tokens = set(leaf_part.split())

    if previous_tokens and previous_tokens.issubset(leaf_tokens) and len(leaf_tokens) > len(previous_tokens):
        return previous_part
    return leaf_part


def build_attribute_signature(spec_pairs: Dict[str, str]) -> str:
    priority_fields = [
        "type",
        "ideal for",
        "occasion",
        "pattern",
        "fabric",
        "material",
        "sleeve",
        "fit",
        "neck",
        "color",
        "shade",
        "size",
        "ring size",
        "pack of",
        "designed for",
        "compatible model",
        "model name",
        "model number",
        "model id",
        "width",
        "height",
        "depth",
        "length",
    ]

    attribute_chunks: List[str] = []
    for field in priority_fields:
        value = spec_pairs.get(field)
        if value:
            attribute_chunks.append(f"{field} {value}")
    return " ".join(attribute_chunks).strip()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    return pd.read_csv(path)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.drop_duplicates(subset=["uniq_id"]).drop_duplicates(subset=["pid"], keep="first")

    data["brand"] = data["brand"].fillna("unknown")
    data["description"] = data["description"].fillna("")
    data["product_specifications"] = data["product_specifications"].fillna("")
    data["retail_price"] = pd.to_numeric(data["retail_price"], errors="coerce")
    data["discounted_price"] = pd.to_numeric(data["discounted_price"], errors="coerce")
    data["retail_price"] = data["retail_price"].fillna(data["discounted_price"].median())
    data["discounted_price"] = data["discounted_price"].fillna(data["retail_price"].median())

    category_parts = data["product_category_tree"].apply(parse_category_tree)
    data["category_parts"] = category_parts
    data["root_category"] = category_parts.apply(lambda items: normalize_text(items[0]) if items else "unknown")
    data["sanitized_category_parts"] = data.apply(
        lambda row: sanitize_category_parts(row["category_parts"], row["brand"]),
        axis=1,
    )
    data["category_path"] = data["sanitized_category_parts"].apply(lambda items: " | ".join(items))
    data["leaf_category"] = data["sanitized_category_parts"].apply(infer_leaf_category)
    data["spec_pairs"] = data["product_specifications"].apply(extract_spec_pairs)
    data["spec_text"] = data["product_specifications"].apply(parse_specifications)
    data["attribute_signature"] = data["spec_pairs"].apply(build_attribute_signature)

    data["clean_name"] = data["product_name"].fillna("").apply(normalize_text)
    data["clean_brand"] = data["brand"].fillna("unknown").apply(normalize_text)
    data["clean_description"] = data["description"].fillna("").apply(normalize_text)
    data["clean_category"] = data["category_path"].fillna("").apply(normalize_text)
    data["clean_leaf_category"] = data["leaf_category"].fillna("unknown").apply(normalize_text)
    data["clean_attributes"] = data["attribute_signature"].fillna("").apply(normalize_text)

    data["combined_text"] = (
        data["clean_leaf_category"]
        + " "
        + data["clean_leaf_category"]
        + " "
        + data["clean_name"]
        + " "
        + data["clean_name"]
        + " "
        + data["clean_brand"]
        + " "
        + data["clean_category"]
        + " "
        + data["clean_attributes"]
        + " "
        + data["clean_attributes"]
        + " "
        + data["clean_description"]
        + " "
        + data["spec_text"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    data = data[data["combined_text"].str.len() > 0].reset_index(drop=True)
    data = data.drop(columns=["category_parts", "sanitized_category_parts", "spec_pairs"])
    return data


def train_test_items(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE, shuffle=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)


def fit_vectorizer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    vectorizer = build_vectorizer()
    train_matrix = vectorizer.fit_transform(train_df["combined_text"])
    test_matrix = vectorizer.transform(test_df["combined_text"])
    return vectorizer, train_matrix, test_matrix


def recommend_from_train_item(
    item_row: pd.Series,
    train_df: pd.DataFrame,
    train_matrix,
    vectorizer: TfidfVectorizer,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    item_vector = vectorizer.transform([item_row["combined_text"]])
    scores = cosine_similarity(item_vector, train_matrix).ravel()
    candidate_df = train_df.copy()
    candidate_df["similarity_score"] = scores
    candidate_df = candidate_df.sort_values("similarity_score", ascending=False)
    candidate_df = candidate_df[candidate_df["pid"] != item_row["pid"]]
    return candidate_df.head(top_k)[
        ["product_name", "brand", "root_category", "leaf_category", "discounted_price", "similarity_score"]
    ]


def recommend_by_product_name(
    product_name: str,
    full_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    train_df: pd.DataFrame,
    train_matrix,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    matches = full_df[full_df["product_name"].str.lower() == product_name.lower()]
    if matches.empty:
        raise ValueError(f"Product name not found: {product_name}")
    item_row = matches.iloc[0]
    return recommend_from_train_item(item_row, train_df, train_matrix, vectorizer, top_k=top_k)


def relevant_train_items(item_row: pd.Series, train_df: pd.DataFrame) -> pd.DataFrame:
    same_leaf = train_df[train_df["leaf_category"] == item_row["leaf_category"]]
    if len(same_leaf) >= TOP_K:
        return same_leaf
    same_root = train_df[train_df["root_category"] == item_row["root_category"]]
    return same_root


def average_precision_at_k(relevant_flags: List[int], num_relevant: int, k: int) -> float:
    if num_relevant == 0:
        return 0.0
    precision_sum = 0.0
    hits = 0
    for rank, is_relevant in enumerate(relevant_flags[:k], start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(num_relevant, k)


def evaluate_recommender(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_matrix,
    train_matrix,
    k: int = TOP_K,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    similarity_matrix = cosine_similarity(test_matrix, train_matrix)
    records: List[Dict[str, float]] = []

    for test_idx, item_row in test_df.iterrows():
        relevant_df = relevant_train_items(item_row, train_df)
        relevant_pids = set(relevant_df["pid"].tolist())
        if not relevant_pids:
            continue

        ranked_idx = np.argsort(similarity_matrix[test_idx])[::-1]
        recommended_pids = []
        for idx in ranked_idx:
            candidate_pid = train_df.iloc[idx]["pid"]
            if candidate_pid == item_row["pid"]:
                continue
            recommended_pids.append(candidate_pid)
            if len(recommended_pids) == k:
                break

        relevant_flags = [1 if pid in relevant_pids else 0 for pid in recommended_pids]
        hits = sum(relevant_flags)
        precision = hits / k
        recall = hits / len(relevant_pids)
        ap = average_precision_at_k(relevant_flags, len(relevant_pids), k)

        records.append(
            {
                "query_product": item_row["product_name"],
                "query_leaf_category": item_row["leaf_category"],
                "num_relevant_train_items": len(relevant_pids),
                f"precision@{k}": precision,
                "recall": recall,
                "average_precision": ap,
            }
        )

    per_item_results = pd.DataFrame(records)
    metrics = {
        f"Precision@{k}": per_item_results[f"precision@{k}"].mean(),
        "Recall": per_item_results["recall"].mean(),
        "MAP": per_item_results["average_precision"].mean(),
        "Evaluated Items": float(len(per_item_results)),
    }
    return per_item_results, metrics


def dataset_summary_table(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "Metric": [
                "Total raw records",
                "Records after deduplication and cleaning",
                "Unique products (pid)",
                "Unique brands",
                "Unique root categories",
                "Missing brand values in raw data",
                "Missing description values in raw data",
                "Median retail price",
                "Median discounted price",
            ],
            "Value": [
                len(raw_df),
                len(clean_df),
                clean_df["pid"].nunique(),
                clean_df["brand"].nunique(),
                clean_df["root_category"].nunique(),
                int(raw_df["brand"].isna().sum()),
                int(raw_df["description"].isna().sum()),
                round(clean_df["retail_price"].median(), 2),
                round(clean_df["discounted_price"].median(), 2),
            ],
        }
    )
    return summary


def sample_products_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["product_name", "brand", "root_category", "leaf_category", "retail_price", "discounted_price"]
    return df[cols].head(n)


def save_tables(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    per_item_results: pd.DataFrame,
    metrics: Dict[str, float],
    example_recommendations: pd.DataFrame,
) -> None:
    dataset_summary_table(raw_df, clean_df).to_csv(TABLE_DIR / "dataset_summary.csv", index=False)
    sample_products_table(clean_df).to_csv(TABLE_DIR / "sample_products.csv", index=False)
    train_df.head(100).to_csv(TABLE_DIR / "train_sample.csv", index=False)
    test_df.head(100).to_csv(TABLE_DIR / "test_sample.csv", index=False)
    per_item_results.to_csv(TABLE_DIR / "per_item_evaluation.csv", index=False)
    pd.DataFrame([metrics]).to_csv(TABLE_DIR / "evaluation_results.csv", index=False)
    example_recommendations.to_csv(TABLE_DIR / "example_recommendations.csv", index=False)


def plot_top_categories(df: pd.DataFrame) -> None:
    top_categories = df["root_category"].value_counts().head(10).sort_values()
    plt.figure(figsize=(10, 6))
    top_categories.plot(kind="barh", color="#2b6cb0")
    plt.title("Top 10 Root Categories")
    plt.xlabel("Number of Products")
    plt.ylabel("Root Category")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top_categories.png", dpi=300)
    plt.close()


def plot_metrics(metrics: Dict[str, float]) -> None:
    plot_data = pd.Series(
        {
            "Precision@5": metrics["Precision@5"],
            "Recall": metrics["Recall"],
            "MAP": metrics["MAP"],
        }
    )
    plt.figure(figsize=(8, 5))
    plot_data.plot(kind="bar", color=["#2f855a", "#d69e2e", "#c53030"])
    plt.title("Recommendation Performance")
    plt.ylabel("Score")
    plt.ylim(0, max(plot_data.max() * 1.15, 0.1))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "evaluation_metrics.png", dpi=300)
    plt.close()


def draw_pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    steps = [
        "Raw CSV Dataset",
        "Preprocessing\nCleaning + Feature Extraction",
        "TF-IDF\nVectorization",
        "Cosine Similarity",
        "Top-5 Recommendations\nand Evaluation",
    ]
    x_positions = [0.02, 0.23, 0.45, 0.65, 0.82]

    for x_pos, label in zip(x_positions, steps):
        patch = FancyBboxPatch(
            (x_pos, 0.35),
            0.14,
            0.28,
            boxstyle="round,pad=0.02",
            facecolor="#edf2f7",
            edgecolor="#2d3748",
            linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(x_pos + 0.07, 0.49, label, ha="center", va="center", fontsize=10)

    for arrow_x in [0.17, 0.39, 0.59, 0.79]:
        ax.annotate("", xy=(arrow_x + 0.05, 0.49), xytext=(arrow_x, 0.49), arrowprops=dict(arrowstyle="->", lw=2))

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "recommendation_pipeline.png", dpi=300)
    plt.close()


def main() -> None:
    ensure_directories()

    raw_df = load_dataset(DATA_PATH)
    clean_df = preprocess_dataset(raw_df)
    train_df, test_df = train_test_items(clean_df)
    vectorizer, train_matrix, test_matrix = fit_vectorizer(train_df, test_df)

    example_item = test_df.iloc[0]
    example_recommendations = recommend_from_train_item(example_item, train_df, train_matrix, vectorizer, top_k=TOP_K)

    per_item_results, metrics = evaluate_recommender(test_df, train_df, test_matrix, train_matrix, k=TOP_K)

    save_tables(raw_df, clean_df, train_df, test_df, per_item_results, metrics, example_recommendations)
    plot_top_categories(clean_df)
    plot_metrics(metrics)
    draw_pipeline_figure()

    clean_df.to_csv(OUTPUT_DIR / "cleaned_products.csv", index=False)

    print("Content-based recommender completed.")
    print(f"Raw records: {len(raw_df)}")
    print(f"Clean records: {len(clean_df)}")
    print(f"Training items: {len(train_df)}")
    print(f"Testing items: {len(test_df)}")
    print(f"Precision@5: {metrics['Precision@5']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"MAP: {metrics['MAP']:.4f}")
    print("\nExample query product:")
    print(example_item["product_name"])
    print("\nTop-5 recommendations:")
    print(example_recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
