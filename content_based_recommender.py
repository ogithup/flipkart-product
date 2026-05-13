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
MIN_CANDIDATE_POOL = 25
STRONG_MATCH_THRESHOLD = 0.50
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


def first_spec_value(spec_pairs: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        value = spec_pairs.get(key, "")
        if value:
            return value
    return ""


def root_category_group(root_category: str) -> str:
    if "clothing" in root_category:
        return "clothing"
    if "jewellery" in root_category:
        return "jewellery"
    if any(keyword in root_category for keyword in ["mobile", "mobiles", "tablet", "accessories"]):
        return "mobile_accessories"
    if any(keyword in root_category for keyword in ["furniture", "home", "kitchen", "decor", "furnishing"]):
        return "furniture_home"
    return "general"


def build_family_key(root_category: str, leaf_category: str, brand: str, spec_pairs: Dict[str, str]) -> str:
    category_group = root_category_group(root_category)
    key_parts = [root_category, leaf_category]

    if category_group == "clothing":
        key_parts.extend(
            [
                first_spec_value(spec_pairs, ["type"]),
                first_spec_value(spec_pairs, ["ideal for"]),
                first_spec_value(spec_pairs, ["sleeve"]),
                first_spec_value(spec_pairs, ["pattern"]),
                first_spec_value(spec_pairs, ["fabric", "material"]),
            ]
        )
    elif category_group == "jewellery":
        key_parts.extend(
            [
                first_spec_value(spec_pairs, ["type"]),
                first_spec_value(spec_pairs, ["base material", "material"]),
                first_spec_value(spec_pairs, ["gemstone", "semi precious stone type"]),
                first_spec_value(spec_pairs, ["ring size", "size"]),
            ]
        )
    elif category_group == "mobile_accessories":
        key_parts.extend(
            [
                first_spec_value(spec_pairs, ["designed for", "compatible model", "model name"]),
                first_spec_value(spec_pairs, ["type"]),
                brand,
            ]
        )
    elif category_group == "furniture_home":
        key_parts.extend(
            [
                first_spec_value(spec_pairs, ["type"]),
                first_spec_value(spec_pairs, ["material", "primary material"]),
                first_spec_value(spec_pairs, ["width"]),
                first_spec_value(spec_pairs, ["height"]),
                first_spec_value(spec_pairs, ["length", "depth"]),
            ]
        )
    else:
        key_parts.extend(
            [
                first_spec_value(spec_pairs, ["type"]),
                first_spec_value(spec_pairs, ["model name", "model number", "model id"]),
                first_spec_value(spec_pairs, ["designed for", "compatible model"]),
                brand,
            ]
        )

    normalized_parts = [normalize_text(part) for part in key_parts if normalize_text(part)]
    deduped_parts: List[str] = []
    seen: Set[str] = set()
    for part in normalized_parts:
        if part not in seen:
            seen.add(part)
            deduped_parts.append(part)
    return " | ".join(deduped_parts) if deduped_parts else "unknown"


def build_core_group_key(root_category: str, leaf_category: str, spec_pairs: Dict[str, str]) -> str:
    category_group = root_category_group(root_category)
    key_parts = [root_category, leaf_category]

    if category_group == "clothing":
        key_parts.extend(
            [
                first_spec_value(spec_pairs, ["type"]),
                first_spec_value(spec_pairs, ["ideal for"]),
            ]
        )
    elif category_group == "jewellery":
        key_parts.extend(
            [
                first_spec_value(spec_pairs, ["type"]),
                first_spec_value(spec_pairs, ["base material", "material"]),
            ]
        )
    elif category_group == "mobile_accessories":
        key_parts.extend(
            [
                first_spec_value(spec_pairs, ["designed for", "compatible model", "model name"]),
                first_spec_value(spec_pairs, ["type"]),
            ]
        )
    elif category_group == "furniture_home":
        key_parts.extend(
            [
                first_spec_value(spec_pairs, ["type"]),
                first_spec_value(spec_pairs, ["material", "primary material"]),
            ]
        )
    else:
        key_parts.append(first_spec_value(spec_pairs, ["type"]))

    normalized_parts = [normalize_text(part) for part in key_parts if normalize_text(part)]
    return " | ".join(dict.fromkeys(normalized_parts)) if normalized_parts else "unknown"


def build_core_attribute_text(root_category: str, spec_pairs: Dict[str, str]) -> str:
    category_group = root_category_group(root_category)

    if category_group == "clothing":
        fields = ["type", "ideal for", "sleeve", "pattern", "fabric", "material", "size", "color", "shade"]
    elif category_group == "jewellery":
        fields = [
            "type",
            "base material",
            "material",
            "gemstone",
            "semi precious stone type",
            "ring size",
            "color",
            "shade",
        ]
    elif category_group == "mobile_accessories":
        fields = [
            "designed for",
            "compatible model",
            "model name",
            "model number",
            "model id",
            "type",
            "color",
            "shade",
        ]
    elif category_group == "furniture_home":
        fields = [
            "type",
            "material",
            "primary material",
            "width",
            "height",
            "length",
            "depth",
            "color",
            "shade",
        ]
    else:
        fields = ["type", "model name", "model number", "model id", "color", "shade", "size"]

    attribute_chunks = []
    for field in fields:
        value = spec_pairs.get(field, "")
        if value:
            attribute_chunks.append(f"{field} {value}")
    return " ".join(attribute_chunks).strip()


def select_candidate_indices(
    item_row: pd.Series,
    train_df: pd.DataFrame,
    min_candidates: int = MIN_CANDIDATE_POOL,
) -> np.ndarray:
    family_mask = (train_df["family_key"] == item_row["family_key"]) & (train_df["pid"] != item_row["pid"])
    family_indices = train_df.index[family_mask].to_numpy()
    if len(family_indices) >= TOP_K:
        return family_indices

    core_group_mask = (train_df["core_group_key"] == item_row["core_group_key"]) & (train_df["pid"] != item_row["pid"])
    core_group_indices = train_df.index[core_group_mask].to_numpy()
    if len(core_group_indices) >= TOP_K:
        return core_group_indices

    leaf_mask = (train_df["leaf_category"] == item_row["leaf_category"]) & (train_df["pid"] != item_row["pid"])
    leaf_indices = train_df.index[leaf_mask].to_numpy()
    if len(leaf_indices) >= max(TOP_K, min_candidates):
        return leaf_indices

    root_mask = (train_df["root_category"] == item_row["root_category"]) & (train_df["pid"] != item_row["pid"])
    root_indices = train_df.index[root_mask].to_numpy()
    if len(root_indices) > 0:
        return root_indices

    return train_df.index[train_df["pid"] != item_row["pid"]].to_numpy()


def attribute_match_score(item_row: pd.Series, candidate_row: pd.Series) -> float:
    category_group = root_category_group(item_row["root_category"])
    if category_group == "clothing":
        spec_keys = ["type", "ideal for", "sleeve", "pattern", "fabric", "material", "size", "color", "shade"]
    elif category_group == "jewellery":
        spec_keys = [
            "type",
            "base material",
            "material",
            "gemstone",
            "semi precious stone type",
            "ring size",
            "size",
            "color",
            "shade",
        ]
    elif category_group == "mobile_accessories":
        spec_keys = [
            "designed for",
            "compatible model",
            "model name",
            "model number",
            "model id",
            "type",
            "color",
            "shade",
        ]
    elif category_group == "furniture_home":
        spec_keys = ["type", "material", "primary material", "width", "height", "length", "depth", "color", "shade"]
    else:
        spec_keys = [
            "type",
            "model name",
            "model number",
            "model id",
            "size",
            "color",
            "shade",
        ]

    item_specs = item_row["spec_pairs"]
    candidate_specs = candidate_row["spec_pairs"]
    compared = 0
    matched = 0.0

    for key in spec_keys:
        item_value = item_specs.get(key, "")
        candidate_value = candidate_specs.get(key, "")
        if not item_value or not candidate_value:
            continue
        compared += 1
        if item_value == candidate_value:
            matched += 1.0
        elif item_value in candidate_value or candidate_value in item_value:
            matched += 0.5

    spec_score = (matched / compared) if compared else 0.0

    item_name_tokens = set(item_row["clean_name"].split())
    candidate_name_tokens = set(candidate_row["clean_name"].split())
    name_union = item_name_tokens | candidate_name_tokens
    name_overlap = (len(item_name_tokens & candidate_name_tokens) / len(name_union)) if name_union else 0.0

    if compared == 0:
        return name_overlap
    return (0.8 * spec_score) + (0.2 * name_overlap)


def category_match_score(item_row: pd.Series, candidate_row: pd.Series) -> float:
    if item_row["family_key"] == candidate_row["family_key"]:
        return 1.0
    if item_row["core_group_key"] == candidate_row["core_group_key"]:
        return 0.85
    if item_row["leaf_category"] == candidate_row["leaf_category"]:
        return 0.7
    if item_row["root_category"] == candidate_row["root_category"]:
        return 0.4
    return 0.0


def hybrid_similarity_score(item_row: pd.Series, candidate_row: pd.Series, cosine_score: float) -> float:
    attribute_score = attribute_match_score(item_row, candidate_row)
    category_score = category_match_score(item_row, candidate_row)
    return (0.60 * cosine_score) + (0.25 * attribute_score) + (0.15 * category_score)


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
    data["core_attribute_text"] = data.apply(
        lambda row: build_core_attribute_text(row["root_category"], row["spec_pairs"]),
        axis=1,
    )
    data["clean_core_attributes"] = data["core_attribute_text"].fillna("").apply(normalize_text)
    data["family_key"] = data.apply(
        lambda row: build_family_key(
            row["root_category"],
            row["leaf_category"],
            row["clean_brand"],
            row["spec_pairs"],
        ),
        axis=1,
    )
    data["core_group_key"] = data.apply(
        lambda row: build_core_group_key(
            row["root_category"],
            row["leaf_category"],
            row["spec_pairs"],
        ),
        axis=1,
    )

    data["combined_text"] = (
        data["family_key"]
        + " "
        + data["family_key"]
        + " "
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
        + data["clean_attributes"]
        + " "
        + data["spec_text"]
        + " "
        + data["clean_description"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    data["focused_text"] = (
        data["core_group_key"]
        + " "
        + data["core_group_key"]
        + " "
        + data["family_key"]
        + " "
        + data["family_key"]
        + " "
        + data["clean_leaf_category"]
        + " "
        + data["clean_name"]
        + " "
        + data["clean_name"]
        + " "
        + data["clean_core_attributes"]
        + " "
        + data["clean_core_attributes"]
        + " "
        + data["clean_core_attributes"]
        + " "
        + data["clean_brand"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    data = data[data["combined_text"].str.len() > 0].reset_index(drop=True)
    data = data.drop(columns=["category_parts", "sanitized_category_parts"])
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
    train_matrix = vectorizer.fit_transform(train_df["focused_text"])
    test_matrix = vectorizer.transform(test_df["focused_text"])
    return vectorizer, train_matrix, test_matrix


def recommend_from_train_item(
    item_row: pd.Series,
    train_df: pd.DataFrame,
    train_matrix,
    vectorizer: TfidfVectorizer,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    item_vector = vectorizer.transform([item_row["focused_text"]])
    candidate_indices = select_candidate_indices(item_row, train_df)
    candidate_scores = cosine_similarity(item_vector, train_matrix[candidate_indices]).ravel()
    candidate_df = train_df.iloc[candidate_indices].copy()
    candidate_df["cosine_score"] = candidate_scores
    candidate_df["attribute_match_score"] = candidate_df.apply(lambda row: attribute_match_score(item_row, row), axis=1)
    candidate_df["category_match_score"] = candidate_df.apply(lambda row: category_match_score(item_row, row), axis=1)
    candidate_df["hybrid_score"] = candidate_df.apply(
        lambda row: hybrid_similarity_score(item_row, row, row["cosine_score"]),
        axis=1,
    )
    candidate_df["similarity_score"] = candidate_df["hybrid_score"]
    candidate_df["is_strong_match"] = candidate_df["hybrid_score"] >= STRONG_MATCH_THRESHOLD
    candidate_df = candidate_df.sort_values("hybrid_score", ascending=False)
    return candidate_df.head(top_k)[
        [
            "product_name",
            "brand",
            "root_category",
            "leaf_category",
            "discounted_price",
            "cosine_score",
            "hybrid_score",
            "similarity_score",
            "is_strong_match",
        ]
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
    records: List[Dict[str, float]] = []

    for test_idx, item_row in test_df.iterrows():
        relevant_df = relevant_train_items(item_row, train_df)
        relevant_pids = set(relevant_df["pid"].tolist())
        if not relevant_pids:
            continue

        candidate_indices = select_candidate_indices(item_row, train_df)
        candidate_scores = cosine_similarity(test_matrix[test_idx], train_matrix[candidate_indices]).ravel()
        candidate_df = train_df.iloc[candidate_indices].copy()
        candidate_df["cosine_score"] = candidate_scores
        candidate_df["hybrid_score"] = candidate_df.apply(
            lambda row: hybrid_similarity_score(item_row, row, row["cosine_score"]),
            axis=1,
        )
        ranked_idx = np.argsort(candidate_df["hybrid_score"].to_numpy())[::-1]
        recommended_pids = []
        for idx in ranked_idx:
            candidate_pid = candidate_df.iloc[idx]["pid"]
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
