from __future__ import annotations

import argparse
import math
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def read_general_txt_files(folder: Path) -> list[str]:
    """Read all .txt files recursively for class 0 (general)."""
    files = sorted(folder.rglob("*.txt"))
    texts: list[str] = []

    for file in files:
        content = normalize_whitespace(
            file.read_text(encoding="utf-8", errors="ignore")
        )
        if content:
            texts.append(content)

    return texts


def extract_novel_text(xml_path: Path) -> str:
    """Extract text from <novel><text>...</text></novel> in an XML file."""
    root = ET.parse(xml_path).getroot()
    if root.tag != "novel":
        return ""

    text_node = root.find("text")
    if text_node is None:
        return ""

    extracted = normalize_whitespace(" ".join(text_node.itertext()))
    return extracted


def read_topic_xml_novels(folder: Path) -> list[str]:
    """Read all .xml files recursively and extract class 1 topic text."""
    files = sorted(folder.rglob("*.xml"))
    texts: list[str] = []

    for file in files:
        try:
            content = extract_novel_text(file)
        except ET.ParseError:
            continue

        if content:
            texts.append(content)

    return texts


def read_prediction_document(path: Path) -> str:
    """Read a prediction input document from plain text file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def prediction_explanation(model: Pipeline, text: str, top_k: int = 10) -> dict[str, object] | None:
    """
    Explain a prediction for linear models by showing:
    - decision score (logit)
    - probability from logit
    - top positive/negative feature contributions
    """
    tfidf = model.named_steps.get("tfidf")
    clf = model.named_steps.get("clf")
    if tfidf is None or clf is None:
        return None
    if not hasattr(clf, "coef_") or not hasattr(clf, "intercept_"):
        return None

    vector = tfidf.transform([text])
    if vector.shape[0] == 0:
        return None

    row = vector.getrow(0)
    indices = row.indices
    values = row.data

    if len(indices) == 0:
        logit = float(clf.intercept_[0])
        return {
            "logit": logit,
            "prob_from_logit": 1.0 / (1.0 + math.exp(-logit)),
            "positive": [],
            "negative": [],
        }

    coef = clf.coef_[0]
    feature_names = tfidf.get_feature_names_out()
    contrib = values * coef[indices]

    pairs = list(zip(indices, contrib))
    pos = sorted([p for p in pairs if p[1] > 0], key=lambda x: x[1], reverse=True)[:top_k]
    neg = sorted([p for p in pairs if p[1] < 0], key=lambda x: x[1])[:top_k]

    logit = float(clf.intercept_[0] + contrib.sum())
    return {
        "logit": logit,
        "prob_from_logit": 1.0 / (1.0 + math.exp(-logit)),
        "positive": [(feature_names[i], float(v)) for i, v in pos],
        "negative": [(feature_names[i], float(v)) for i, v in neg],
    }


def print_explanation(model: Pipeline, text: str, top_k: int) -> None:
    exp = prediction_explanation(model, text, top_k=top_k)
    if exp is None:
        print("  [explain] Explanation unavailable for this model type.")
        return

    print(
        f"  [explain] decision_score(logit)={exp['logit']:.6f}, "
        f"prob_from_logit={exp['prob_from_logit']:.6f}"
    )
    print("  [explain] top positive features (toward class 1):")
    for feat, val in exp["positive"]:
        print(f"    + {feat}: {val:.6f}")
    print("  [explain] top negative features (toward class 0):")
    for feat, val in exp["negative"]:
        print(f"    - {feat}: {val:.6f}")


def build_dataset(general_dir: Path, topic_dir: Path) -> tuple[list[str], list[int]]:
    general_texts = read_general_txt_files(general_dir)
    topic_texts = read_topic_xml_novels(topic_dir)

    if not general_texts:
        raise ValueError(f"No usable .txt files found in: {general_dir}")
    if not topic_texts:
        raise ValueError(
            f"No usable XML files with <novel><text> found in: {topic_dir}"
        )

    x = general_texts + topic_texts
    y = [0] * len(general_texts) + [1] * len(topic_texts)
    return x, y


def split_dataset(
    x: list[str], y: list[int], random_state: int = 42
) -> tuple[list[str], list[str], list[str], list[int], list[int], list[int]]:
    """Split dataset into train/valid/test = 8/1/1 with stratification."""
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 3),
                    min_df=3,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, solver="liblinear"),
            ),
        ]
    )


def evaluate(name: str, y_true: list[int], y_pred: list[int]) -> None:
    print(f"\n[{name}] metrics")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


def train(args: argparse.Namespace) -> None:
    x, y = build_dataset(args.general_dir, args.topic_dir)
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_dataset(
        x, y, random_state=args.random_state
    )

    print(
        f"Loaded documents: total={len(x)}, train={len(x_train)}, valid={len(x_valid)}, test={len(x_test)}"
    )

    pipeline = make_pipeline()

    param_grid = {
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "tfidf__min_df": [2, 3, 5],
        "tfidf__max_features": [3000, 5000],
        "clf__C": [0.1, 1.0, 3.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)

    print("\nBest CV params:", search.best_params_)
    print(f"Best CV F1: {search.best_score_:.4f}")

    best_model: Pipeline = search.best_estimator_

    valid_pred = best_model.predict(x_valid)
    evaluate("Validation", y_valid, valid_pred)

    test_pred = best_model.predict(x_test)
    evaluate("Test", y_test, test_pred)

    if args.model_out:
        with args.model_out.open("wb") as f:
            pickle.dump(best_model, f)
        print(f"\nSaved model: {args.model_out}")


def predict(args: argparse.Namespace) -> None:
    with args.model_path.open("rb") as f:
        model: Pipeline = pickle.load(f)

    if args.text is not None:
        input_text = args.text
        label = int(model.predict([input_text])[0])
        line = f"text_input => class={label}"
        if hasattr(model.named_steps["clf"], "predict_proba"):
            prob = model.predict_proba([input_text])[0][1]
            line += f" (topic=1 probability: {prob:.4f})"
        print(line)
        if args.explain:
            print_explanation(model, input_text, args.top_k)
        return
    if args.file is not None:
        input_text = read_prediction_document(args.file)
        label = int(model.predict([input_text])[0])
        line = f"{args.file} => class={label}"
        if hasattr(model.named_steps["clf"], "predict_proba"):
            prob = model.predict_proba([input_text])[0][1]
            line += f" (topic=1 probability: {prob:.4f})"
        print(line)
        if args.explain:
            print_explanation(model, input_text, args.top_k)
        return

    files = sorted(args.dir.rglob("*.txt")) if args.recursive else sorted(args.dir.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in directory: {args.dir}")

    docs = [read_prediction_document(p) for p in files]
    preds = model.predict(docs)

    probas = None
    if hasattr(model.named_steps["clf"], "predict_proba"):
        probas = model.predict_proba(docs)

    print(f"Directory prediction results ({len(files)} files):")
    for idx, path in enumerate(files):
        line = f"{path} => class={int(preds[idx])}"
        if probas is not None:
            line += f" (topic=1 probability: {probas[idx][1]:.4f})"
        print(line)
        if args.explain:
            print_explanation(model, docs[idx], args.top_k)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate a 2-class text topic classifier "
            "(class 0: TXT general, class 1: XML <novel><text>) and predict unknown text labels."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate model")
    train_parser.add_argument(
        "--general-dir",
        type=Path,
        required=True,
        help="Folder for class 0 TXT files (general corpus)",
    )
    train_parser.add_argument(
        "--topic-dir",
        type=Path,
        required=True,
        help="Folder for class 1 XML files (<novel><text>...)",
    )
    train_parser.add_argument("--model-out", type=Path, default=Path("topic_classifier.joblib"))
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.set_defaults(func=train)

    pred_parser = subparsers.add_parser("predict", help="Predict class for unknown text")
    pred_parser.add_argument("--model-path", type=Path, required=True)
    src = pred_parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Raw text to classify")
    src.add_argument("--file", type=Path, help="Path to text file to classify")
    src.add_argument("--dir", type=Path, help="Directory of .txt files to classify in batch")
    pred_parser.add_argument(
        "--recursive",
        action="store_true",
        help="When used with --dir, recursively include .txt files in subdirectories",
    )
    pred_parser.add_argument(
        "--explain",
        action="store_true",
        help="Show why the probability was produced (decision score and top feature contributions)",
    )
    pred_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of positive/negative contributing features to show with --explain",
    )
    pred_parser.set_defaults(func=predict)

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
