from __future__ import annotations

import csv
import re
from pathlib import Path
import xml.etree.ElementTree as ET
 # 스탑워즈 제거 용으로 사용함

def load_stopwords(csv_path: Path) -> set[str]:
    """Load stopwords from every non-empty cell in a CSV file."""
    stopwords: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                word = cell.strip().lower()
                if word:
                    stopwords.add(word)
    return stopwords


def extract_text(xml_path: Path) -> str:
    """Extract all text contained inside the <text> tag."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    text_node = root.find("text")
    if text_node is None:
        return ""

    pieces = [part.strip() for part in text_node.itertext() if part.strip()]
    return " ".join(pieces)


def remove_stopwords(text: str, stopwords: set[str]) -> str:
    """Remove all stopwords as whole words and normalize spacing."""
    if not stopwords:
        return text.strip()

    pattern = re.compile(
        r"\b(?:"
        + "|".join(re.escape(word) for word in sorted(stopwords, key=len, reverse=True))
        + r")\b",
        flags=re.IGNORECASE,
    )
    cleaned = pattern.sub("", text)

    # 1) 문장부호 제거
    cleaned = re.sub(r"[,.!?;:\-\u2026]", " ", cleaned)  # , . ! ? ; : - … 를 공백으로 치환

    # 2) 공백 정리
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def process_folder(folder: Path, stopwords_file: str = "stopwords.csv") -> None:
    csv_path = folder / stopwords_file
    if not csv_path.exists():
        raise FileNotFoundError(f"Stopwords file not found: {csv_path}")

    stopwords = load_stopwords(csv_path)

    for xml_path in sorted(folder.glob("*.xml")):
        extracted = extract_text(xml_path)
        filtered = remove_stopwords(extracted, stopwords)
        output_path = xml_path.with_suffix(".txt")
        output_path.write_text(filtered, encoding="utf-8")
        print(f"Saved: {output_path.name}")


if __name__ == "__main__":
    process_folder(Path.cwd() / "idiot")
