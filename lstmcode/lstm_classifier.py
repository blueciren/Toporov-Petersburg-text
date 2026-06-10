from __future__ import annotations

import argparse
import pickle
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def simple_tokenize(text: str) -> list[str]:
    """
    Very simple tokenizer:
    - lowercase
    - split words / numbers / punctuation
    """
    text = text.lower()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def read_txt_files(folder: Path) -> list[str]:
    """Read all .txt files recursively."""
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
    """
    Extract text from XML.
    Priority:
    1) <novel><text>...</text></novel>
    2) any <text> node
    3) all text in the XML
    """
    root = ET.parse(xml_path).getroot()

    if root.tag == "novel":
        text_node = root.find("text")
        if text_node is not None:
            return normalize_whitespace(" ".join(text_node.itertext()))

    text_node = root.find(".//text")
    if text_node is not None:
        return normalize_whitespace(" ".join(text_node.itertext()))

    return normalize_whitespace(" ".join(root.itertext()))


def read_xml_files(folder: Path) -> list[str]:
    """Read all .xml files recursively."""
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


def build_dataset(
    general_txt_dir: Path,
    topic_txt_dir: Path,
    general_xml_dir: Path,
    topic_xml_dir: Path,
) -> tuple[list[str], list[int]]:
    general_txts = read_txt_files(general_txt_dir)
    topic_txts = read_txt_files(topic_txt_dir)
    general_xmls = read_xml_files(general_xml_dir)
    topic_xmls = read_xml_files(topic_xml_dir)

    general_texts = general_txts + general_xmls
    topic_texts = topic_txts + topic_xmls

    if not general_texts:
        raise ValueError("No usable general texts found.")
    if not topic_texts:
        raise ValueError("No usable topic texts found.")

    x = general_texts + topic_texts
    y = [0] * len(general_texts) + [1] * len(topic_texts)

    print(f"general txt: {len(general_txts)}")
    print(f"general xml: {len(general_xmls)}")
    print(f"topic txt  : {len(topic_txts)}")
    print(f"topic xml  : {len(topic_xmls)}")
    print(f"total general: {len(general_texts)}")
    print(f"total topic  : {len(topic_texts)}")

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


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

class Vocab:
    def __init__(self, stoi: dict[str, int], itos: list[str]):
        self.stoi = stoi
        self.itos = itos

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, text: str, max_len: int) -> list[int]:
        tokens = simple_tokenize(text)
        ids = [self.stoi.get(tok, self.unk_idx) for tok in tokens][:max_len]
        return ids

    def __len__(self) -> int:
        return len(self.itos)
    

def build_vocab(texts: list[str], min_freq: int = 5, max_vocab_size: int = 5000) -> Vocab:
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    itos = [PAD_TOKEN, UNK_TOKEN]

    sorted_items = sorted(
        [(tok, freq) for tok, freq in counter.items() if freq >= min_freq],
        key=lambda x: (-x[1], x[0])
    )

    for token, _ in sorted_items[: max_vocab_size - 2]:
        itos.append(token)

    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi, itos)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        token_ids = self.vocab.encode(text, self.max_len)
        length = len(token_ids)
        return {
            "input_ids": token_ids,
            "length": length,
            "label": label,
        }


def collate_fn(batch, pad_idx: int):
    input_ids = [item["input_ids"] for item in batch]
    lengths = [item["length"] for item in batch]
    labels = [item["label"] for item in batch]

    max_batch_len = max(lengths) if lengths else 1
    padded = []
    for seq in input_ids:
        padded_seq = seq + [pad_idx] * (max_batch_len - len(seq))
        padded.append(padded_seq)

    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 2,
        pad_idx: int = 0,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_size)

    def forward(self, input_ids, lengths):
        """
        input_ids: (batch, seq_len)
        lengths:   (batch,)
        """
        embedded = self.embedding(input_ids)  # (B, L, E)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)

        if self.bidirectional:
            # 마지막 layer의 forward/backward hidden concat
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            final_hidden = hidden[-1]

        logits = self.fc(self.dropout(final_hidden))
        return logits
    

def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return avg_loss, acc, f1, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--general_txt_dir", type=Path, required=True)
    parser.add_argument("--topic_txt_dir", type=Path, required=True)
    parser.add_argument("--general_xml_dir", type=Path, required=True)
    parser.add_argument("--topic_xml_dir", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, default=Path("./lstm_cls_output"))

    parser.add_argument("--max_len", type=int, default=400)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load data

    x, y = build_dataset(
        args.general_txt_dir,
        args.topic_txt_dir,
        args.general_xml_dir,
        args.topic_xml_dir,
    )

    x_train, x_valid, x_test, y_train, y_valid, y_test = split_dataset(
        x, y, random_state=args.seed
    )

    print(f"Train size: {len(x_train)}")
    print(f"Valid size: {len(x_valid)}")
    print(f"Test size : {len(x_test)}")

    # 2) Build vocab using train only
    vocab = build_vocab(x_train, min_freq=args.min_freq)
    print(f"Vocab size: {len(vocab)}")

    # Save vocab
    with open(args.save_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # 3) Datasets / loaders
    train_dataset = TextDataset(x_train, y_train, vocab, args.max_len)
    valid_dataset = TextDataset(x_valid, y_valid, vocab, args.max_len)
    test_dataset = TextDataset(x_test, y_test, vocab, args.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx),
    )

    # 4) Model
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=2,
        pad_idx=vocab.pad_idx,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5) Train
    best_valid_f1 = -1.0
    best_model_path = args.save_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        valid_loss, valid_acc, valid_f1, _, _ = evaluate(
            model, valid_loader, criterion, device
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} valid_f1={valid_f1:.4f}"
        )

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved to {best_model_path}")

    # 6) Test
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )

    print("\n=== Test Results ===")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc ={test_acc:.4f}")
    print(f"test_f1  ={test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    print(f"\nSaved outputs to: {args.save_dir}")


if __name__ == "__main__":
    main()