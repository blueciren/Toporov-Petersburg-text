import joblib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

model = joblib.load("topic_classifier.joblib")
tfidf = model.named_steps["tfidf"]
clf = model.named_steps["clf"]

feats = list(tfidf.get_feature_names_out())
print("n_features:", len(feats))
print("ngram_range:", tfidf.ngram_range, "min_df:", tfidf.min_df, "max_features:", tfidf.max_features)
print("clf C:", clf.C)
print("sample features:", feats[:30])

def extract_novel_text(xml_path):
    root = ET.parse(xml_path).getroot()
    if root.tag != "novel":
        return ""
    text_node = root.find("text")
    if text_node is None:
        return ""
    return " ".join(text_node.itertext())

corpus_words = set()
n_docs = 0
for folder in ["general", "topic"]:
    for f in sorted(Path(folder).glob("*.xml")):
        txt = extract_novel_text(f)
        if txt:
            n_docs += 1
            for w in re.findall(r"\w+", txt.lower()):
                corpus_words.add(w)

print("n_docs (general+topic xml):", n_docs)
print("corpus vocab size:", len(corpus_words))

# unigram features only for fair comparison
unigram_feats = [f for f in feats if " " not in f]
overlap = sum(1 for f in unigram_feats if f in corpus_words)
print(f"unigram features: {len(unigram_feats)}, overlap with general+topic vocab: {overlap} ({100*overlap/len(unigram_feats):.1f}%)")
