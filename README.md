# 🔍 Semantic Search & Recommendation System Using Word2Vec

> **LPU Project 8 — Python & Full Stack Development**
> Course: Python & Full Stack Development | Lovely Professional University

---

## 📌 Project Overview

This project builds an **intelligent semantic search and article recommendation system** using **Word2Vec embeddings** trained from scratch on the **BBC News full-text dataset** (2,225 real English news articles across 5 categories). Unlike traditional keyword-based search, this system understands the *meaning* of text — finding relevant articles even when they don't share exact words.

---

## 🗂️ Dataset

**BBC Full-Text Document Classification** — available on Kaggle:
🔗 https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification

### Folder Structure (after download & extraction)
```
bbc-fulltext (document classification)/
├── business/
│   ├── 001.txt
│   ├── 002.txt
│   └── ...
├── tech/
├── sport/
├── entertainment/
└── politics/
```

Each `.txt` file contains one full news article. The dataset has **2,225 articles** across **5 categories**.

---

## 🧠 System Pipeline

```
BBC News .txt Files
        ↓
Text Preprocessing
(lowercase → remove punctuation → stopword filter → tokenize)
        ↓
Word2Vec Skip-Gram Training
(trained from scratch, 100 dimensions, window=5, epochs=100)
        ↓
Sentence Vectors via Mean Pooling
        ↓
    ┌───────────────────────────┐
    │                           │
Semantic Search          Recommendation
(query → cosine rank)    (article → find similar)
    │                           │
    └──────────┬────────────────┘
               ↓
    PCA + t-SNE + Heatmap Visualization
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Embedding Model | Word2Vec (Gensim) — Skip-Gram |
| Similarity Metric | Cosine Similarity (scikit-learn) |
| Dimensionality Reduction | PCA, t-SNE (scikit-learn) |
| Visualization | Matplotlib, Seaborn |
| Data Handling | Pandas, NumPy |

---

## 📁 File Structure

```
project/
├── LPU_Project8_Semantic_Search.ipynb   ← Main notebook (run this)
├── README.md                            ← This file
├── bbc-fulltext (document classification)/
│   ├── business/  001.txt ... 510.txt
│   ├── tech/      001.txt ... 401.txt
│   ├── sport/     001.txt ... 511.txt
│   ├── entertainment/ 001.txt ... 386.txt
│   └── politics/  001.txt ... 417.txt
└── outputs/                             ← Auto-generated after running
    ├── pca_visualization.png
    ├── tsne_visualization.png
    ├── similarity_heatmap.png
    └── word_similarity_bars.png
```

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project
```bash
git clone https://github.com/srinivasgupth81/embedding.git
cd lpu-semantic-search
```

### 2. Install dependencies
```bash
pip install gensim scikit-learn matplotlib seaborn pandas numpy
```

Or run the first cell in the notebook — it installs everything automatically.

### 3. Download the dataset
- Go to: https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification
- Click **Download** and extract the zip
- Place the extracted folder next to the notebook (see File Structure above)

### 4. Open the notebook
```bash
jupyter notebook main.ipynb
```
Run all cells top to bottom (`Kernel → Restart & Run All`).

> **No dataset?** The notebook includes a 100-article built-in fallback — it will run completely even without the Kaggle download.

---

## 🔎 Features

### Module 1 — Data Loading & Preprocessing
- Auto-detects the BBC folder structure and reads all `.txt` files
- Lowercases text, removes punctuation, filters 80+ stopwords
- Falls back gracefully to a built-in sample if dataset not found

### Module 2 — Word2Vec Embedding Generation
- Trains a **Skip-Gram Word2Vec** model from scratch on the corpus
- 100-dimensional vectors, window size 5, 100 training epochs
- Generates sentence-level vectors via **mean pooling** of word vectors

### Module 3A — Semantic Search
- Accepts a plain English query string
- Encodes query using the same Word2Vec pipeline
- Ranks all articles by **cosine similarity** and returns top-N with scores

### Module 3B — Recommendation System
- Given any article index, finds the most semantically similar articles
- Excludes the seed article itself from results

### Module 4 — Visualization
- **PCA plot** — 2D scatter of all articles coloured by category
- **t-SNE plot** — non-linear cluster visualization
- **Cosine similarity heatmap** — 20×20 pairwise similarity matrix
- **Word similarity bar charts** — top-10 nearest words for key terms

### Module 5 — Evaluation
- Computes **intra-category** vs **inter-category** average similarity
- Higher intra + lower inter = well-separated semantic clusters

---

## 📊 Sample Output

```
🔎 Query: 'smartphone internet mobile technology'
─────────────────────────────────────────────────
  #1 [TECH] (score=0.9312)
     Mobile phone sales hit record high as smartphone manufacturers compete...
  #2 [TECH] (score=0.9187)
     5G network rollout accelerates as telecoms operators compete...
  #3 [TECH] (score=0.8934)
     Broadband speeds slow as demand for internet access rises...
```

---

## 📈 Evaluation Metrics

| Category | Intra-Similarity | Inter-Similarity | Separation |
|---|---|---|---|
| tech | higher | lower | positive |
| sport | higher | lower | positive |
| business | higher | lower | positive |
| politics | moderate | lower | positive |
| entertainment | moderate | lower | positive |

---

## 🆚 How This Differs From the Keyword Approach

| Feature | Keyword Search | This System (Word2Vec) |
|---|---|---|
| Matching method | Exact word match | Semantic meaning |
| Handles synonyms | ❌ No | ✅ Yes |
| Context awareness | ❌ No | ✅ Yes |
| Dataset | Synthetic Hindi (80 rows) | Real BBC English (2,225 articles) |
| Visualization | None | PCA + t-SNE + Heatmap |
| Recommendation | None | ✅ Article-to-article |

---

## 📋 Notebook Steps at a Glance

| Step | Description |
|---|---|
| 0 | Install libraries |
| 1 | Load BBC dataset from folder structure |
| 2 | Text preprocessing |
| 3 | Train Word2Vec Skip-Gram model |
| 4 | Explore word vectors (most similar words) |
| 5 | Generate sentence vectors via mean pooling |
| 6 | Semantic search with cosine similarity |
| 7 | Article-to-article recommendation |
| 8 | PCA 2D visualization |
| 9 | t-SNE 2D visualization |
| 10 | Cosine similarity heatmap |
| 11 | Word similarity bar charts |
| 12 | Intra vs inter-category evaluation |
| 13 | Full pipeline summary |

---
