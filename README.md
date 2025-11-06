# SimuMatch

SimuMatch is an AI-powered system designed to match athletes to sports events based on their skill profile, performance attributes, and historical data patterns. The goal is to create a recommendation engine that can identify the most suitable events for athletes — useful for sports analytics, training guidance, scouting, and talent mapping.

---

## 🚀 Vision

Enable intelligent athlete-event matchmaking using machine learning + graph intelligence:

* Provide event recommendations based on athlete attributes
* Analyze athlete similarities & career paths
* Build a knowledge graph for explainability
* Create tools useful for coaches, analysts, and sports science research

---

## 📦 Data Used

We currently use **Olympic athlete & event datasets** (Kaggle / open-source) containing:

* Athlete ID, Name, Age
* Country / Team
* Sport & Event
* Physical data: Height, Weight

Dataset link:  
https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results

We clean and process this data to generate **feature text** and embeddings.

---

## 🧠 Matching Logic

We generate embeddings from text features like:

```

name + age + sex + country + sport + event

````

Then compare athlete → event similarity using **cosine similarity**.

**Pipeline Flow:**
1. Preprocess data
2. Create combined feature text
3. Generate embeddings (SentenceTransformer model)
4. Save embeddings locally (not version-controlled)
5. (Optional) Build graph in Neo4j
6. Recommend events using similarity search

---

## ⚙️ Local Setup Guide

### ✅ Requirements
* Python 3.9+
* Git
* Virtual environment
* Neo4j (optional for now)

---

### 🛠️ Setup

#### 1. Clone the repo
```bash
git clone https://github.com/shauryayay/SimuMatch.git
cd SimuMatch
````

#### 2. Ensure data folders exist

```bash
mkdir -p data/raw
mkdir -p data/processed
```

> Note: `data/processed/` is intentionally **not** stored in Git.
> Everyone generates their own embedded data locally.

#### 3. Create & activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # mac/linux
venv\Scripts\activate          # windows
```

#### 4. Install dependencies

⚠️ **IMPORTANT: install PyTorch separately (CPU version)**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

### Generate Embeddings (must do before using the recommender)

```bash
python src/matching/vector_search.py
python src/matching/event_embeddings.py
```

This will create:

```
data/processed/athletes_with_embeddings.csv
data/processed/events_with_embeddings.csv
data/athlete_vectors.npy
data/event_vectors.npy
```

---

### Run Recommender

```python
from src.matching.match_engine import recommend_events
print(recommend_events("Usain Bolt"))
```

---

### (Optional) Build the Knowledge Graph in Neo4j

```bash
python -m src.graph.build_graph
```

or

```bash
python src/graph/build_graph.py
```

---

## 🎯 Example Output

```
Top recommended events for Usain Bolt:
- 100m Sprint
- 200m Sprint
- 4×100m Relay
```

