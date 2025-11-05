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

We clean and process this data to generate **feature text** and embeddings.

### Matching Logic

We generate embeddings from text features like:

```
name + age + sex + country + sport + event
```

Then we compute **cosine similarity** between athletes & event vectors.

**Pipeline:**

1. Preprocess data
2. Create feature text for athletes & events
3. Generate embeddings (OpenAI model)
4. Save embeddings to disk
5. Build graph in Neo4j (athletes, sports, events relationships)
6. Recommend events by similarity search

---

## ⚙️ Local Setup Guide

Follow these steps to run SimuMatch on your machine.

### ✅ Requirements

* Python 3.9+
* Neo4j Desktop / Aura DB
* Virtual environment
* Git

---

### 🛠️ Setup

#### 1️⃣ Clone the repo

```bash
git clone https://github.com/yourusername/SimuMatch.git
cd SimuMatch
```

#### 2️⃣ Create & activate virtual env

```bash
python3 -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate      # windows
```

#### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Add your `.env` file

Create `.env`:

```
OPENAI_API_KEY=your_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---


### Run Pipeline

#### Generate embeddings

```bash
python src/embeddings/generate_embeddings.py
```

#### Build graph

```bash
python -m src.graph.build_graph
```

#### Run event recommender

```python
from src.matching.match_engine import recommend_events
print(recommend_events("Usain Bolt"))
```

---

## 🎯 Output

Example Response:

```
Top recommended events for Usain Bolt:
- 100m Sprint
- 200m Sprint
- 4×100m Relay
```



