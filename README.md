# Toki Pona Particles Prediction Model
Predicting constructed language Toki Pona grammatical particles (`li`, `e`, `la`, `pi`, `o`) using Decision Tree and Random Forest, with Word2Vec embeddings as features.

## Task

A particle is removed from a sentence. The model predicts which particle belongs in the gap based on the surrounding words.

**Example:** `jan ___ moku e kili` → Prediction: `li`

## Data

- **Dataset Source:** https://tatoeba.org/en/downloads
- **Size:** ~75,000 sentences, resulting in 158,503 training examples
- **Features:** Word2Vec vectors (pretrained from [HuggingFace](https://huggingface.co/finnnnnnnnnnnn/toki-pona-word2vec)), 2 words left + 2 words right = 400 features

## Models and Results

| Model | F1-Score (weighted) |
|---|---|
| Decision Tree | 0.888 |
| Random Forest | 0.930 |

## Usage

```bash
pip install gensim huggingface_hub scikit-learn
python toki_pona_particles_prediction.py
```

## Dependencies

- Python 3.11+
- scikit-learn
- gensim
- huggingface_hub
- matplotlib
- pandas, numpy
