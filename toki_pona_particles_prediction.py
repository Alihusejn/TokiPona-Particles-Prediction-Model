import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay

# Reproduzierbarkeit
SEED = 42
np.random.seed(SEED)

df = pd.read_csv('tok_sentences.tsv', sep='\t', header=None, names=['id', 'lang', 'text'])
print(f'Datensatz geladen: {len(df)} Sätze')
df.head(10)

from huggingface_hub import hf_hub_download
from gensim.models import KeyedVectors

model_path = hf_hub_download(
    repo_id='finnnnnnnnnnnn/toki-pona-word2vec',
    filename='model.txt',
)
w2v = KeyedVectors.load_word2vec_format(model_path, binary=False)
print(f'Vokabular: {len(w2v)} Wörter, Vektorgröße: {w2v.vector_size}')
print(f'Ähnlichste Wörter zu "mi": {w2v.most_similar("mi", topn=5)}')

# Vorverarbeitung: Satzzeichen entfernen, Kleinbuchstaben
sentences = []
for text in df['text']:
    clean = re.sub(r'[^\w\s]', '', text.lower())
    words = clean.split()
    if words:
        sentences.append(words)
        
####################### Partikeln aus Trainingsbeispielen entfernen #####
PARTICLES = {'li', 'e', 'la', 'pi', 'o'}
WINDOW = 2
VEC_SIZE = w2v.vector_size
PAD_VEC = np.zeros(VEC_SIZE)

X_list = []
y_list = []

for words in sentences:
    for i, word in enumerate(words):
        if word in PARTICLES:
            # Kontextwörter sammeln
            context = []
            for j in range(i - WINDOW, i):
                context.append(words[j] if 0 <= j < len(words) else '<PAD>')
            for j in range(i + 1, i + 1 + WINDOW):
                context.append(words[j] if j < len(words) else '<PAD>')
            
            # Wörter in Vektoren umwandeln
            vecs = [w2v[w] if w in w2v else PAD_VEC for w in context]
            X_list.append(np.concatenate(vecs))  # 4 × 100 = 400 Features
            y_list.append(word)

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)

print(f'Beispiele: {len(X)}, Features: {X.shape[1]}')
print(f'Verteilung der Partikeln:')
for p, c in zip(*np.unique(y, return_counts=True)):
    print(f'  {p}: {c}')

####################### Daten in Train und Test aufteilen #######

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f'Train: {len(X_train)}, Test: {len(X_test)}')

####################### Decision Tree ###########################

dt = DecisionTreeClassifier(random_state=SEED)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
print(f'Decision Tree – Test F1: {f1_dt:.3f}')
print(classification_report(y_test, y_pred_dt, zero_division=0))

####################### Confusion Matrix - Decision Tree #########

labels = sorted(PARTICLES)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, labels=labels, cmap='Reds')
plt.title('Konfusionsmatrix – Decision Tree')
plt.show()

##################### Random Forest ##############################

rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
print(f'Random Forest – Test F1: {f1_rf:.3f}')
print(classification_report(y_test, y_pred_rf, zero_division=0))

##################### Confusijn Matrix für Random Forest #########

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, labels=labels, cmap='Reds')
plt.title('Konfusionsmatrix – Random Forest')
plt.show()

##################### Modelle Vergleichen ########################

print(f'Decision Tree F1: {f1_dt:.3f}')
print(f'Random Forest F1: {f1_rf:.3f}')
print(f'Besser: {"Decision Tree" if f1_dt > f1_rf else "Random Forest"}')

methods = ['Decision Tree', 'Random Forest']
scores = [f1_dt, f1_rf]
plt.bar(methods, scores, color=['steelblue', 'coral'])
plt.ylabel('F1-Score (weighted)')
plt.title('Vergleich: Decision Tree vs. Random Forest')
plt.ylim(0, 1)
for i, s in enumerate(scores):
    plt.text(i, s + 0.01, f'{s:.3f}', ha='center')
plt.show()

###################################################
# Cross-Validation auf einem Subset (schneller)
subset = 15000

cv_dt = cross_val_score(dt, X[:subset], y[:subset], cv=5)
cv_rf = cross_val_score(rf, X[:subset], y[:subset], cv=5, n_jobs=-1)

print(f'Decision Tree CV: {cv_dt.mean():.4f} (+/- {cv_dt.std():.4f})')
print(f'Random Forest CV: {cv_rf.mean():.4f} (+/- {cv_rf.std():.4f})')

plt.boxplot([cv_dt, cv_rf], labels=['Decision Tree', 'Random Forest'])
plt.ylabel('Accuracy')
plt.title('Cross-Validation (5-Fold)')
plt.show()
