# K-Nearest Neighbors (KNN)

> Implemented from scratch using NumPy only — benchmarked against Sklearn

---

## Table of Contents
- [Concept](#concept)
- [Algorithm Walkthrough](#algorithm-walkthrough)
- [Distance Metric](#distance-metric)
- [The Role of K](#the-role-of-k)
- [Implementation](#implementation)
- [Results](#results)
- [Usage](#usage)
- [Files](#files)
- [Key Takeaways](#key-takeaways)

---

## Concept

> *"Tell me who your friends are, and I will tell you who you are."*

This proverb is the entire philosophy of KNN. A new data point is classified by looking at the K closest points in the training data — its nearest neighbors — and letting them vote. Whatever class most of them belong to, that is the prediction.

KNN is a **lazy learner** — it performs zero computation during training. It simply memorizes the entire training dataset and defers all work to prediction time.

---

## Algorithm Walkthrough

Prediction happens in four steps:

**Step 1 — Memorize**
During `fit()`, the model stores the training features `X_train` and labels `y_train`. Nothing else happens.

**Step 2 — Compute Distances**
For each new point, compute the Euclidean distance to every training point. This produces one distance value per training sample.

**Step 3 — Find K Nearest Neighbors**
Sort the distances and identify the K training points with the smallest distances — the K nearest neighbors.

**Step 4 — Majority Vote**
Look up the class labels of those K neighbors and return whichever class appears most frequently as the prediction.

---

## Distance Metric

We use **Euclidean distance** (L2 norm) — the straight line distance between two points in feature space:

$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

In NumPy this is computed in a single vectorized line across all training points at once:

```python
distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
```

The `axis=1` ensures we sum across features per row — giving one distance per training point rather than one global sum.

---

## The Role of K

K is the only hyperparameter in KNN. It controls how many neighbors vote on each prediction.

| K Value | Effect |
|---|---|
| K = 1 | Very sensitive — every point decides its own neighborhood. Prone to overfitting |
| K = 3-5 | Good balance for most problems |
| K = large | Smoother decision boundary — may underfit |
| K = even | Avoid in binary classification — can cause ties |

---

## Implementation

The implementation lives in `knn.py` as a clean, importable class that mirrors sklearn's API — `fit()`, `predict()`, `score()`.

### Key design decisions

**Vectorized distance computation** — instead of looping over training points in Python, we subtract the entire `X_train` matrix from the single test point `x` using NumPy broadcasting. This computes all distances simultaneously at C speed.

**`argsort` over sorting pairs** — instead of bundling `(distance, label)` tuples and sorting them, we use `np.argsort()` to get the indices that would sort the distances. We then use those indices to look up labels directly — cleaner and faster.

**Index alignment** — `X_train` and `y_train` are stored as separate arrays aligned by index. The K nearest indices retrieved from distances are used directly to index into `y_train` — no label carrying required.

---

## Results

**Dataset:** Iris (150 samples, 4 features, 3 classes)  
**Split:** 80% train / 20% test  
**K:** 3

```
=============================================
  Model                      Accuracy
  -----------------------------------
  Our KNN                      1.0000
  Sklearn KNN                  1.0000
=============================================

  ✅ Predictions match exactly — implementation verified!
```

### Confusion Matrix

Both models produced identical confusion matrices — all predictions on the diagonal, zero misclassifications:

```
                setosa   versicolor   virginica
setosa            15          0           0
versicolor         0         11           0
virginica          0          0          12
```

### K Sweep

Both models track identically at every K value. Notice K=7 produces a small dip to 0.9667 — a reminder that K is a meaningful hyperparameter and the right value depends on the data:

| K | Our KNN | Sklearn |
|---|---|---|
| 1 | 1.0000 | 1.0000 |
| 3 | 1.0000 | 1.0000 |
| 5 | 1.0000 | 1.0000 |
| 7 | 1.0000 | 1.0000 |
| 9 | 1.0000 | 1.0000 |
| 11 | 1.0000 | 1.0000 |
| 13 | 1.0000 | 1.0000 |
| 15 | 1.0000 | 1.0000 |



## Usage

```python
from knn import KNN

# instantiate
model = KNN(k=3)

# train — just memorizes the data
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Files

| File | Purpose |
|---|---|
| `knn.py` | KNN class — NumPy only, no sklearn |
| `model_evaluation.ipynb` | Full visual walkthrough — comparison, confusion matrix, K sweep, decision boundary |
| `README.md` | This file |

---

## Key Takeaways

- KNN is simple but powerful — no assumptions about data distribution, no training phase
- The entire algorithm is just distance computation + sorting + voting
- Main cost is at **prediction time** O(n·d) per point — slow on large datasets
- Feature scaling matters — always normalize before using KNN since large-scale features dominate distances
- 100% accuracy on Iris reflects how clean and separable the dataset is — not a guarantee of real world performance
