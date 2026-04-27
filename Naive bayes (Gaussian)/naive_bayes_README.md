# Naive Bayes (Gaussian)

> Implemented from scratch using NumPy only — benchmarked against Sklearn

---

## Table of Contents
- [Concept](#concept)
- [Bayes Theorem](#bayes-theorem)
- [The Naive Assumption](#the-naive-assumption)
- [Algorithm Walkthrough](#algorithm-walkthrough)
- [Gaussian Likelihood](#gaussian-likelihood)
- [Implementation](#implementation)
- [Results](#results)
- [Usage](#usage)
- [Files](#files)
- [Key Takeaways](#key-takeaways)

---

## Concept

Naive Bayes is a **probability based classifier** built on Bayes Theorem. Instead of finding distances like KNN or asking questions like Decision Trees, it asks:

> *"Given what I know about this data point, what is the probability it belongs to each class? Whichever class has the highest probability wins."*

It is called **Naive** because it makes one bold simplifying assumption — every feature is completely independent of every other feature. This is rarely true in reality, but the simplification makes the math clean and the algorithm surprisingly powerful.

---

## Bayes Theorem

```
P(class | features) ∝ P(class) * P(feature1 | class) * P(feature2 | class) * ...
```

| Term | Name | What it means |
|---|---|---|
| `P(class)` | Prior | How common is this class in the training data |
| `P(feature \| class)` | Likelihood | How probable is this feature value given this class |
| `P(class \| features)` | Posterior | What we want — probability of class given the data |

The normalizer `P(features)` is ignored because it is identical for every class and cannot change which class wins.

---

## The Naive Assumption

The word **Naive** refers to assuming every feature is independent:

```
P(f1, f2, f3 | class) = P(f1 | class) * P(f2 | class) * P(f3 | class)
```

Instead of computing one complex joint probability across all features, you multiply individual probabilities together — one per feature. This assumption is almost never strictly true, but Naive Bayes works remarkably well in practice even when features are correlated.

---

## Algorithm Walkthrough

### Training — two things computed and stored

**Step 1 — Prior probability**

How common is each class in the training data?

```
P(class) = count of samples in class / total samples
```

**Step 2 — Likelihood parameters**

For each class, for each feature, store the mean and variance of that feature's values within the class:

```python
for each class:
    rows = X_train[y_train == class]      # filter rows belonging to this class
    mean[class] = rows.mean(axis=0)       # one mean per feature
    var[class]  = rows.var(axis=0) + ε    # one variance per feature (+ epsilon)
```

The classes are **disjoint** — a row belonging to class 0 never appears in class 1 calculations. Each class gets its own separate mean and variance per feature.

### Prediction — posterior for each class

For a new data point, compute the log posterior for every class and pick the highest:

```
log P(class | x) = log P(class) + Σ log P(xᵢ | class)
```

Using logs prevents numerical underflow — multiplying many small decimals together can round to zero. Adding logs gives the same winner without that risk.

---

## Gaussian Likelihood

Since the Breast Cancer dataset has continuous features, we use the **Gaussian probability density function** to compute likelihoods:

$$P(x | class) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Where `μ` is the stored mean and `σ²` is the stored variance for that feature and class.

---

## Implementation

The implementation lives in `bayes.py` as a clean importable class mirroring sklearn's API.

### Key design decisions

**Disjoint class filtering with boolean indexing** — rows belonging to each class are filtered using `X_train[y_train == class]` — no manual loops needed. Each class gets a completely separate subset of the data for its mean and variance calculations.

**Variance smoothing** — a small epsilon `1e-9` is added to every variance to prevent division by zero when all values of a feature within a class are identical:
```python
var = np.var(rows, axis=0) + 1e-9
```

**Vectorized gaussian** — the gaussian function accepts full NumPy arrays for `x`, `mean`, and `var` — computing the probability for all features simultaneously in one call through NumPy broadcasting.

**Log probabilities** — multiplying many small probabilities causes numerical underflow. We take `np.log` of the gaussian outputs and sum them instead — turning multiplication into addition and keeping values numerically stable.

**`np.argmax` for prediction** — log posteriors are stored in an array, one per class. `np.argmax` finds the winning class index in one operation — no sorting, no tuple unpacking.

---

## Results

**Dataset:** Breast Cancer Wisconsin (569 samples, 30 features, 2 classes)
**Split:** 75% train / 25% test
**Classes:** Malignant (212), Benign (357)

```
=============================================
  Model                      Accuracy
  -----------------------------------
  Our Naive Bayes              0.9510
  Sklearn GaussianNB           0.9580
=============================================
```

### Classification Report

**Our Naive Bayes:**
```
              precision    recall  f1-score   support

   malignant       0.93      0.94      0.94        54
      benign       0.97      0.96      0.96        89

    accuracy                           0.95       143
   macro avg       0.95      0.95      0.95       143
weighted avg       0.95      0.95      0.95       143
```

**Sklearn GaussianNB:**
```
              precision    recall  f1-score   support

   malignant       0.94      0.94      0.94        54
      benign       0.97      0.97      0.97        89

    accuracy                           0.96       143
   macro avg       0.96      0.96      0.96       143
weighted avg       0.96      0.96      0.96       143
```

### Observations

Our implementation achieves **95.10%** accuracy vs sklearn's **95.80%** — a difference of only 0.70%. The small gap comes from minor floating point differences in how variance and log probabilities are handled internally.

Both models struggle slightly more with **malignant** classification (lower recall) than benign — which is expected since malignant tumors have more feature overlap with benign ones in this dataset.

---

## Usage

```python
from bayes import naive

# instantiate
model = naive()

# train — computes priors, means, and variances per class
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Files

| File | Purpose |
|---|---|
| `bayes.py` | Gaussian Naive Bayes class — NumPy only |
| `model_evaluation_comparison.ipynb` | Full visual walkthrough — comparison, classification report, confusion matrix |
| `README.md` | This file |

---

## Key Takeaways

- Naive Bayes is one of the fastest classifiers — training is just counting and averaging, prediction is just multiplication
- The independence assumption is almost always violated in real data — yet the model still performs remarkably well
- Gaussian Naive Bayes works well when features are continuous and roughly normally distributed
- Variance smoothing is essential — without epsilon, zero variance crashes the entire gaussian computation
- Log probabilities are not optional — raw probability multiplication causes underflow on any dataset with more than a handful of features
- The 0.70% gap vs sklearn is expected — a from scratch implementation will always have minor floating point differences vs a highly optimized library
