"""
Error Analysis Pipeline for Text Classification

This script demonstrates:
1. Error detection
2. Error categorization
3. Failure pattern analysis
4. Slice-based evaluation
5. Confusion matrix
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# -------------------------------
# 1. Sample dataset (replace with real data)
# -------------------------------
data = {
    "text": [
        "I love this product!",
        "This is not good at all",
        "Absolutely terrible service",
        "Not bad actually",
        "Worst experience ever",
        "I am happy with this",
        "I don’t think this is great",
        "Amazing quality!",
        "meh it was okay",
        "I wouldn’t recommend this"
    ],
    "true_label": [1, 0, 0, 1, 0, 1, 0, 1, 0, 0],  # 1=positive, 0=negative
    "predicted": [1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    "confidence": [0.95, 0.80, 0.90, 0.60, 0.85, 0.92, 0.70, 0.96, 0.55, 0.75]
}

df = pd.DataFrame(data)

# -------------------------------
# 2. Identify errors
# -------------------------------
df["is_error"] = df["true_label"] != df["predicted"]
errors = df[df["is_error"]].copy()

print("=" * 50)
print("TOTAL ERRORS:", len(errors))
print("=" * 50)
print(errors[["text", "true_label", "predicted"]])


# -------------------------------
# 3. Error categorization (rule-based)
# -------------------------------
def categorize_error(text):
    text_lower = text.lower()

    if "not" in text_lower or "n't" in text_lower:
        return "negation_error"
    elif any(word in text_lower for word in ["meh", "okay"]):
        return "ambiguous_sentiment"
    elif len(text.split()) < 3:
        return "short_text"
    else:
        return "other"


errors["error_type"] = errors["text"].apply(categorize_error)

print("\n" + "=" * 50)
print("ERROR CATEGORIES")
print("=" * 50)
print(errors[["text", "error_type"]])


# -------------------------------
# 4. Error distribution (failure patterns)
# -------------------------------
error_distribution = errors["error_type"].value_counts(normalize=True) * 100

print("\n" + "=" * 50)
print("ERROR DISTRIBUTION (%)")
print("=" * 50)
print(error_distribution)


# -------------------------------
# 5. Slice-based analysis
# -------------------------------

# ---- By text length ----
df["text_length"] = df["text"].apply(lambda x: len(x.split()))
length_analysis = df.groupby("text_length")["is_error"].mean()

print("\n" + "=" * 50)
print("ERROR RATE BY TEXT LENGTH")
print("=" * 50)
print(length_analysis)


# ---- By confidence bins ----
df["confidence_bin"] = pd.cut(df["confidence"], bins=[0, 0.6, 0.8, 1.0])
confidence_analysis = df.groupby("confidence_bin")["is_error"].mean()

print("\n" + "=" * 50)
print("ERROR RATE BY CONFIDENCE")
print("=" * 50)
print(confidence_analysis)


# -------------------------------
# 6. Inspect examples per error type
# -------------------------------
print("\n" + "=" * 50)
print("ERROR EXAMPLES BY CATEGORY")
print("=" * 50)

for error_type in errors["error_type"].unique():
    print(f"\n--- {error_type.upper()} ---")
    samples = errors[errors["error_type"] == error_type]["text"].head(3)
    for s in samples:
        print("-", s)


# -------------------------------
# 7. Confusion Matrix
# -------------------------------
cm = confusion_matrix(df["true_label"], df["predicted"])

print("\n" + "=" * 50)
print("CONFUSION MATRIX")
print("=" * 50)
print(cm)


# -------------------------------
# 8. Summary Insights (simple auto-summary)
# -------------------------------
print("\n" + "=" * 50)
print("SUMMARY INSIGHTS")
print("=" * 50)

if len(error_distribution) > 0:
    top_error = error_distribution.idxmax()
    print(f"Most common error type: {top_error}")

high_conf_errors = df[(df["confidence"] > 0.8) & (df["is_error"])]
print(f"High-confidence errors: {len(high_conf_errors)}")

print("\nPipeline completed successfully.")