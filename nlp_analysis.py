"""
NLP Sentiment Analysis
----------------------
Loads the scraped YouTube comments dataset and performs sentiment analysis
using HuggingFace Transformers (DistilBERT model).
"""

import pandas as pd
import os

try:
    from transformers import pipeline
except ImportError:
    print("Installing transformers and torch...")
    os.system("pip install transformers torch")
    from transformers import pipeline


def load_dataset(filename="dataset.csv"):
    """Load the scraped comments dataset."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} comments from {filename}")
    return df


def run_sentiment_analysis(df):
    """
    Run sentiment analysis on each comment using DistilBERT.
    Adds 'sentiment' and 'confidence' columns to the DataFrame.
    """
    print("\nLoading sentiment analysis model (first run downloads ~260MB)...")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    sentiments = []
    confidences = []
    total = len(df)

    print(f"Analyzing {total} comments...\n")

    for i, text in enumerate(df["comment"]):
        try:
            # Truncate to 512 tokens (model limit)
            result = sentiment_model(str(text)[:512])[0]
            sentiments.append(result["label"])
            confidences.append(round(result["score"], 3))
        except Exception:
            sentiments.append("UNKNOWN")
            confidences.append(0.0)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total} comments...")

    df["sentiment"] = sentiments
    df["confidence"] = confidences

    print(f"\nSentiment analysis complete!")
    return df


def print_summary(df):
    """Print a summary of the sentiment analysis results."""
    print("\n" + "=" * 50)
    print("         SENTIMENT ANALYSIS SUMMARY")
    print("=" * 50)

    counts = df["sentiment"].value_counts()
    total = len(df)

    for label, count in counts.items():
        pct = round(count / total * 100, 1)
        bar = "#" * int(pct / 2)
        print(f"  {label:10s}: {count:4d} ({pct:5.1f}%) {bar}")

    avg_conf = round(df["confidence"].mean(), 3)
    print(f"\n  Average confidence: {avg_conf}")
    print("=" * 50)

    # Show top positive and negative comments
    pos = df[df["sentiment"] == "POSITIVE"].nlargest(3, "confidence")
    neg = df[df["sentiment"] == "NEGATIVE"].nlargest(3, "confidence")

    print("\nTop 3 most POSITIVE comments:")
    for _, row in pos.iterrows():
        print(f"  [{row['confidence']}] {row['comment'][:80]}...")

    print("\nTop 3 most NEGATIVE comments:")
    for _, row in neg.iterrows():
        print(f"  [{row['confidence']}] {row['comment'][:80]}...")


if __name__ == "__main__":
    # Load scraped dataset
    df = load_dataset("dataset.csv")

    # Run sentiment analysis
    df = run_sentiment_analysis(df)

    # Print summary
    print_summary(df)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset_with_sentiment.csv"
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to: {output_path}")
