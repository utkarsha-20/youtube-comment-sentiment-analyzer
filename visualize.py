"""
Visualization
--------------
Creates charts and word clouds from the sentiment analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    from wordcloud import WordCloud
except ImportError:
    print("Installing wordcloud...")
    os.system("pip install wordcloud")
    from wordcloud import WordCloud


def load_results(filename="dataset_with_sentiment.csv"):
    """Load the sentiment analysis results."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} analyzed comments")
    return df


def plot_sentiment_distribution(df, save_path="charts"):
    """Bar chart showing positive vs negative count."""
    os.makedirs(save_path, exist_ok=True)

    counts = df["sentiment"].value_counts()
    colors = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "UNKNOWN": "#95a5a6"}
    bar_colors = [colors.get(label, "#3498db") for label in counts.index]

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values, color=bar_colors, edgecolor="white", linewidth=1.5)
    plt.title("Sentiment Distribution of YouTube Comments", fontsize=14, fontweight="bold")
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Number of Comments", fontsize=12)

    for i, (label, val) in enumerate(zip(counts.index, counts.values)):
        pct = round(val / len(df) * 100, 1)
        plt.text(i, val + 2, f"{val} ({pct}%)", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    filepath = os.path.join(save_path, "sentiment_distribution.png")
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"Saved: {filepath}")


def plot_sentiment_pie(df, save_path="charts"):
    """Pie chart of sentiment percentages."""
    os.makedirs(save_path, exist_ok=True)

    counts = df["sentiment"].value_counts()
    colors = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "UNKNOWN": "#95a5a6"}
    pie_colors = [colors.get(label, "#3498db") for label in counts.index]

    plt.figure(figsize=(7, 7))
    plt.pie(
        counts.values,
        labels=counts.index,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    plt.title("Sentiment Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    filepath = os.path.join(save_path, "sentiment_pie.png")
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"Saved: {filepath}")


def plot_confidence_histogram(df, save_path="charts"):
    """Histogram of model confidence scores."""
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(df["confidence"], bins=20, color="#3498db", edgecolor="white", linewidth=1.2)
    plt.title("Confidence Score Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    filepath = os.path.join(save_path, "confidence_histogram.png")
    plt.savefig(filepath, dpi=150)
    plt.show()
    print(f"Saved: {filepath}")


def generate_wordclouds(df, save_path="charts"):
    """Word clouds for positive and negative comments."""
    os.makedirs(save_path, exist_ok=True)

    for sentiment, color in [("POSITIVE", "Greens"), ("NEGATIVE", "Reds")]:
        text = " ".join(df[df["sentiment"] == sentiment]["comment"].dropna().astype(str))
        if not text.strip():
            continue

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap=color,
            max_words=100,
        ).generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{sentiment} Comments — Word Cloud", fontsize=14, fontweight="bold")
        plt.tight_layout()
        filepath = os.path.join(save_path, f"wordcloud_{sentiment.lower()}.png")
        plt.savefig(filepath, dpi=150)
        plt.show()
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    df = load_results()

    print("\nGenerating visualizations...\n")
    plot_sentiment_distribution(df)
    plot_sentiment_pie(df)
    plot_confidence_histogram(df)
    generate_wordclouds(df)

    print("\nAll charts saved in 'charts/' folder!")
