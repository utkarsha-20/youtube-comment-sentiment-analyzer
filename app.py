"""
YouTube Comment Sentiment Analyzer — Gradio Web App
====================================================
Paste a YouTube URL → Scrape comments → NLP sentiment analysis → Visualize
"""

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline
import io
import tempfile
import os

# --- Load NLP model once ---
print("Loading sentiment analysis model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
print("Model loaded!")


def scrape_comments(video_url, max_comments):
    """Scrape YouTube comments using web scraping."""
    downloader = YoutubeCommentDownloader()
    comments_data = []
    count = 0

    comments = downloader.get_comments_from_url(video_url, sort_by=0)

    for comment in comments:
        if count >= max_comments:
            break
        text = comment.get("text", "").strip()
        if text:
            comments_data.append({
                "comment": text,
                "author": comment.get("author", "Unknown"),
                "date": comment.get("time", "N/A"),
                "likes": comment.get("votes", 0),
            })
            count += 1

    return pd.DataFrame(comments_data)


def analyze_sentiment(df):
    """Run sentiment analysis on comments."""
    sentiments = []
    confidences = []

    for text in df["comment"]:
        result = sentiment_model(str(text)[:512])[0]
        sentiments.append(result["label"])
        confidences.append(round(result["score"], 3))

    df["sentiment"] = sentiments
    df["confidence"] = confidences
    return df


def create_bar_chart(df):
    """Create sentiment distribution bar chart."""
    counts = df["sentiment"].value_counts()
    colors = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"}
    bar_colors = [colors.get(l, "#95a5a6") for l in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="white", linewidth=1.5)
    for i, (label, val) in enumerate(zip(counts.index, counts.values)):
        pct = round(val / len(df) * 100, 1)
        ax.text(i, val + 1, f"{val} ({pct}%)", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylabel("Number of Comments")
    ax.set_title("Sentiment Distribution")
    plt.tight_layout()
    return fig


def create_pie_chart(df):
    """Create sentiment pie chart."""
    counts = df["sentiment"].value_counts()
    colors = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"}
    pie_colors = [colors.get(l, "#95a5a6") for l in counts.index]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        counts.values,
        labels=counts.index,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Sentiment Breakdown")
    plt.tight_layout()
    return fig


def create_wordcloud(df, sentiment, colormap):
    """Create word cloud for a specific sentiment."""
    text = " ".join(df[df["sentiment"] == sentiment]["comment"].dropna().astype(str))
    if not text.strip():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No {sentiment.lower()} comments", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    wc = WordCloud(width=800, height=400, background_color="white", colormap=colormap, max_words=80).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{sentiment} Comments — Word Cloud", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def analyze_video(video_url, max_comments):
    """Main function: scrape → analyze → visualize."""
    if not video_url or not video_url.strip():
        return "Please enter a YouTube URL.", None, None, None, None, None, None

    max_comments = int(max_comments)

    # Step 1: Scrape
    try:
        df = scrape_comments(video_url.strip(), max_comments)
    except Exception as e:
        return f"Scraping failed: {e}", None, None, None, None, None, None

    if df.empty:
        return "No comments found. Check the URL and try again.", None, None, None, None, None, None

    # Step 2: Sentiment Analysis
    df = analyze_sentiment(df)

    # Step 3: Summary
    total = len(df)
    pos_count = len(df[df["sentiment"] == "POSITIVE"])
    neg_count = len(df[df["sentiment"] == "NEGATIVE"])
    avg_conf = round(df["confidence"].mean(), 3)

    summary = f"""## Results

| Metric | Value |
|--------|-------|
| Total Comments | {total} |
| Positive | {pos_count} ({round(pos_count/total*100, 1)}%) |
| Negative | {neg_count} ({round(neg_count/total*100, 1)}%) |
| Avg Confidence | {avg_conf} |

### Top Positive Comments
"""
    top_pos = df[df["sentiment"] == "POSITIVE"].nlargest(3, "confidence")
    for _, row in top_pos.iterrows():
        summary += f"- **{row['author']}** ({row['confidence']}): {row['comment'][:100]}...\n"

    summary += "\n### Top Negative Comments\n"
    top_neg = df[df["sentiment"] == "NEGATIVE"].nlargest(3, "confidence")
    for _, row in top_neg.iterrows():
        summary += f"- **{row['author']}** ({row['confidence']}): {row['comment'][:100]}...\n"

    # Step 4: Charts
    bar_chart = create_bar_chart(df)
    pie_chart = create_pie_chart(df)
    wc_positive = create_wordcloud(df, "POSITIVE", "Greens")
    wc_negative = create_wordcloud(df, "NEGATIVE", "Reds")

    # Step 5: Data table
    display_df = df[["comment", "author", "date", "likes", "sentiment", "confidence"]]

    # Step 6: CSV download
    csv_path = os.path.join(tempfile.gettempdir(), "sentiment_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return summary, bar_chart, pie_chart, wc_positive, wc_negative, display_df, csv_path


# --- Gradio UI ---
with gr.Blocks(
    title="YouTube Comment Sentiment Analyzer",
    theme=gr.themes.Soft(primary_hue="red", secondary_hue="green"),
) as demo:

    gr.Markdown(
        """
        # YouTube Comment Sentiment Analyzer
        **Web Scraping + NLP Pipeline**

        Paste any YouTube video URL → Scrape comments → Analyze sentiment → Visualize results
        """
    )

    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            scale=3,
        )
        count_input = gr.Slider(
            minimum=50, maximum=500, value=200, step=50,
            label="Number of Comments",
            scale=1,
        )

    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

    # Output components
    summary_output = gr.Markdown(label="Summary")

    with gr.Row():
        bar_chart = gr.Plot(label="Sentiment Distribution")
        pie_chart = gr.Plot(label="Sentiment Breakdown")

    with gr.Row():
        wc_pos = gr.Plot(label="Positive Word Cloud")
        wc_neg = gr.Plot(label="Negative Word Cloud")

    data_table = gr.Dataframe(label="All Comments with Sentiment", wrap=True)
    csv_download = gr.File(label="Download Results (CSV)")

    # Connect button
    analyze_btn.click(
        fn=analyze_video,
        inputs=[url_input, count_input],
        outputs=[summary_output, bar_chart, pie_chart, wc_pos, wc_neg, data_table, csv_download],
    )

    gr.Markdown(
        """
        ---
        **How it works:** Web scraping (youtube-comment-downloader) → NLP (DistilBERT) → Visualization (matplotlib, wordcloud)
        """
    )

demo.launch()
