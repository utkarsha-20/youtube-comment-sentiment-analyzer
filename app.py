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
import tempfile

# --- Load NLP model once ---
print("Loading sentiment analysis model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,
    truncation=True,
    max_length=512,
)
print("Model loaded!")


def clean_youtube_url(url):
    """Clean YouTube URL — remove extra parameters, handle all URL formats."""
    url = url.strip()
    # Extract video ID from any YouTube URL format
    video_id = None
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    elif "v=" in url:
        video_id = url.split("v=")[1].split("&")[0].split("#")[0]
    elif "shorts/" in url:
        video_id = url.split("shorts/")[1].split("?")[0].split("&")[0]
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def scrape_comments(video_url, max_comments, fetch_all=False):
    """Scrape YouTube comments using web scraping."""
    video_url = clean_youtube_url(video_url)
    downloader = YoutubeCommentDownloader()
    comments_data = []
    count = 0

    # Get comments without sorting to avoid "Failed to set sorting" error
    try:
        comments = downloader.get_comments_from_url(video_url)
    except Exception:
        # Extract video ID and try with get_comments method
        video_id = video_url.split("v=")[1].split("&")[0] if "v=" in video_url else video_url
        comments = downloader.get_comments(video_id)

    for comment in comments:
        if not fetch_all and count >= max_comments:
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
    """Run sentiment analysis in batches for speed."""
    comments = [str(text)[:512] for text in df["comment"]]

    # Process in batches of 64 — much faster than one by one
    batch_size = 64
    all_results = []

    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        results = sentiment_model(batch, batch_size=batch_size)
        all_results.extend(results)

    df["sentiment"] = [r["label"] for r in all_results]
    df["confidence"] = [round(r["score"], 3) for r in all_results]
    return df


def create_bar_chart(df):
    """Create sentiment distribution bar chart."""
    counts = df["sentiment"].value_counts()
    colors = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"}
    bar_colors = [colors.get(l, "#95a5a6") for l in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="white", linewidth=1.5)
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
    """Create word cloud for a specific sentiment with filtered words."""
    text = " ".join(df[df["sentiment"] == sentiment]["comment"].dropna().astype(str))
    if not text.strip():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No {sentiment.lower()} comments", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Words to exclude — common words that don't indicate sentiment
    stop_words = {
        "the", "a", "an", "is", "it", "i", "you", "he", "she", "we", "they",
        "my", "your", "his", "her", "our", "this", "that", "these", "those",
        "am", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "about", "but", "or", "and", "not", "no", "so",
        "if", "than", "too", "very", "just", "also", "more", "most", "all",
        "each", "every", "both", "few", "some", "any", "other", "one", "two",
        "up", "out", "its", "what", "which", "who", "when", "where", "how",
        "me", "him", "them", "here", "there", "then", "now", "only", "own",
        "same", "much", "even", "still", "after", "before", "because", "while",
        "de", "el", "la", "en", "que", "por", "un", "una", "los", "las",
        "del", "al", "es", "ya", "se", "le", "da", "di", "ke", "ni", "ka",
        "dan", "ini", "itu", "yang", "dengan", "untuk", "pada",
        "como", "mas", "mais", "vai", "com", "pra", "nos", "das", "dos",
        "video", "comment", "like", "watch", "channel", "subscribe",
    }

    # Add opposite-sentiment words to stop list
    if sentiment == "NEGATIVE":
        stop_words.update({"love", "great", "good", "best", "amazing", "awesome",
                           "beautiful", "perfect", "excellent", "wonderful", "fantastic",
                           "cool", "nice", "hope", "happy", "gracias", "hermosos",
                           "grande", "indah", "love"})
    else:
        stop_words.update({"bad", "worst", "terrible", "horrible", "hate", "boring",
                           "waste", "poor", "ugly", "stupid", "awful", "annoying"})

    wc = WordCloud(width=800, height=400, background_color="white", colormap=colormap,
                   max_words=80, stopwords=stop_words).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{sentiment} Comments — Word Cloud", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def analyze_video(video_url, max_comments, fetch_all, progress=gr.Progress()):
    """Main function: scrape → analyze → visualize."""
    if not video_url or not video_url.strip():
        return "Please enter a YouTube URL.", None, None, None, None, None, None

    max_comments = int(max_comments)

    # Step 1: Scrape
    progress(0.0, desc="Step 1/4: Scraping comments from YouTube...")
    try:
        df = scrape_comments(video_url.strip(), max_comments, fetch_all)
    except Exception as e:
        return f"Scraping failed: {e}", None, None, None, None, None, None

    if df.empty:
        return "No comments found. Check the URL and try again.", None, None, None, None, None, None

    # Step 2: Sentiment Analysis (batch processing — fast)
    progress(0.3, desc=f"Step 2/4: Analyzing sentiment of {len(df)} comments...")
    df = analyze_sentiment(df)

    # Step 3: Summary
    progress(0.7, desc="Step 3/4: Generating charts and word clouds...")
    total = len(df)
    pos_count = len(df[df["sentiment"] == "POSITIVE"])
    neg_count = len(df[df["sentiment"] == "NEGATIVE"])
    avg_conf = round(df["confidence"].mean(), 3)

    summary = f"""## Results

| Metric | Value |
|--------|-------|
| Total Comments Analyzed | {total} |
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

    # Step 6: CSV download — save to a temp file Gradio can serve
    progress(0.9, desc="Step 4/4: Preparing CSV download...")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="sentiment_results_")
    df.to_csv(tmp.name, index=False, encoding="utf-8-sig")
    tmp.close()

    progress(1.0, desc="Done!")
    return summary, bar_chart, pie_chart, wc_positive, wc_negative, display_df, tmp.name


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
            minimum=50, maximum=5000, value=500, step=50,
            label="Number of Comments (ignored if 'Fetch All' is checked)",
            scale=1,
        )

    fetch_all_input = gr.Checkbox(
        label="Fetch ALL comments (may take a few minutes for popular videos)",
        value=False,
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
        inputs=[url_input, count_input, fetch_all_input],
        outputs=[summary_output, bar_chart, pie_chart, wc_pos, wc_neg, data_table, csv_download],
    )

    gr.Markdown(
        """
        ---
        **How it works:** Web scraping (youtube-comment-downloader) → NLP (DistilBERT with batch processing) → Visualization (matplotlib, wordcloud)
        """
    )

demo.launch()
