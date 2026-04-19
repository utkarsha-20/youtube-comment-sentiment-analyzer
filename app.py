
"""
YouTube Comment Sentiment Analyzer — Gradio Web App
====================================================
Paste a YouTube URL → Scrape comments → NLP sentiment analysis → Visualize
"""

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import tempfile
import yt_dlp

# --- Load NLP model once ---
print("Loading sentiment analysis model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
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
    """Scrape YouTube comments using yt-dlp."""
    video_url = clean_youtube_url(video_url)
    print(f"[scraper] Starting scrape for: {video_url}")

    limit = None if fetch_all else max_comments

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
        "getcomments": True,
        "extractor_args": {
            "youtube": {
                "max_comments": [str(limit) if limit else "all"],
                "comment_sort": ["top"],
                "player_client": ["android", "web"],
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36",
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("[scraper] Extracting info with yt-dlp...")
            info = ydl.extract_info(video_url, download=False)
    except Exception as e:
        import traceback
        print(f"[scraper] yt-dlp FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    raw_comments = info.get("comments") or []
    print(f"[scraper] yt-dlp returned {len(raw_comments)} comments")

    comments_data = []
    for c in raw_comments:
        try:
            text = str(c.get("text", "")).strip()
            if not text:
                continue
            comments_data.append({
                "comment": text,
                "author": str(c.get("author", "Unknown")),
                "date": str(c.get("timestamp", "N/A")),
                "likes": c.get("like_count", 0) or 0,
            })
            if not fetch_all and len(comments_data) >= max_comments:
                break
        except (UnicodeError, ValueError):
            continue

    print(f"[scraper] Final result: {len(comments_data)} comments")
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


CHART_BG = "#1e1e1e"
CHART_FG = "#c9c9c9"
CHART_GRID = "#2e2e2e"
COLOR_POS = "#3fb950"
COLOR_NEG = "#e05c4b"
COLOR_NEU = "#666666"

def create_bar_chart(df):
    """Create sentiment distribution bar chart."""
    counts = df["sentiment"].value_counts()
    colors = {"positive": COLOR_POS, "negative": COLOR_NEG, "neutral": COLOR_NEU}
    bar_colors = [colors.get(l, "#8b949e") for l in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.bar(counts.index, counts.values, color=bar_colors, edgecolor=CHART_BG, linewidth=1, width=0.5)
    for i, (label, val) in enumerate(zip(counts.index, counts.values)):
        pct = round(val / len(df) * 100, 1)
        ax.text(i, val + 1, f"{val} ({pct}%)", ha="center", fontsize=11, color=CHART_FG)
    ax.set_ylabel("Comments", color="#8b949e", fontsize=12)
    ax.tick_params(colors=CHART_FG)
    ax.spines[:].set_color(CHART_GRID)
    ax.yaxis.grid(True, color=CHART_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def create_pie_chart(df):
    """Create sentiment pie chart."""
    counts = df["sentiment"].value_counts()
    colors = {"positive": COLOR_POS, "negative": COLOR_NEG, "neutral": COLOR_NEU}
    pie_colors = [colors.get(l, "#8b949e") for l in counts.index]

    fig, ax = plt.subplots(figsize=(5, 5), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.pie(
        counts.values,
        labels=counts.index,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 12, "color": CHART_FG},
        wedgeprops={"edgecolor": CHART_BG, "linewidth": 2},
    )
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
    if sentiment == "negative":
        stop_words.update({"love", "great", "good", "best", "amazing", "awesome",
                           "beautiful", "perfect", "excellent", "wonderful", "fantastic",
                           "cool", "nice", "hope", "happy", "gracias", "hermosos",
                           "grande", "indah"})
    elif sentiment == "positive":
        stop_words.update({"bad", "worst", "terrible", "horrible", "hate", "boring",
                           "waste", "poor", "ugly", "stupid", "awful", "annoying"})

    wc = WordCloud(width=800, height=400, background_color=CHART_BG, colormap=colormap,
                   max_words=80, stopwords=stop_words).generate(text)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
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
        import traceback
        traceback.print_exc()
        return f"Scraping failed: {type(e).__name__}: {e}", None, None, None, None, None, None

    if df.empty:
        return f"No comments found for URL: {clean_youtube_url(video_url.strip())} — the video may have comments disabled or YouTube is blocking this server.", None, None, None, None, None, None

    # Step 2: Sentiment Analysis (batch processing — fast)
    progress(0.3, desc=f"Step 2/4: Analyzing sentiment of {len(df)} comments...")
    df = analyze_sentiment(df)

    # Step 3: Summary
    progress(0.7, desc="Step 3/4: Generating charts and word clouds...")
    total = len(df)
    pos_count = len(df[df["sentiment"] == "positive"])
    neg_count = len(df[df["sentiment"] == "negative"])
    neu_count = len(df[df["sentiment"] == "neutral"])
    avg_conf = round(df["confidence"].mean(), 3)

    summary = f"""## Results

| Metric | Value |
|--------|-------|
| Total Comments Analyzed | {total} |
| Positive | {pos_count} ({round(pos_count/total*100, 1)}%) |
| Neutral | {neu_count} ({round(neu_count/total*100, 1)}%) |
| Negative | {neg_count} ({round(neg_count/total*100, 1)}%) |
| Avg Confidence | {avg_conf} |

### Top Positive Comments
"""
    top_pos = df[df["sentiment"] == "positive"].nlargest(3, "confidence")
    for _, row in top_pos.iterrows():
        summary += f"- **{row['author']}** ({row['confidence']}): {row['comment'][:100]}...\n"

    summary += "\n### Top Negative Comments\n"
    top_neg = df[df["sentiment"] == "negative"].nlargest(3, "confidence")
    for _, row in top_neg.iterrows():
        summary += f"- **{row['author']}** ({row['confidence']}): {row['comment'][:100]}...\n"

    # Step 4: Charts
    bar_chart = create_bar_chart(df)
    pie_chart = create_pie_chart(df)
    wc_positive = create_wordcloud(df, "positive", "Greens")
    wc_negative = create_wordcloud(df, "negative", "Reds")

    # Step 5: Data table
    display_df = df[["comment", "author", "date", "likes", "sentiment", "confidence"]]

    # Step 6: CSV download — save to a temp file Gradio can serve
    progress(0.9, desc="Step 4/4: Preparing CSV download...")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="sentiment_results_")
    df.to_csv(tmp.name, index=False, encoding="utf-8-sig")
    tmp.close()

    progress(1.0, desc="Done!")
    return summary, bar_chart, pie_chart, wc_positive, wc_negative, display_df, tmp.name


CSS = """
body, .gradio-container {
    background: #121212 !important;
    color: #e1e1e1 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif !important;
    font-size: 14px !important;
}
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 24px 20px !important;
}

/* Labels */
label { color: #888 !important; font-size: 12px !important; font-weight: 500 !important; }

/* Inputs */
input, textarea {
    background: #1e1e1e !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 6px !important;
    color: #e1e1e1 !important;
    font-size: 13px !important;
    padding: 8px 10px !important;
}
input:focus, textarea:focus { border-color: #4a4a4a !important; outline: none !important; }

/* Button */
.run-btn button {
    background: #1e1e1e !important;
    color: #e1e1e1 !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 6px !important;
    padding: 8px 0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: border-color 150ms !important;
}
.run-btn button:hover { border-color: #3fb950 !important; color: #3fb950 !important; }

/* Checkbox */
.gr-checkbox label { color: #888 !important; font-size: 12px !important; }

/* Slider */
input[type=range] { accent-color: #3fb950 !important; }

/* Plots */
.gr-plot {
    background: #1e1e1e !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 6px !important;
    padding: 8px !important;
}

/* Section title */
.section-title {
    font-size: 12px;
    color: #666;
    font-weight: 500;
    padding: 16px 0 6px 0;
    border-bottom: 1px solid #2e2e2e;
    margin-bottom: 8px;
}

/* Summary markdown */
.gr-markdown {
    background: #1e1e1e !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 6px !important;
    padding: 14px 16px !important;
    font-size: 13px !important;
}
.gr-markdown h2 { font-size: 14px !important; color: #e1e1e1 !important; font-weight: 600 !important; margin-bottom: 8px !important; border: none !important; }
.gr-markdown h3 { font-size: 12px !important; color: #666 !important; font-weight: 500 !important; margin: 12px 0 4px !important; }
.gr-markdown table { font-size: 12px !important; }
.gr-markdown th { color: #666 !important; font-weight: 500 !important; padding: 4px 8px !important; border-bottom: 1px solid #2e2e2e !important; text-align: left !important; }
.gr-markdown td { color: #c9c9c9 !important; padding: 4px 8px !important; border-bottom: 1px solid #2e2e2e !important; }

/* Dataframe */
.gr-dataframe { background: #1e1e1e !important; border: 1px solid #2e2e2e !important; border-radius: 6px !important; }
.gr-dataframe th { background: #1e1e1e !important; color: #666 !important; font-size: 11px !important; padding: 6px 8px !important; border-bottom: 1px solid #2e2e2e !important; }
.gr-dataframe td { color: #c9c9c9 !important; font-size: 12px !important; padding: 5px 8px !important; border-bottom: 1px solid #2e2e2e !important; }

/* File */
.gr-file { background: #1e1e1e !important; border: 1px solid #2e2e2e !important; border-radius: 6px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2e2e2e; border-radius: 3px; }

/* Hide HF popup */
.gradio-container ~ div, div[data-testid="share-btn"],
.share-button, .built-with, div.absolute.bottom-0,
div.fixed.bottom-0, div[class*="space-info"] { display: none !important; }
"""

# --- Gradio UI ---
with gr.Blocks(title="YouTube Comment Sentiment Analyzer", css=CSS) as demo:

    gr.Markdown("## YouTube Comment Sentiment Analyzer")

    with gr.Row():
        url_input = gr.Textbox(label="Video URL", placeholder="https://www.youtube.com/watch?v=...", lines=1, scale=3)
        count_input = gr.Slider(minimum=50, maximum=5000, value=500, step=50, label="Comments", scale=1)

    with gr.Row():
        fetch_all_input = gr.Checkbox(label="Fetch all comments", value=False)

    with gr.Row(elem_classes="run-btn"):
        analyze_btn = gr.Button("Analyze", variant="secondary")

    summary_output = gr.Markdown(visible=False)

    gr.HTML('<div class="section-title">Charts</div>')
    with gr.Row():
        bar_chart = gr.Plot(label="Distribution")
        pie_chart = gr.Plot(label="Breakdown")

    gr.HTML('<div class="section-title">Word clouds</div>')
    with gr.Row():
        wc_pos = gr.Plot(label="Positive")
        wc_neg = gr.Plot(label="Negative")

    gr.HTML('<div class="section-title">Comments</div>')
    data_table = gr.Dataframe(label="All comments with sentiment", wrap=True)

    csv_download = gr.File(label="Download CSV")

    def analyze_and_show(video_url, max_comments, fetch_all, progress=gr.Progress()):
        result = analyze_video(video_url, max_comments, fetch_all, progress)
        summary = result[0]
        rest = result[1:]
        return (gr.update(value=summary, visible=True),) + rest

    analyze_btn.click(
        fn=analyze_and_show,
        inputs=[url_input, count_input, fetch_all_input],
        outputs=[summary_output, bar_chart, pie_chart, wc_pos, wc_neg, data_table, csv_download],
    )

demo.launch()
