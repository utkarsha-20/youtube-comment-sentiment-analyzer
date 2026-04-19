
"""
YouTube Comment Sentiment Analyzer — Gradio Web App
====================================================
Paste a YouTube URL → Scrape comments → NLP sentiment analysis → Visualize
"""

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
from transformers import pipeline
import tempfile
import requests
import os

# --- Load NLP model once ---
print("Loading sentiment analysis model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
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


def get_video_title(video_id, api_key):
    """Fetch video title using YouTube Data API v3."""
    try:
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {"part": "snippet", "id": video_id, "key": api_key}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        items = data.get("items", [])
        if items:
            return items[0]["snippet"]["title"]
    except Exception as e:
        print(f"[scraper] Could not fetch title: {e}")
    return "Unknown Video"


def scrape_comments(video_url, max_comments, fetch_all=False):
    """Fetch YouTube comments using YouTube Data API v3."""
    api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if not api_key:
        print("[scraper] ERROR: YOUTUBE_API_KEY not set")
        return pd.DataFrame(), "Unknown Video"

    video_url = clean_youtube_url(video_url)
    # Extract video ID
    video_id = video_url.split("v=")[1] if "v=" in video_url else ""
    print(f"[scraper] Starting scrape for video_id: {video_id}")

    title = get_video_title(video_id, api_key)
    print(f"[scraper] Video title: {title}")

    comments_data = []
    next_page_token = None
    limit = max_comments if not fetch_all else 999999

    while len(comments_data) < limit:
        try:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, limit - len(comments_data)),
                "order": "relevance",
                "key": api_key,
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            r = requests.get(
                "https://www.googleapis.com/youtube/v3/commentThreads",
                params=params,
                timeout=15,
            )
            data = r.json()

            if "error" in data:
                print(f"[scraper] API error: {data['error']['message']}")
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments_data.append({
                    "comment": str(snippet.get("textDisplay", "")).strip(),
                    "author": str(snippet.get("authorDisplayName", "Unknown")),
                    "date": str(snippet.get("publishedAt", "N/A"))[:10],
                    "likes": snippet.get("likeCount", 0) or 0,
                })

            next_page_token = data.get("nextPageToken")
            print(f"[scraper] Fetched {len(comments_data)} comments so far...")

            if not next_page_token:
                break

        except Exception as e:
            import traceback
            print(f"[scraper] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            break

    print(f"[scraper] Final result: {len(comments_data)} comments")
    return pd.DataFrame(comments_data), title


CONFIDENCE_THRESHOLD = 0.85

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

    # Map model labels to positive/neutral/negative
    label_map = {
        "positive": "positive", "POSITIVE": "positive", "POS": "positive",
        "negative": "negative", "NEGATIVE": "negative", "NEG": "negative",
        "neutral": "neutral",   "NEUTRAL": "neutral",   "NEU": "neutral",
    }

    sentiments = []
    confidences = []
    for r in all_results:
        label = label_map.get(r["label"], "neutral")
        score = round(r["score"], 3)
        # Low confidence → classify as neutral
        if score < CONFIDENCE_THRESHOLD and label != "neutral":
            label = "neutral"
        sentiments.append(label)
        confidences.append(score)

    df["sentiment"] = sentiments
    df["confidence"] = confidences
    return df


CHART_BG = "#1e1e1e"
CHART_FG = "#c9c9c9"
CHART_GRID = "#2e2e2e"
COLOR_POS = "#3fb950"
COLOR_NEG = "#e05c4b"
COLOR_NEU = "#666666"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=CHART_FG, size=12),
    margin=dict(l=40, r=20, t=20, b=40),
)


def create_bar_chart(df):
    """Create sentiment distribution bar chart using Plotly."""
    counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"]).dropna()
    colors = {"positive": COLOR_POS, "negative": COLOR_NEG, "neutral": COLOR_NEU}
    bar_colors = [colors.get(l, "#8b949e") for l in counts.index]
    total = len(df)
    labels = [f"{v} ({round(v/total*100,1)}%)" for v in counts.values]

    fig = go.Figure(go.Bar(
        x=list(counts.index),
        y=list(counts.values),
        marker_color=bar_colors,
        text=labels,
        textposition="outside",
        textfont=dict(color=CHART_FG, size=12),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        yaxis=dict(gridcolor=CHART_GRID, zeroline=False, title="Comments", title_font=dict(color="#888")),
        xaxis=dict(gridcolor=CHART_GRID, zeroline=False),
        bargap=0.4,
    )
    return fig


def create_pie_chart(df):
    """Create sentiment pie chart using Plotly."""
    counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"]).dropna()
    colors = {"positive": COLOR_POS, "negative": COLOR_NEG, "neutral": COLOR_NEU}
    pie_colors = [colors.get(l, "#8b949e") for l in counts.index]

    fig = go.Figure(go.Pie(
        labels=list(counts.index),
        values=list(counts.values),
        marker=dict(colors=pie_colors, line=dict(color=CHART_BG, width=2)),
        textfont=dict(color=CHART_FG, size=12),
        hole=0.3,
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      legend=dict(font=dict(color=CHART_FG)))
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
    progress(0.0, desc="Step 1/4: Fetching comments from YouTube...")
    try:
        df, video_title = scrape_comments(video_url.strip(), max_comments, fetch_all)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Scraping failed: {type(e).__name__}: {e}", None, None, None, None, None, None

    if df.empty:
        return f"No comments found. The video may have comments disabled or the URL is invalid.", None, None, None, None, None, None

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

    summary = f"""## {video_title}

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
input[type=checkbox] {
    appearance: none !important;
    width: 14px !important;
    height: 14px !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 3px !important;
    background: #1e1e1e !important;
    cursor: pointer !important;
    position: relative !important;
    transition: background 150ms ease, border-color 150ms ease !important;
}
input[type=checkbox]:checked {
    background: #3fb950 !important;
    border-color: #3fb950 !important;
}
input[type=checkbox]:checked::after {
    content: '' !important;
    position: absolute !important;
    left: 4px !important;
    top: 1px !important;
    width: 4px !important;
    height: 8px !important;
    border: 2px solid #121212 !important;
    border-top: none !important;
    border-left: none !important;
    transform: rotate(45deg) !important;
}

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
        count_input = gr.Slider(minimum=50, maximum=5000, value=500, step=50, label="Number of comments to fetch", scale=1, info="Slide to choose how many comments to analyze")

    with gr.Row():
        fetch_all_input = gr.Checkbox(label="Fetch all comments (ignores slider)", value=False)

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
        summary, bar, pie, wc_p, wc_n, display_df, csv = analyze_video(video_url, max_comments, fetch_all, progress)
        actual_count = len(display_df) if display_df is not None and hasattr(display_df, '__len__') else max_comments
        new_slider = gr.update(value=actual_count, maximum=max(5000, actual_count))
        return gr.update(value=summary, visible=True), new_slider, bar, pie, wc_p, wc_n, display_df, csv

    analyze_btn.click(
        fn=analyze_and_show,
        inputs=[url_input, count_input, fetch_all_input],
        outputs=[summary_output, count_input, bar_chart, pie_chart, wc_pos, wc_neg, data_table, csv_download],
    )

demo.launch()
