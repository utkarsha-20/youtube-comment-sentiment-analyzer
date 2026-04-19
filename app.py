
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


CHART_BG = "#161b22"
CHART_FG = "#c9d1d9"
CHART_GRID = "#21262d"
COLOR_POS = "#3fb950"
COLOR_NEG = "#f78166"
COLOR_NEU = "#8b949e"

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
        return f"Scraping failed: {e}", None, None, None, None, None, None

    if df.empty:
        return "No comments found. Check the URL and try again.", None, None, None, None, None, None

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
* { box-sizing: border-box; margin: 0; padding: 0; }

html, body { height: 100%; overflow: hidden; }

body, .gradio-container {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    font-size: 14px !important;
    height: 100vh !important;
    overflow: hidden !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
}

/* Top toolbar */
.toolbar {
    height: 48px;
    min-height: 48px;
    border-bottom: 1px solid #21262d;
    display: flex;
    align-items: center;
    padding: 0 20px;
    gap: 16px;
    background: #0d1117;
}
.toolbar-title {
    font-size: 14px;
    font-weight: 600;
    color: #f0f6fc;
    white-space: nowrap;
}
.toolbar-sub {
    font-size: 12px;
    color: #484f58;
}

/* Main layout — two columns */
.main-layout {
    display: flex !important;
    flex: 1 !important;
    height: calc(100vh - 48px) !important;
    overflow: hidden !important;
    gap: 0 !important;
}

/* Left panel — input + stats */
.left-panel {
    width: 300px;
    min-width: 300px;
    border-right: 1px solid #21262d;
    background: #0d1117;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 16px;
    gap: 12px;
}

/* Right panel — tabs with charts */
.right-panel {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    background: #0d1117;
}

/* Stats row inside left panel */
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 12px;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    font-size: 13px;
}
.stat-label { color: #8b949e; }
.stat-value { color: #f0f6fc; font-weight: 500; }
.stat-pos { color: #3fb950; font-weight: 500; }
.stat-neg { color: #f78166; font-weight: 500; }
.stat-neu { color: #8b949e; font-weight: 500; }

/* Inputs */
input, textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
    font-size: 13px !important;
    padding: 7px 10px !important;
    width: 100% !important;
}
input:focus, textarea:focus {
    border-color: #388bfd !important;
    outline: none !important;
}

/* Labels */
label {
    color: #8b949e !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    margin-bottom: 4px !important;
    display: block !important;
}

/* Analyze button */
.analyze-btn button {
    background: #238636 !important;
    color: #fff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 6px !important;
    padding: 7px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 150ms ease !important;
}
.analyze-btn button:hover { background: #2ea043 !important; }

/* Checkbox */
.gr-checkbox label {
    color: #c9d1d9 !important;
    font-size: 12px !important;
}

/* Tabs */
.gr-tab-nav {
    border-bottom: 1px solid #21262d !important;
    background: #0d1117 !important;
    padding: 0 16px !important;
}
.gr-tab-nav button {
    background: none !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #8b949e !important;
    font-size: 13px !important;
    padding: 10px 12px !important;
    cursor: pointer !important;
}
.gr-tab-nav button.selected {
    color: #f0f6fc !important;
    border-bottom-color: #f78166 !important;
}

/* Plots fill their tab */
.gr-plot {
    background: #0d1117 !important;
    border: none !important;
    height: 100% !important;
    padding: 12px !important;
}

/* Dataframe */
.gr-dataframe {
    background: #0d1117 !important;
    border: none !important;
    font-size: 12px !important;
}
.gr-dataframe th {
    background: #161b22 !important;
    color: #8b949e !important;
    font-size: 11px !important;
    border-bottom: 1px solid #21262d !important;
    padding: 6px 8px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
.gr-dataframe td {
    color: #c9d1d9 !important;
    font-size: 12px !important;
    border-bottom: 1px solid #21262d !important;
    padding: 6px 8px !important;
}

/* File download */
.gr-file {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    font-size: 12px !important;
}

/* Slider */
input[type=range] { accent-color: #388bfd !important; }

/* Markdown summary in left panel */
.summary-md {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 12px;
    font-size: 12px;
    color: #c9d1d9;
    line-height: 1.5;
}
.summary-md p { margin: 2px 0; }

/* Hide HF popup */
.gradio-container ~ div, .embed-container,
div[data-testid="share-btn"], .share-button, .built-with,
.svelte-1ipelgc, .space-info, footer .svelte-1rjryqp,
div.absolute.bottom-0, div.fixed.bottom-0,
div[class*="space-info"], div[class*="embed"],
.gradio-container + div {
    display: none !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
"""

# --- Gradio UI ---
with gr.Blocks(title="YouTube Comment Sentiment Analyzer", css=CSS) as demo:

    gr.HTML("""
    <div class="toolbar">
        <span class="toolbar-title">YouTube Comment Sentiment Analyzer</span>
        <span class="toolbar-sub">web scraping + NLP</span>
    </div>
    """)

    with gr.Row(elem_classes="main-layout"):

        # --- LEFT PANEL ---
        with gr.Column(elem_classes="left-panel"):
            url_input = gr.Textbox(
                label="Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                lines=1,
            )
            count_input = gr.Slider(
                minimum=50, maximum=5000, value=500, step=50,
                label="Comments to fetch",
            )
            fetch_all_input = gr.Checkbox(
                label="Fetch all comments (ignores limit)",
                value=False,
            )
            with gr.Row(elem_classes="analyze-btn"):
                analyze_btn = gr.Button("Analyze", variant="primary")

            summary_output = gr.HTML(value="", visible=False)
            csv_download = gr.File(label="Download CSV", visible=False)

        # --- RIGHT PANEL ---
        with gr.Column(elem_classes="right-panel"):
            with gr.Tabs():
                with gr.Tab("Distribution"):
                    bar_chart = gr.Plot(show_label=False)
                with gr.Tab("Breakdown"):
                    pie_chart = gr.Plot(show_label=False)
                with gr.Tab("Positive words"):
                    wc_pos = gr.Plot(show_label=False)
                with gr.Tab("Negative words"):
                    wc_neg = gr.Plot(show_label=False)
                with gr.Tab("Comments"):
                    data_table = gr.Dataframe(wrap=True, show_label=False)

    def analyze_and_show(video_url, max_comments, fetch_all, progress=gr.Progress()):
        result = analyze_video(video_url, max_comments, fetch_all, progress)
        summary_text, bar, pie, wcp, wcn, table, csv_path = result

        if bar is None:
            html = f'<div class="summary-md"><p style="color:#f78166">{summary_text}</p></div>'
            return gr.update(value=html, visible=True), bar, pie, wcp, wcn, table, gr.update(visible=False)

        total = len(table)
        pos = len(table[table["sentiment"] == "positive"]) if "sentiment" in table.columns else 0
        neg = len(table[table["sentiment"] == "negative"]) if "sentiment" in table.columns else 0
        neu = len(table[table["sentiment"] == "neutral"]) if "sentiment" in table.columns else 0

        html = f"""
        <div style="display:flex;flex-direction:column;gap:8px;margin-top:4px">
            <div class="stat-row"><span class="stat-label">Total</span><span class="stat-value">{total}</span></div>
            <div class="stat-row"><span class="stat-label">Positive</span><span class="stat-pos">{pos} ({round(pos/total*100,1) if total else 0}%)</span></div>
            <div class="stat-row"><span class="stat-label">Neutral</span><span class="stat-neu">{neu} ({round(neu/total*100,1) if total else 0}%)</span></div>
            <div class="stat-row"><span class="stat-label">Negative</span><span class="stat-neg">{neg} ({round(neg/total*100,1) if total else 0}%)</span></div>
        </div>
        """
        return gr.update(value=html, visible=True), bar, pie, wcp, wcn, table, gr.update(value=csv_path, visible=True)

    analyze_btn.click(
        fn=analyze_and_show,
        inputs=[url_input, count_input, fetch_all_input],
        outputs=[summary_output, bar_chart, pie_chart, wc_pos, wc_neg, data_table, csv_download],
    )

demo.launch()
