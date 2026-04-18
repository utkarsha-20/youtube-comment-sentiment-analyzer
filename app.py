"""
YouTube Comment Sentiment Analyzer — Streamlit Web App
======================================================
Paste a YouTube URL → Scrape comments → NLP sentiment analysis → Visualize
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline


# --- Page Config ---
st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
)


# --- Cache the NLP model so it loads only once ---
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


# --- Scraper ---
def scrape_comments(video_url, max_comments):
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


# --- Sentiment Analysis ---
def analyze_sentiment(df, model):
    sentiments = []
    confidences = []

    for text in df["comment"]:
        result = model(str(text)[:512])[0]
        sentiments.append(result["label"])
        confidences.append(round(result["score"], 3))

    df["sentiment"] = sentiments
    df["confidence"] = confidences
    return df


# --- App UI ---
st.title("YouTube Comment Sentiment Analyzer")
st.markdown("Paste a YouTube video URL to scrape comments and analyze sentiment using NLP.")

# Sidebar
st.sidebar.header("Settings")
max_comments = st.sidebar.slider("Number of comments to scrape", 50, 500, 200, step=50)

# Input
video_url = st.text_input("Paste YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze", type="primary") and video_url:

    # Step 1: Scrape
    with st.status("Scraping comments...", expanded=True) as status:
        st.write(f"Fetching up to {max_comments} comments...")
        try:
            df = scrape_comments(video_url, max_comments)
        except Exception as e:
            st.error(f"Scraping failed: {e}")
            st.stop()

        if df.empty:
            st.error("No comments found. Check the URL and try again.")
            st.stop()

        st.write(f"Scraped **{len(df)}** comments")
        status.update(label=f"Scraped {len(df)} comments", state="complete")

    # Step 2: Sentiment Analysis
    with st.status("Running sentiment analysis...", expanded=True) as status:
        st.write("Loading DistilBERT model...")
        model = load_model()
        st.write("Analyzing comments...")
        df = analyze_sentiment(df, model)
        status.update(label="Sentiment analysis complete", state="complete")

    # --- Results ---
    st.divider()
    st.header("Results")

    # Metrics row
    total = len(df)
    pos_count = len(df[df["sentiment"] == "POSITIVE"])
    neg_count = len(df[df["sentiment"] == "NEGATIVE"])
    avg_conf = round(df["confidence"].mean(), 3)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Comments", total)
    col2.metric("Positive", f"{pos_count} ({round(pos_count/total*100, 1)}%)")
    col3.metric("Negative", f"{neg_count} ({round(neg_count/total*100, 1)}%)")
    col4.metric("Avg Confidence", avg_conf)

    # Charts side by side
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Sentiment Distribution")
        counts = df["sentiment"].value_counts()
        colors = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"}
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        bar_colors = [colors.get(l, "#95a5a6") for l in counts.index]
        ax1.bar(counts.index, counts.values, color=bar_colors, edgecolor="white", linewidth=1.5)
        for i, (label, val) in enumerate(zip(counts.index, counts.values)):
            ax1.text(i, val + 1, str(val), ha="center", fontweight="bold")
        ax1.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig1)

    with chart_col2:
        st.subheader("Sentiment Breakdown")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        pie_colors = [colors.get(l, "#95a5a6") for l in counts.index]
        ax2.pie(
            counts.values,
            labels=counts.index,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 12},
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        plt.tight_layout()
        st.pyplot(fig2)

    # Word Clouds
    st.subheader("Word Clouds")
    wc_col1, wc_col2 = st.columns(2)

    for col, sentiment, colormap, title in [
        (wc_col1, "POSITIVE", "Greens", "Positive Comments"),
        (wc_col2, "NEGATIVE", "Reds", "Negative Comments"),
    ]:
        text = " ".join(df[df["sentiment"] == sentiment]["comment"].dropna().astype(str))
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color="white", colormap=colormap, max_words=80).generate(text)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontsize=14, fontweight="bold")
            col.pyplot(fig)
        else:
            col.info(f"No {sentiment.lower()} comments found.")

    # Confidence histogram
    st.subheader("Confidence Score Distribution")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.hist(df["confidence"], bins=20, color="#3498db", edgecolor="white")
    ax3.set_xlabel("Confidence")
    ax3.set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig3)

    # Data table
    st.subheader("All Comments with Sentiment")
    st.dataframe(
        df[["comment", "author", "date", "likes", "sentiment", "confidence"]],
        use_container_width=True,
        height=400,
    )

    # Top comments
    top_col1, top_col2 = st.columns(2)

    with top_col1:
        st.subheader("Top Positive Comments")
        top_pos = df[df["sentiment"] == "POSITIVE"].nlargest(5, "confidence")
        for _, row in top_pos.iterrows():
            st.success(f"**{row['author']}** ({row['confidence']})\n\n{row['comment'][:150]}")

    with top_col2:
        st.subheader("Top Negative Comments")
        top_neg = df[df["sentiment"] == "NEGATIVE"].nlargest(5, "confidence")
        for _, row in top_neg.iterrows():
            st.error(f"**{row['author']}** ({row['confidence']})\n\n{row['comment'][:150]}")

    # Download button
    st.divider()
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="sentiment_results.csv",
        mime="text/csv",
    )
