"""
YouTube Comment Sentiment Analyzer
====================================
Complete NLP pipeline:
  1. Paste a YouTube URL
  2. Scrape comments (web scraping)
  3. Run sentiment analysis (NLP)
  4. Visualize results (charts + word clouds)

Usage:
  python main.py
"""

from scraper import scrape_youtube_comments, save_dataset
from nlp_analysis import run_sentiment_analysis, print_summary
from visualize import (
    load_results,
    plot_sentiment_distribution,
    plot_sentiment_pie,
    plot_confidence_histogram,
    generate_wordclouds,
)
import os


def main():
    print("=" * 55)
    print("   YOUTUBE COMMENT SENTIMENT ANALYZER")
    print("   Web Scraping + NLP Pipeline")
    print("=" * 55)

    # Step 1: Get YouTube URL from user
    video_url = input("\nPaste YouTube video URL: ").strip()
    if not video_url:
        print("No URL provided. Using default demo video.")
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    try:
        max_comments = int(input("How many comments to scrape? (default 500): ").strip() or "500")
    except ValueError:
        max_comments = 500

    # Step 2: Scrape comments (Web Scraping)
    print("\n--- STEP 1: WEB SCRAPING ---")
    df = scrape_youtube_comments(video_url, max_comments=max_comments)

    if df.empty:
        print("No comments scraped. Exiting.")
        return

    save_dataset(df, "dataset.csv")

    # Step 3: Sentiment Analysis (NLP)
    print("\n--- STEP 2: NLP SENTIMENT ANALYSIS ---")
    df = run_sentiment_analysis(df)
    print_summary(df)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset_with_sentiment.csv"
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to: {output_path}")

    # Step 4: Visualization
    print("\n--- STEP 3: VISUALIZATION ---")
    charts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
    plot_sentiment_distribution(df, charts_path)
    plot_sentiment_pie(df, charts_path)
    plot_confidence_histogram(df, charts_path)
    generate_wordclouds(df, charts_path)

    print("\n" + "=" * 55)
    print("   PIPELINE COMPLETE!")
    print(f"   Dataset: dataset_with_sentiment.csv")
    print(f"   Charts:  charts/ folder")
    print("=" * 55)


if __name__ == "__main__":
    main()
