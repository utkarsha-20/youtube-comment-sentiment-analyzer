"""
YouTube Comment Scraper
-----------------------
Scrapes comments from any YouTube video URL using web scraping (no API key needed).
Saves the scraped data as a CSV dataset for NLP analysis.
"""

import pandas as pd
import os
import sys
from datetime import datetime

try:
    from youtube_comment_downloader import YoutubeCommentDownloader
except ImportError:
    print("Installing youtube-comment-downloader...")
    os.system("pip install youtube-comment-downloader")
    from youtube_comment_downloader import YoutubeCommentDownloader


def scrape_youtube_comments(video_url, max_comments=500):
    """
    Scrapes comments from a YouTube video URL.

    Args:
        video_url: Full YouTube video URL (e.g. https://www.youtube.com/watch?v=xxxxx)
        max_comments: Maximum number of comments to scrape (default 500)

    Returns:
        pandas DataFrame with scraped comments
    """
    print(f"\nScraping comments from: {video_url}")
    print(f"Target: {max_comments} comments\n")

    downloader = YoutubeCommentDownloader()
    comments_data = []
    count = 0

    try:
        comments = downloader.get_comments_from_url(video_url, sort_by=0)  # 0 = newest first

        for comment in comments:
            if count >= max_comments:
                break

            comments_data.append({
                "comment": comment.get("text", ""),
                "author": comment.get("author", "Unknown"),
                "date": comment.get("time", "N/A"),
                "likes": comment.get("votes", 0),
            })

            count += 1
            if count % 50 == 0:
                print(f"  Scraped {count} comments...")

    except Exception as e:
        print(f"Error while scraping: {e}")

    if not comments_data:
        print("No comments found. Check the URL and try again.")
        return pd.DataFrame()

    df = pd.DataFrame(comments_data)

    # Remove empty comments
    df = df[df["comment"].str.strip().str.len() > 0].reset_index(drop=True)

    print(f"\nTotal comments scraped: {len(df)}")
    return df


def save_dataset(df, filename="dataset.csv"):
    """Save DataFrame to CSV file."""
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Dataset saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # --- PASTE YOUR YOUTUBE VIDEO URL HERE ---
    VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # You can also pass URL as command line argument:
    #   python scraper.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
    if len(sys.argv) > 1:
        VIDEO_URL = sys.argv[1]

    MAX_COMMENTS = 500

    # Scrape
    df = scrape_youtube_comments(VIDEO_URL, max_comments=MAX_COMMENTS)

    if not df.empty:
        # Save
        save_dataset(df)
        print("\nSample data:")
        print(df.head(10).to_string(index=False))
        print(f"\nDataset shape: {df.shape}")
