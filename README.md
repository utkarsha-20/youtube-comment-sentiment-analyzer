# YouTube Comment Sentiment Analyzer

A complete **Web Scraping + NLP** pipeline that scrapes YouTube comments and performs sentiment analysis.

## How It Works

```
YouTube URL → Web Scraping → Dataset (CSV) → Sentiment Analysis (NLP) → Visualization
```

1. **Paste a YouTube video URL**
2. **Scraper** fetches comments using web scraping (no API key needed)
3. **NLP Model** (DistilBERT) classifies each comment as POSITIVE or NEGATIVE
4. **Visualizer** generates charts and word clouds

## Project Structure

```
├── main.py                # Run full pipeline (scrape → analyze → visualize)
├── scraper.py             # Web scraping module (YouTube comments)
├── nlp_analysis.py        # Sentiment analysis using HuggingFace Transformers
├── visualize.py           # Charts and word clouds
├── requirements.txt       # Python dependencies
├── dataset.csv            # Scraped comments (generated after running)
├── dataset_with_sentiment.csv  # Comments with sentiment labels (generated)
└── charts/                # Visualization outputs (generated)
    ├── sentiment_distribution.png
    ├── sentiment_pie.png
    ├── confidence_histogram.png
    ├── wordcloud_positive.png
    └── wordcloud_negative.png
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python main.py
```

It will ask you to paste a YouTube URL and number of comments to scrape.

### 3. Or Run Each Step Individually

```bash
# Step 1: Scrape comments
python scraper.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Step 2: Run sentiment analysis
python nlp_analysis.py

# Step 3: Generate visualizations
python visualize.py
```

## Sample Output

| comment | author | sentiment | confidence |
|---------|--------|-----------|------------|
| "This song is amazing!" | User1 | POSITIVE | 0.998 |
| "Waste of time" | User2 | NEGATIVE | 0.991 |
| "Not bad, pretty decent" | User3 | POSITIVE | 0.834 |

### Sentiment Summary
```
POSITIVE:  387 (77.4%) #####################################
NEGATIVE:  113 (22.6%) ###########
```

## Technologies Used

- **Web Scraping**: `youtube-comment-downloader`, `BeautifulSoup4`
- **NLP**: `transformers` (HuggingFace), `DistilBERT` model
- **Data Processing**: `pandas`
- **Visualization**: `matplotlib`, `wordcloud`

## Requirements

- Python 3.8+
- Internet connection (for scraping and first-time model download)
