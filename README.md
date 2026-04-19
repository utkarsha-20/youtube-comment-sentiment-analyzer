---
title: Brainrot Scanner 🍿
emoji: 🎬
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
---

# YouTube Comment Sentiment Analyzer

A complete **Web Scraping + NLP** web application that scrapes YouTube comments and performs sentiment analysis.

## How It Works

```
YouTube URL → Web Scraping → Dataset (CSV) → Sentiment Analysis (NLP) → Visualization
```

1. **Paste a YouTube video URL**
2. **Scraper** fetches comments using web scraping (no API key needed)
3. **NLP Model** (DistilBERT) classifies each comment as POSITIVE or NEGATIVE
4. **Visualizer** generates charts and word clouds
5. **Download** the results as CSV

## Features

- Scrape up to 500 comments from any YouTube video
- Real-time sentiment analysis using DistilBERT
- Interactive bar chart and pie chart
- Word clouds for positive and negative comments
- Confidence score distribution
- Top positive and negative comments highlighted
- Download results as CSV

## Project Structure

```
├── app.py                 # Gradio web app (main entry point)
├── main.py                # CLI pipeline (scrape → analyze → visualize)
├── scraper.py             # Web scraping module (YouTube comments)
├── nlp_analysis.py        # Sentiment analysis using HuggingFace Transformers
├── visualize.py           # Charts and word clouds (CLI version)
├── requirements.txt       # Python dependencies
└── README.md
```

## Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web App

```bash
python app.py
```

### 3. Or Run via CLI

```bash
python main.py
```

## Technologies Used

- **Web App**: `Gradio`
- **Web Scraping**: `youtube-comment-downloader`
- **NLP**: `transformers` (HuggingFace), `DistilBERT` model
- **Data Processing**: `pandas`
- **Visualization**: `matplotlib`, `wordcloud`

## Requirements

- Python 3.8+
- Internet connection (for scraping and first-time model download)
