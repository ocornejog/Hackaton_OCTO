# Reddit Country Sentiment Analyzer

## Overview
This Streamlit application analyzes Reddit posts to gauge sentiment and security perceptions about different countries. It processes posts from specified subreddits, extracts country mentions, and performs sentiment analysis using OpenAI's GPT-3.5 model and TextBlob.

## Features
- Real-time Reddit post analysis
- Interactive world map visualizations
- Country sentiment analysis
- Security perception analysis
- Engagement metrics tracking
- Data export capabilities

## Visualizations
- World Map: Global distribution of sentiments
- Country Rankings: Top countries by satisfaction and mention frequency
- Engagement Analysis: User interaction patterns
- Correlation Analysis: Relationships between different metrics
- Security Analysis: Global security perception patterns

## Setup

### Prerequisites
- Python 3.7+
- Reddit API credentials
- OpenAI API key

### Environment Variables
Create a `.env` file in the root directory with:

OPENAI_API_KEY=your_openai_api_key
CLIENT_ID=your_reddit_client_id
CLIENT_SECRET=your_reddit_client_secret
USER_AGENT=your_reddit_user_agent

### Installation
1. Clone the repository:
bash
git clone [repository-url]

2. Install required packages:
bash
pip install -r requirements.txt

3. Run the application:
bash
streamlit run main.py

## Usage
1. Open the application in your web browser
2. Enter a subreddit name (default: "travel")
3. Adjust the number of posts to analyze using the slider
4. Click "Analyser" to start the analysis
5. Explore the various visualization tabs
6. Download the analyzed data as CSV if needed

## Data Analysis
The application provides several metrics:
- Sentiment scores (-1 to 1)
- Security perception scores
- Engagement rates
- Mention frequency
- Upvotes and comments statistics