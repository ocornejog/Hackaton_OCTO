import os
from dotenv import load_dotenv
import praw
from textblob import TextBlob
import pycountry
import nltk
from nltk.tokenize import word_tokenize

load_dotenv()

CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET= os.environ['CLIENT_SECRET']
USER_AGENT = os.environ['USER_AGENT']


# Configuration de l'API Reddit
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT[0]
)

def get_countries_from_text(text):
    countries = []
    words = word_tokenize(text.lower())
    for country in pycountry.countries:
        if country.name.lower() in text.lower() or (
            hasattr(country, 'common_name') and 
            country.common_name.lower() in text.lower()
        ):
            countries.append(country.name)
    return list(set(countries))

def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Returns polarity (-1 to 1) where -1 is negative, 1 is positive
    return analysis.sentiment.polarity

def collect_reddit_data(subreddit_name, limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    
    for post in subreddit.hot(limit=limit):
        # Analyze post content
        post_text = f"{post.title} {post.selftext}"
        countries_in_post = get_countries_from_text(post_text)
        post_sentiment = analyze_sentiment(post_text)
        
        # Get comments
        post.comments.replace_more(limit=0)
        comments_data = []
        
        for comment in post.comments.list():
            countries_in_comment = get_countries_from_text(comment.body)
            if countries_in_comment:  # Only add comments that mention countries
                comments_data.append({
                    'text': comment.body,
                    'countries': countries_in_comment,
                    'sentiment': analyze_sentiment(comment.body)
                })
        
        posts.append({
            'title': post.title,
            'score': post.score,
            'id': post.id,
            'url': post.url,
            'num_comments': post.num_comments,
            'created': post.created,
            'body': post.selftext,
            'countries_mentioned': countries_in_post,
            'post_sentiment': post_sentiment,
            'comments': comments_data
        })
    return posts

# Example usage
subreddit_name = 'travel'
data = collect_reddit_data(subreddit_name, limit=5)

# Print results in a more readable format
for post in data:
    print(f"\nPost: {post['title']}")
    print(f"Countries mentioned in post: {post['countries_mentioned']}")
    print(f"Post sentiment: {post['post_sentiment']:.2f}")
    
    print("\nRelevant comments:")
    for comment in post['comments']:
        print(f"\tCountries: {comment['countries']}")
        print(f"\tSentiment: {comment['sentiment']:.2f}")
        print(f"\tText: {comment['text'][:100]}...")  # Show first 100 chars
    print("-" * 80)