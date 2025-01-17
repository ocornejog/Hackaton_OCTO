import os
from dotenv import load_dotenv
import praw
from textblob import TextBlob
import pycountry

import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

import plotly.express as px
import pandas

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









#Example of how to use OpenAI
import openai
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

llm_model = "gpt-3.5-turbo"
def get_response_from_model(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=llm_model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content
    
prompt = """
I'm giving you a list of posts. These posts are {list_posts}. For each post and each country
mentioned in the post, evaluate if the post gives an opinion on this country
and determine the sentiment of the poster toward that country. If the post doesn't give an opinion on a country and either
doesn't mention any country or alternatively just mention countries without giving an opinion on them, then their opinion will be
considered neutral. Additionally, specify the intensity of the sentiment as a decimal number between 0.000 and 1.000.
text : the full text of the post
country : the country concerned by the sentiment
sentiment : negative, postive or neutral
intensity : the intensity of the sentiment
reasoning : the reasoning behind the sentiment
Format the output as list of JSON with the following keys and without \n: 
text :
country :
sentiment :
intensity :
reasoning :
"""
print(get_response_from_model(prompt.format(list_posts=["I love Italy and Spain", "I kind of dislike the UK", "The population of France is 10000000"])))