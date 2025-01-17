import os
from dotenv import load_dotenv
from openai import OpenAI
import praw
import pycountry
import pandas as pd
import plotly.express as px
from nltk.tokenize import word_tokenize
import streamlit as st
from textblob import TextBlob

load_dotenv()

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

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
    # Make detection more strict by requiring country names to be standalone words
    countries = []
    words = word_tokenize(text.lower())
    
    # Add common country variations and abbreviations
    country_variations = {
        'usa': 'United States',
        'uk': 'United Kingdom',
        'uae': 'United Arab Emirates',
    }
    
    # Check for variations first - require them to be standalone words
    for variation, full_name in country_variations.items():
        if variation in words:  # Changed from text.lower()
            countries.append(full_name)
    
    # Then check official names - require them to be standalone words
    for country in pycountry.countries:
        country_name_words = word_tokenize(country.name.lower())
        # Only match if all words of the country name are present in sequence
        if all(word in text.lower() for word in country_name_words):
            countries.append(country.name)
        elif hasattr(country, 'common_name'):
            common_name_words = word_tokenize(country.common_name.lower())
            if all(word in text.lower() for word in common_name_words):
                countries.append(country.name)
    
    # Return only the first country found to avoid multiple matches
    return countries[:1]

def analyze_sentiment(text):
    analysis = TextBlob(text)
    # Returns polarity (-1 to 1) where -1 is negative, 1 is positive
    return analysis.sentiment.polarity

def is_post_relevant_for_country(title, country):
    prompt = f"""Analyze if this post title is relevant for {country}. 
    Title: "{title}"
    Return only 'yes' or 'no'. Consider both explicit mentions and implicit references."""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower() == 'yes'

def analyze_sentiment_openai(text, country):
    prompt = f"""Analyze the sentiment towards {country} in this text. Consider both explicit and implicit references.
    Text: "{text}"
    Rate the sentiment on a scale from -1 (very negative) to 1 (very positive).
    Return only the numerical score."""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.0

def is_opinion_post(title, text):
    prompt = f"""Est-ce que ce post exprime une opinion ou un sentiment à propos d'un pays? 
    Le post peut mentionner un pays même indirectement.
    
    Titre: "{title}"
    Texte: "{text}"
    
    Répondre uniquement par 'yes' ou 'no'."""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower() == 'yes'

def calculate_satisfaction_score(upvotes, downvotes, comments_sentiment):
    # Calculate satisfaction based on vote ratio and comments sentiment
    if upvotes + downvotes == 0:
        vote_ratio = 0
    else:
        vote_ratio = upvotes / (upvotes + downvotes)
    
    # Combine vote ratio with comments sentiment (weighted average)
    if not comments_sentiment:
        return vote_ratio * 2 - 1  # Scale to [-1, 1]
    
    avg_comment_sentiment = sum(comments_sentiment) / len(comments_sentiment)
    # Weight: 70% votes, 30% comments
    return (0.7 * (vote_ratio * 2 - 1)) + (0.3 * avg_comment_sentiment)

def collect_reddit_data(subreddit_name, limit=100000):
    with st.spinner('Analyse des posts Reddit en cours...'):
        try:
            subreddit = reddit.subreddit(subreddit_name)
            country_data = {}
            total_posts = 0
            
            st.info(f"Analyse de r/{subreddit_name}")
            progress_bar = st.progress(0)
            
            
            posts = list(subreddit.hot(limit=min(limit, 1000)))
            if not posts:
                st.warning("Aucun post trouvé dans ce subreddit")
                return {}
                
            for idx, post in enumerate(posts):
                total_posts += 1
                post_text = f"{post.title} {post.selftext}"
                
                # Update progress bar
                progress_bar.progress(idx / limit)
                
                countries = get_countries_from_text(post_text)
                if not countries:
                    continue
                    
                is_opinion = is_opinion_post(post.title, post.selftext)
                if not is_opinion:
                    continue
                    
                # Initialize country data if not exists
                country = countries[0]
                if country not in country_data:
                    country_data[country] = {
                        'satisfaction_sum': 0,
                        'post_count': 0,
                        'mentions_count': 0,
                        'total_upvotes': 0,
                        'total_downvotes': 0,
                        'total_comments': 0,
                        'popularity_ratio': 0
                    }
                
                # Calculate sentiment using OpenAI
                sentiment = analyze_sentiment_openai(post_text, country)
                
                # Update country statistics
                country_data[country]['satisfaction_sum'] += sentiment
                country_data[country]['post_count'] += 1
                country_data[country]['mentions_count'] += 1
                country_data[country]['total_upvotes'] += post.ups
                country_data[country]['total_downvotes'] += post.downs
                country_data[country]['total_comments'] += post.num_comments
                
                # Update popularity ratio
                total_votes = post.ups + post.downs
                if total_votes > 0:
                    country_data[country]['popularity_ratio'] = post.ups / total_votes
                
                # Update progress bar
                progress_bar.progress(idx / limit)
            
            if not country_data:
                st.warning("Aucune mention de pays trouvée dans les posts analysés")
                
            return country_data
            
        except Exception as e:
            st.error(f"Une erreur est survenue: {str(e)}")
            return {}

def create_interactive_visualizations(country_data):
    if not country_data:
        st.error("Aucune données trouvées, re-essayer avec un autre sub-reddit.")
        return
        
    # Prepare data for visualization
    df_data = []
    for country, data in country_data.items():
        try:
            country_code = pycountry.countries.lookup(country).alpha_3
            avg_satisfaction = data['satisfaction_sum'] / data['post_count'] if data['post_count'] > 0 else 0
            
            df_data.append({
                'country': country,
                'country_code': country_code,
                'avg_satisfaction': avg_satisfaction,
                'post_count': data['post_count'],
                'popularity_ratio': data['popularity_ratio'],
                'total_mentions': data['mentions_count'],
                'upvotes': data['total_upvotes'],
                'downvotes': data['total_downvotes'],
                'comments': data['total_comments'],
                'engagement_rate': (data['total_upvotes'] + data['total_comments']) / data['mentions_count'] if data['mentions_count'] > 0 else 0
            })
        except LookupError:
            continue

    df = pd.DataFrame(df_data)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["World Map", "Country Rankings", "Engagement Analysis", "Correlation Analysis"])
    
    with tab1:
        st.subheader("Distribution global des sentiments")
        fig_map = px.choropleth(df,
                           locations='country_code',
                           color='avg_satisfaction',
                           hover_data=['post_count', 'popularity_ratio', 'total_mentions'],
                           color_continuous_scale='RdYlGn',
                           title='Country Satisfaction Analysis from Reddit')
        st.plotly_chart(fig_map, use_container_width=True)
    
    with tab2:
        st.subheader("Country Rankings")
        
        # Top 10 countries by different metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Top 10 Countries by Satisfaction")
            fig_satisfaction = px.bar(
                df.nlargest(10, 'avg_satisfaction'),
                x='country',
                y='avg_satisfaction',
                color='avg_satisfaction',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_satisfaction, use_container_width=True)
        
        with col2:
            st.write("Top 10 Most Mentioned Countries")
            fig_mentions = px.bar(
                df.nlargest(10, 'total_mentions'),
                x='country',
                y='total_mentions',
                color='avg_satisfaction',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_mentions, use_container_width=True)
    
    with tab3:
        st.subheader("Engagement Analysis")
        
        # Scatter plot of mentions vs engagement
        fig_engagement = px.scatter(
            df,
            x='total_mentions',
            y='engagement_rate',
            size='post_count',
            color='avg_satisfaction',
            hover_data=['country'],
            color_continuous_scale='RdYlGn',
            title='Country Engagement Analysis'
        )
        st.plotly_chart(fig_engagement, use_container_width=True)
        
        # Engagement metrics distribution
        col1, col2 = st.columns(2)
        with col1:
            fig_upvotes = px.box(df, y='upvotes', title='Upvotes Distribution')
            st.plotly_chart(fig_upvotes, use_container_width=True)
        with col2:
            fig_comments = px.box(df, y='comments', title='Comments Distribution')
            st.plotly_chart(fig_comments, use_container_width=True)
    
    with tab4:
        st.subheader("Correlation Analysis")
        
        # Correlation matrix
        correlation_matrix = df[['avg_satisfaction', 'post_count', 'popularity_ratio', 
                               'total_mentions', 'engagement_rate']].corr()
        fig_corr = px.imshow(
            correlation_matrix,
            title='Correlation Matrix of Key Metrics',
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    return df

def main():
    st.title("Analyse de sentiment par pays")
    st.write("Hackaton OCTO (part of Accenture)")
    
    with st.sidebar:
        subreddit_name = st.text_input("Nom du subreddit:", value="travel")
        post_limit = st.slider("Nombre de posts à analyser:", 10, 100000, 100)
        
    if st.sidebar.button("Analyser"):
        country_data = collect_reddit_data(subreddit_name, limit=post_limit)
        df = create_interactive_visualizations(country_data)
        
        if df is not None:
            st.subheader("Statistiques clés")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Nombre total de pays analysés", len(df))
                
            with col2:
                avg_satisfaction = df['avg_satisfaction'].mean()
                st.metric("Satisfaction globale moyenne", f"{avg_satisfaction:.2f}")
                
            with col3:
                total_mentions = df['total_mentions'].sum()
                st.metric("Mentions totales de pays", total_mentions)
            
            st.subheader("Statistiques détaillées")
            sorted_df = df.sort_values('avg_satisfaction', ascending=False)
            st.dataframe(sorted_df)
            
            csv = sorted_df.to_csv(index=False)
            st.download_button(
                label="Télécharger les données (CSV)",
                data=csv,
                file_name="analyse_pays_reddit.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()