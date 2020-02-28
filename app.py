import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
from nytimes_scraper.nyt_api import NytApi
from nytimes_scraper.articles import fetch_articles_by_month, articles_to_df
from nytimes_scraper.comments import fetch_comments, fetch_comments_by_article, comments_to_df
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from visualizer import visualize

#Intializing the SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

st.title("CommEntiMent : Recognition of Named Entities in the comment section of New York Times articles and the sentiment attached to them")
#API
api = NytApi('<your_api_key>')
#SPACY MODEL FOR PERFORMING NER
SPACY_MODEL_NAMES = ["en_core_web_sm"] #, "en_core_web_md", "de_core_news_sm"]

DEFAULT_URL = 'https://www.nytimes.com/2020/02/27/health/coronavirus-testing-california.html'
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)


st.sidebar.title("CommEntiMent (Comment Entity sentiMent)")
st.sidebar.markdown(
"""
For a given article in New York Times (URL provided by the user),
scraps the comments, processes them the [spaCy](https://spacy.io) model 
"core_web_sm" to find the named entities. Then for each entity, sentiment analysis 
is done across the comments. \n 
Sentiment analysis is done with VaderSentiment, a library built on 
NLTK and tuned to perform very well on social media platforms.
""")

spacy_model = st.sidebar.selectbox("Model name", SPACY_MODEL_NAMES)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model(spacy_model)
model_load_state.empty()
#grap the input article url provided by the user
article_url = st.text_area("Nytimes article to analyze", DEFAULT_URL)
#fetch the comments
comments = fetch_comments_by_article(api, article_url)
print(comments)
#put the comments in a dataframe
comments_df = pd.DataFrame(comments)
st.write("These are some of the scrapped comments from the article URL")
st.write(comments_df.head())

st.header("Named Entities")
st.sidebar.header("Named Entities")
default_labels = ["PERSON", "ORG", "GPE", "LOC"]
labels = st.sidebar.multiselect(
        "Entity labels", nlp.get_pipe("ner").labels, default_labels)

def process_all_comments(comments_df):
    
    df_list = []
    ncomments = len(comments_df)
    for i in range(ncomments):
        text = comments_df['commentBody'][i]
        doc = process_text(spacy_model, text)
        if i == 0:
            final_df = process_one_comment(text, doc, visualize = True)
        else:
            final_df = process_one_comment(text, doc, visualize = False)
        df_list.append(final_df)

    df_all = pd.concat(df_list, axis = 0)
    
    for ent in ['PERSON', 'LOC', 'ORG']:
        visualize(ent, df_all)


    return None    


def process_one_comment(text, doc, visualize = False):
    
    #Extract the Sentiment Score from the comment body
    vs = analyzer.polarity_scores(text)    
    
    #If vizualize = True, then render the comment body with the highlighted named entities
    if visualize == True:
        html = displacy.render(doc, style="ent", options={"ents": labels})
        # Newlines seem to mess with the rendering
        html = html.replace("\n", " ")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    
    #Extract the NER attributes from the spaCy model 
    attrs = ["text", "label_"]
    if "entity_linker" in nlp.pipe_names:
        attrs.append("kb_id_")
    data = [
        [str(getattr(ent, attr)) for attr in attrs]
        for ent in doc.ents
        if ent.label_ in labels]
    #Named Entity recognition DataFrame
    ner_df = pd.DataFrame(data, columns=attrs, index=range(len(data)))
    #Sentiment DataFrame
    sentiment_df = pd.DataFrame([vs.values()], index=range(len(ner_df)), columns=vs.keys())
    #Combined DataFrame 
    final_df = pd.concat([ner_df, sentiment_df], axis =1)
    
    #If visualize = True, then show the combined dataframe
    if visualize == True:
        st.dataframe(final_df)
    
    return final_df

process_all_comments(comments_df)
