# CommEntiMent
nytimes Comment Entity sentiMent: An application built by [Streamlit](https://www.streamlit.io/).
This tool scraps the comments written on the comment section of a given nytimes article provided by the user. 
Then it uses [spaCy](https://spaCy.io) to perform [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition):
recognition of entities such as persons, locations, etc. In particular it takes advantage of a pretrained statistical model of 
the English language [en\_core\_web\_sm](https://spacy.io/models/en).

After finding the named entities, it uses a rule-based sentiment analyzer called [vaderSentiment](https://github.com/cjhutto/vaderSentiment) 
to assess the distribution of sentiment polarity associated with the named entities mentioned in the comments.

You can run the app local by simply running this command:

```streamlit run app.py```
