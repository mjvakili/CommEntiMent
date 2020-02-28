import streamlit as st
import seaborn as sns
sns.set(style="darkgrid")

def visualize(type_entity, df_all):
    '''
    input: type_entity: category of the Named entity
           df_all: dataframe containing the recognized 
           named entities and their corresponding sentiment 
           scores.
    output: make plots of the distribution of Named Entities in 
            order of the number of appearances. 
            Show the distribution of compound sentiment scores 
            associated to the top Named Entitties in each category.
    '''
    df_person = df_all[df_all.label_ == str(type_entity)]
    
    st.write('The dataset of the '+str(type_entity)+' entities extracted from the comments.') 
        #The negative, positive, and neutral fractions of the comment in which an entity is appeared are provided in the dataset.
        #The compound score determines the sentiment polarity of each comment with negative (positive) one indicating the 
        #most negative (positive) sentiment.')
    st.dataframe(df_person)
    
    st.write('The most frequently mentioned '+str(type_entity)+' entities are:')
    ax = sns.countplot(y="text", data=df_person, 
        order=df_person.text.value_counts().iloc[:3].index)
    st.pyplot(dpi=100)

    
    top_person = df_person.text.value_counts().iloc[:3].index.tolist()
    for person in top_person:
        st.write('The sentiment distribution for comments containing the entity '+person+":")
        sns.set(style="darkgrid")
        ax = sns.distplot(df_person[df_person.text == person]["compound"], kde = False, hist_kws={"histtype": "step", "linewidth": 3,
            "alpha": 1, "color": "g", "range": (-1, 1)})
        st.pyplot(dpi=100)
    
    return None
