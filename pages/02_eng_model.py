import streamlit as st 
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from bing_image_urls import bing_image_urls

#title
st.title("Manga Recommendation ver.Eng model")

#data
all_title = np.load("all_title.npy")
all_index = np.load("all_index.npy")

path = 'Madnesss/fine-tune-all-MiniLM-L6-v2'
model = SentenceTransformer(path)
embeddings_des = torch.load("embeddings_des.pt", map_location=torch.device('cpu'))

query = st.text_input('Enter query :')

col1, col2, col3 = st.columns(3)
with col1:
    col1 = st.button("query 1")
with col2:
    col2 = st.button("query 2")
with col3:
    col3 = st.button("query 3")

# but1, but2 = st.columns(2)
# but1 = st.button("Only Eng")
# but2 = st.button("Other language")

st.subheader('Example query')
st.text('''query 1 : What is the name of a manga about a middle school student who wants to be a hero but doesn’t have any powers?
query 2 : What is a story about a robot maid?
query 3 : What is the name of the manga about a team of basketball prodigies?''')

if(col1):
    query = "What is the name of a manga about a middle school student who wants to be a hero but doesn’t have any powers?"

if(col2):
    query = "What is a story about a robot maid?"

if(col3):
    query = "What is the name of the manga about a team of basketball prodigies?"


if(query):
    print(query)

    f = open('counter.txt','r+')
    oldscore = (f.read())
    oldscore = int(oldscore)
    score = oldscore + 1
    f.seek(0)
    f.write(str(score))
    f.close()
    query_en = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_en, embeddings_des)

    all_idx = torch.topk(cosine_scores.flatten(), 5).indices
    print(all_idx)
    st.write("We recommend : ")
    for i in range(len(all_idx)):
        title = all_title[all_idx[i]]
        each_score = float(cosine_scores[0][i])*100
        print(i+1, title)
        st.write(f'{i+1}. {title}')
        keyword = title+" manga"
        url = bing_image_urls(keyword, limit=1)[0]
        print(url)

        # show cover
        st.image(url, width=100)
