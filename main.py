import streamlit as st 
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from bing_image_urls import bing_image_urls


st.set_page_config(page_title="Manga Recommendation",page_icon="📚")

st.title("Manga Recommendation 📚")

df = pd.read_csv("data/df.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_des = torch.load("data/embeddings_des.pt")
all_index = np.load("data/all_index.npy")

query = st.text_input('Enter query : ex. What manga has a character who is smooth and cool at work but a total softie at home?')

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
        title = df.loc[all_index[all_idx[int(i)]], "title"]
        print(i+1, title)
        st.write(f'{i+1}. {title}')
        url = bing_image_urls(title, limit=1)[0]
        print(url)

        # show cover
        st.image(url, width=100)
