import streamlit as st 
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from bing_image_urls import bing_image_urls

#title
st.title("Manga Recommendation ver.Mutilingual model")

#data
all_title = np.load("all_title.npy")
all_index = np.load("all_index.npy")

@st.cache_resource 
def load_model():
    path = 'Madnesss/fine-tune-paraphrase-multilingual-mpnet-base-v2'
    return SentenceTransformer(path)

model = load_model()
embeddings_des = torch.load("embeddings_des_multi.pt", map_location=torch.device('cpu'))


query1 = st.text_input('Enter query :')

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
st.text('''query 1 : นักสืบตายแล้ว
query 2 : 농구 신동
query 3 : ปวดตับ''')

if(col1):
    query1 = "นักสืบตายแล้ว"

if(col2):
    query1 = "농구 신동"

if(col3):
    query1 = "ปวดตับ"


if(query1):
    print(query1)

    f = open('counter.txt','r+')
    oldscore = (f.read())
    oldscore = int(oldscore)
    score = oldscore + 1
    f.seek(0)
    f.write(str(score))
    f.close()
    query_en = model.encode(query1, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_en, embeddings_des)

    all_idx = torch.topk(cosine_scores.flatten(), 5).indices
    print(all_idx)
    st.write("We recommend : ")
    for i in range(len(all_idx)):
        title = all_title[all_idx[i]]
        print(i+1, title)
        score_each = cosine_scores[0][all_idx[i]]*100
        print("score:", cosine_scores[0][all_idx[i]])
        st.write(f'{i+1}. {title}')
        st.write(f'{round(float(score_each), 2)}% match')
        keyword = title+" manga"
        url = bing_image_urls(keyword, limit=2)[0]
        print(url)

        # show cover
        st.image(url, width=100)
