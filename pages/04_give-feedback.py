import streamlit as st
import pandas as pd

st.title("Give Feedback")

def cal_avg_score():
    fb_df = pd.read_csv("feedback.csv")
    total = fb_df['score'].sum()+10
    nn = len(fb_df["score"])+1
    return round(total/nn, 2)

def get_num_usr():
    with open('counter.txt') as f:
        contents = f.read()
        return int(contents)

avg_scr = cal_avg_score()
usr_cnt = get_num_usr()
col1, col2 = st.columns(2)
col1.metric('Score', avg_scr)
col2.metric('User used', usr_cnt)

score = st.slider('How good', 0, 10, 1)
ffeedback = st.text_input('Pls enter some feedback :')

if st.button('Submit'):
    feedback_data = pd.read_csv("feedback.csv")
    
    new_feedback = pd.DataFrame({"score": [score], "feedback": [ffeedback]})
    feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)

    feedback_data.to_csv("feedback.csv", index=False)

    st.markdown("![Alt Text](https://media.tenor.com/r1KDajSj-wsAAAAC/thanks-cat.gif)")

old_feedback_data = pd.read_csv("feedback.csv")
