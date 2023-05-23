import streamlit as st
import pandas as pd

st.title("Give Feedback")

score = st.slider('How good', 0, 10, 1)
ffeedback = st.text_input('Pls enter some feedback :')

if st.button('Submit'):
    feedback_data = pd.read_csv("feedback.csv")
    
    new_feedback = pd.DataFrame({"score": [score], "feedback": [ffeedback]})
    feedback_data = pd.concat([feedback_data, new_feedback], ignore_index=True)

    feedback_data.to_csv("feedback.csv", index=False)

    st.markdown("![Alt Text](https://tenor.com/en-GB/view/catthankyou-cat-gif-24824893)")

old_feedback_data = pd.read_csv("feedback.csv")

# with open('feedback.csv', 'rb') as file:
#     csv = file.read()
#     st.download_button(label='📥 Download Feeddback', data=csv, file_name='feedback.csv')