import streamlit as st 
import pandas as pd 

st.title("Show feedback")

password = st.text_input("Enter a password", type="password")

if(password=='thepasswordispassword'):
    datafb = pd.read_csv("feedback.csv")
    st.dataframe(datafb)