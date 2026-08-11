from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
st.header('Research Tool')
user_input=st.text_input('enter your prompt')
if st.button('summarize'):
    result=model.invoke(user_input)
    st.write(result.content)