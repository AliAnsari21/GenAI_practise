from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st
load_dotenv()

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)

st.header('Research Tool')

paper_input=st.selectbox("select research paper name",["Attention Is All You Need","BERT:Pre-Training of Deep Bidirectional Transformers","GPT-3:Language Models are Few-Shot Learners","Diffusion Model Beat GANs on Image synthesis"])
style_input=st.selectbox("select explanation style",["Beginer-Friendly","Technical","Code-Oriented","Mathematical"])
length_input=st.selectbox("select explanation length",["short(1-2 paragraph)","medium(3-5 paragraph)","long(detailed explanation)"])

template=PromptTemplate(
    template="""
please summarize the research paper titled"{paper_input}" with the following specifications:
Explanation Style:{style_input}
explanation length:{length_input}
1. Mathematical Details:
-Include relevant mathematical equations if present in the paper.
-explain the mathematical concepts using simple,intuitive code snippets where applicable.
2. Analogies:
-use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficien information available" instead of guessing.
ensure the summary is clear, accurate and aligned with the provided style and length.
""",
input_variables=['paper_input','style_input','length_input'],
validate_template=True
)

#fill the place holder
prompt=template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input,
})

if st.button('summarize'):
    result=model.invoke(prompt)
    st.write(result.content)

#you can also use chain
'''
if st.button('summarize'):
   chain=template|model
   result=chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input,
   })
   st.write(result.content) 
'''