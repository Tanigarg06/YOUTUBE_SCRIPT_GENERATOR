from dotenv import load_dotenv
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv() 

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)

st.title("📝YouTube Script Writer📝")

topic = st.text_input("Enter the topic of video", placeholder="e.g., The Surprising Benefits of Cold Showers")
duration = st.slider("Select video duration (in minutes)", 1, 20, 5)
tone = st.text_input("How do you want the tone of script", placeholder="e.g., Humorous, Informative, Inspirational")

prompt_template_text = """
You are a professional YouTube scriptwriter.
Generate a script for a YouTube video on the topic: "{topic}".

The script should be approximately {duration} minutes long.

Please write in a {tone} tone.

The script must include:
1.  An engaging hook to grab the viewer's attention.
2.  The main content should be clear.
3.  A call-to-action (e.g., asking viewers to like, subscribe, or check out a link).
4.  An outro.

Here is the structure to follow:
---
**Title:** [A catchy title for the video]

**Hook:**
(Your engaging opening here)

**Introduction:**
(Briefly introduce the topic and what the viewer will learn)

**Main Content:**
(Detailed sections covering the topic)

**Call to Action (CTA):**
(Your CTA here)

**Outro:**
(Your concluding remarks here)
---
"""

# Prompt templates
title_template = PromptTemplate(
    input_variables=["topic", "duration", "tone"], template=prompt_template_text 
)

chain = LLMChain(
    llm=llm,
    prompt=title_template,
    verbose=True,
    output_key="title",
)

if topic:
    input_data = {
        'topic': topic,
        'duration': str(duration),
        'tone': tone.lower()
    }

    with st.spinner("Generating Script..."):
        response = chain.invoke(input_data)
        st.subheader("Your YouTube Script 📜 :")
        st.markdown(response["title"])