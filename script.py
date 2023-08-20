import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

apikey = st.secrets['apikey']
os.environ['OPENAI_API_KEY'] = apikey
st.set_page_config(
    page_title="YouTubeGPT",
    page_icon="📝",
    layout="wide",
)
st.title("🌕🔗YOUTUBE SCRIPT GPT")
prompt = st.text_input("Add the topic of your video here")
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='write me a youtube video title about {topic}'
)
script_template = PromptTemplate(
    input_variables = ['title'],
    template='write me a youtube video script with a video duration of 10 minutes based on this title TITLE: {title}'
)

#memory
memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')



llm = OpenAI(temperature=0.9,max_tokens = 2056)
title_chain = LLMChain(llm=llm,prompt=title_template,verbose=True,output_key='title',memory=memory)
script_chain = LLMChain(llm=llm,prompt=script_template,verbose=True,output_key='script',memory=memory)
sequential_chain = SequentialChain(chains=[title_chain,script_chain],input_variables=['topic'],output_variables=['title','script'],verbose=True)
if prompt:
    response = sequential_chain({'topic': prompt})
    st.write('Title:')
    st.write(response['title'])
    st.write('Video Script:')
    st.write(response['script'])
    with st.expander('Prompt History'):
       st.info(memory.buffer)

