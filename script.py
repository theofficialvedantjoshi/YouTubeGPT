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
prompt_time = st.text_input("Add the duration of your video here(Min;sec)")
prompt_tone= st.text_input("Add the tone of your video")
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='write me a youtube video title about {topic}'
)
script_template = PromptTemplate(
    input_variables = ['title','time','tone'],
    template='write me a detailed youtube video script with this time duration DURATION:{time}, based on this title TITLE: {title} and in a {tone} tone'
)

#memory
memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')



llm = OpenAI(temperature=0.9,max_tokens = 2056)
title_chain = LLMChain(llm=llm,prompt=title_template,verbose=True,output_key='title',memory=memory)
script_chain = LLMChain(llm=llm,prompt=script_template,verbose=True,output_key='script',memory=memory)
sequential_chain = SequentialChain(chains=[title_chain,script_chain],input_variables=['topic','time','tone'],output_variables=['title','script'],verbose=True)
if prompt and prompt_time and prompt_tone:
    response = sequential_chain({'topic': prompt,'time': prompt_time,'tone': prompt_tone})
    st.write('Title:')
    st.write(response['title'])
    st.write('Video Script:')
    st.write(response['script'])
    with st.expander('Prompt History'):
       st.info(memory.buffer)
