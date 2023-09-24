# from dotenv import load_dotenv
# load_dotenv()

from langchain.chat_models import ChatOpenAI
import streamlit as st

st.title("L2M 업데이트 검색")
title = st.text_input('업데이트 관련 질문해주세요', '가장 최근 업데이트 정보를 알려줘')

if st.button('검색'):
    with st.spinner('Wait for it...'):
            st.write('질문 내용: ', title)
    st.success('Done!')

# chat_model = ChatOpenAI()

# result = chat_model.predict("hi!")

# print(result)