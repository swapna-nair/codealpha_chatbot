import streamlit as st
from chatbot import load_faqs, get_best_match

st.title("FAQ Chatbot")

faqs = load_faqs()

user_question = st.text_input("Ask a question")

if user_question:
    answer = get_best_match(user_question, faqs)
    st.write("**Answer:**")
    st.write(answer)