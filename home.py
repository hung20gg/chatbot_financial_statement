import streamlit as st
st.set_page_config(
    page_title="Chatbot",
    page_icon="graphics/Icon-BIDV.png" 
)

st.title('Welcome!!!')

st.header('Notice')
st.markdown("""
The chatbot only works with question related to financial statements. Other questions will either not be answered, or the chatbot will hallucinate
Sample questions:
- What is the revenue of the company VIC?
- What is the CASA of banking industry from 2016 to 2023?
- Top 5 bank with the highest Bad Debt Ratio in Q3 2024?
""")

st.header('Chat')
st.markdown("""
- Begin by typing your question or topic in the chatbox below.
- Our chatbot can help answer your queries, provide guidance in financial statements.
- Simply type a message and click **Send**. The chatbot will respond with helpful insights and suggestions.
""")


st.header('Reasoning')
st.markdown("""
- Debugging and reasoning are important in the development of a chatbot.
""")


st.header('Text2SQL')
st.markdown("""
Directly interact with the text2sql model. Only use for debugging purposes.
""")