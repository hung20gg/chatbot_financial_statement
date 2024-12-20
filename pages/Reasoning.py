import os

current_dir = os.path.dirname(__file__)


import streamlit as st
import json

# Load JSON data (simulating loading from a file)


st.set_page_config(
    page_title="Reasoning",
    page_icon="graphics/Icon-BIDV.png" 
)

st.markdown("### Chat Messages")

def dev():
    with open('temp/sql_history.json', 'r') as file:
        chat_data = json.load(file)


    # Render each message in markdown format
    for message in chat_data:
        if message['role'] == 'user':
            with st.chat_message(name="user", avatar="graphics/user.jpg"):
                st.write(message['content'])
        if message['role'] == 'assistant':
            with st.chat_message(name="assistant", avatar="graphics/assistant.png"):
                st.write(message['content'])


users = {
    'dev': 'ngokienhuy123',

}     
     

def login():
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        if users.get(username, r'!!@@&&$$%%.') == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f'Welcome back {username}')
        else:
            st.error('Invalid username or password')
            
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success('You have been logged out')
    
    
def main():
    
    if st.session_state.logged_in:
        st.write(f'Logged in as {st.session_state.username}')
        dev()
    else:
        st.title('Welcome!!!')
        st.write('Press Login button 2 times to login')
        login()
        
main()