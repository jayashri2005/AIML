#ow to creater a session with increment value

import streamlit as st

st.title("How to create a session")
inp = st.text_input("your question:")

if st.button("Submit"):
    if inp:
        i=int(inp)
        if 'counterr' not in st.session_state:
           st.session_state['counterr'] = i
        st.session_state['counterr'] = st.session_state['counterr'] + 1
        st.write(f"{st.session_state['counterr']}")

if st.button("reset"):
    st.session_state['counterr']=0
    st.write("Session reset to 0")