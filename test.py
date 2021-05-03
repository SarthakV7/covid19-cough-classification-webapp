import time, os
import streamlit as st
import numpy as np
from sound import sound
import SessionState

title = "Guitar Chord Recognition"
st.title(title)
session_state = SessionState.get(name='', path=None)

if st.button('Record'):
    with st.spinner(f'Recording for 5 seconds ....'):
        session_state.path = sound.record()
        st.write(session_state.path)
    st.success("Recording completed")

if st.button('Play'):
    audio_file = open(session_state.path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')
