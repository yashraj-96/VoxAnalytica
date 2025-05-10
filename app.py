# app.py - Main application entry point
import os
import streamlit as st
from src.speech import record_audio, transcribe_audio
from src.llm import process_query
from src.data_ops import execute_code, load_dataset
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DataWhisperer", page_icon="🎙️", layout="wide")

def main():
    st.title("🎙️ DataWhisperer")
    st.subheader("Voice-Powered Data Analysis")
    
    # Sidebar for dataset selection
    st.sidebar.header("Dataset Selection")
    dataset_options = ["sales.csv", "customers.csv", "Upload your own..."]
    selected_dataset = st.sidebar.selectbox("Choose a dataset", dataset_options)
    
    if selected_dataset == "Upload your own...":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.current_df = df
            st.sidebar.success(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    else:
        df = load_dataset(f"examples/{selected_dataset}")
        st.session_state.current_df = df
        st.sidebar.success(f"Loaded {selected_dataset} with {len(df)} rows")
    
    # Display dataset preview
    if 'current_df' in st.session_state:
        with st.expander("Dataset Preview", expanded=False):
            st.dataframe(st.session_state.current_df.head())
    
    # Voice recording section
    st.header("Ask Your Data")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Text input as alternative to voice
        text_query = st.text_input("Type your query or use voice button")
    
    with col2:
        # Voice recording button
        if st.button("🎤 Record Voice", type="primary"):
            with st.spinner("Recording... (speak now)"):
                audio_file = record_audio(duration=5)  # 5 seconds recording
                
            with st.spinner("Transcribing..."):
                transcription = transcribe_audio(audio_file)
                st.session_state.transcription = transcription
                
            st.success("Transcribed successfully!")
    
    # Display transcription if available
    if 'transcription' in st.session_state and not text_query:
        st.info(f"Transcribed query: {st.session_state.transcription}")
        active_query = st.session_state.transcription
    else:
        active_query = text_query
    
    # Process the query
    if active_query:
        with st.spinner("Analyzing your query..."):
            # Process through LLM to get executable code
            code = process_query(active_query, st.session_state.current_df)
            
            # Display the generated code
            with st.expander("Generated Code", expanded=False):
                st.code(code, language="python")
            
            # Execute the code
            result, fig = execute_code(code, st.session_state.current_df)
            
            # Display results
            st.header("Results")
            
            # If there's a figure, display it
            if fig:
                st.pyplot(fig)
                plt.close(fig)  # Clean up
            
            # If there's a dataframe result, display it
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            # If it's a scalar or simple result
            elif result is not None:
                st.success(result)
    
    # Add query to history
    if active_query and active_query not in st.session_state.get('history', []):
        if 'history' not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append(active_query)
    
    # Display history
    if 'history' in st.session_state and st.session_state.history:
        with st.sidebar.expander("Query History", expanded=True):
            for i, query in enumerate(st.session_state.history):
                st.sidebar.button(
                    query, 
                    key=f"history_{i}",
                    on_click=lambda q=query: st.session_state.update({"transcription": q})
                )

if __name__ == "__main__":
    main()