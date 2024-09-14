import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client with the API key from the environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("API key not found. Please set the GROQ_API_KEY in your .env file.")
else:
    client = Groq(api_key=api_key)

# Step 1: Transcribe the audio file using Whisper
def transcribe_audio(file_path):
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, file.read()),
            model="whisper-large-v3",
            language="en",
            response_format="json",
            temperature=0.0
        )
        return transcription.text

# Step 2: Summarize the transcription using the LLM
def summarize_text(text):
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"Summarize this transcription: {text}"
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    summary = ""
    for chunk in completion:
        summary += chunk.choices[0].delta.content or ""

    return summary

# Step 3: Handle user questions based on transcription and summary
def ask_question(text, question):
    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"Using the following information: {text}. Now, answer the question: {question}"
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""

    return answer

# Streamlit app
def main():
    # Center the image
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("icon.png", width=300)  # Adjust the file path and width as needed

    # Initialize session state to store transcription and summary
    if "transcription_text" not in st.session_state:
        st.session_state.transcription_text = None

    if "summary" not in st.session_state:
        st.session_state.summary = None

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.write("Processing the audio file...")

        # Only transcribe and summarize if not already done
        if st.session_state.transcription_text is None:
            st.session_state.transcription_text = transcribe_audio(file_path)
        st.subheader("Transcription")
        st.write(st.session_state.transcription_text)

        if st.session_state.summary is None:
            st.session_state.summary = summarize_text(st.session_state.transcription_text)
        st.subheader("Summary")
        st.write(st.session_state.summary)

    # Chatbox for user questions
    if st.session_state.transcription_text and st.session_state.summary:
        st.subheader("Ask questions about the content")
        user_question = st.text_input("Enter your question")

        if st.button("Ask"):
            if user_question:
                # Use the stored transcription and summary to answer the user's question
                response = ask_question(f"{st.session_state.transcription_text}\n\nSummary: {st.session_state.summary}", user_question)
                st.write("Answer:")
                st.write(response)
            else:
                st.write("Please enter a question.")

if __name__ == "__main__":
    main()
