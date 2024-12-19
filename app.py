from dataclasses import dataclass
from typing import Literal
import streamlit as st
from datetime import datetime
from groq import Groq

# Initialize Groq client
client = Groq()

# Define the Message data class
@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

# Function to initialize session state
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "streaming_response" not in st.session_state:
        st.session_state.streaming_response = ""
    if "final_report" not in st.session_state:
        st.session_state.final_report = "" 

# Function to generate a response using Groq with streaming
def generate_response_with_streaming(prompt):
    st.session_state.streaming_response = ""  # Clear previous response
    messages = [
        {"role": "system", "content": "You are a helpful and uplifting assistant."},
        {"role": "user", "content": prompt},
    ]

    stream = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        stream=True,  # Enable streaming
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:  # Only process non-None content
            st.session_state.streaming_response += content
            yield content  # Yield content for live streaming in Streamlit

# Function to handle chat submissions
def on_click_callback():
    human_prompt = st.session_state.human_prompt
    st.session_state.history.append(Message("human", human_prompt))

    # Placeholder for streaming response
    response_placeholder = st.empty()
    response_content = ""

    # Stream the response in real-time
    for chunk in generate_response_with_streaming(human_prompt):
        response_content += chunk
        response_placeholder.text(response_content)  # Update text in real-time

    # Clear placeholder after streaming finishes
    response_placeholder.empty()

    # Add the final response to history
    st.session_state.history.append(Message("ai", response_content))

# Function to generate the psychiatrist report
def generate_report():
    """
    Generates a psychiatrist-style summary report based on the conversation history.
    Uses Groq to summarize the key points.
    """
    if not st.session_state.history:
        st.warning("No conversation to summarize!")
        return

    # Extract conversation history
    conversation_history = "\n".join(
        [
            f"User: {msg.message}" if msg.origin == "human" else f"AI: {msg.message}"
            for msg in st.session_state.history
        ]
    )

    # Generate summary using Groq
    messages = [
        {"role": "system", "content": "You are a psychiatrist summarizing a conversation. You want to outline the things that were noted, new insights on subject and final impressions and plan for improvement."},
        {"role": "user", "content": f"Summarize the following conversation:\n{conversation_history}"},
    ]

    response = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=512,
    )

    summary = response.choices[0].message.content

    # Generate the final report
    report = f"""
    Psychiatrist Report
    --------------------
    Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Conversation Summary:
    {summary}
    
    """
    st.session_state.final_report = report

# Function to handle "End Conversation" button
def end_conversation():
    generate_report()
    st.success("Conversation ended and summarized!")
    st.download_button("Download Report", st.session_state.final_report, file_name="psychiatrist_report.txt")

# Initialize session state
initialize_session_state()

# Streamlit app layout
st.title("Mood Minder 😊")
st.markdown(
    "Welcome to **Mood Minder**, your personal mental uplifter! Share your thoughts, "
    "ask for advice, or just chat about anything to brighten your mood."
)

# Chat history display
st.subheader("Conversation")
if st.session_state.history:
    for chat in st.session_state.history:
        if chat.origin == "human":
            st.text(f"🧑 You: {chat.message}")
        else:
            st.text(f"🤖 Luna: {chat.message}")  # Luna is the assistant's name

# User input form
with st.form("chat-form"):
    st.text_input("What's on your mind?", key="human_prompt", placeholder="Type your message here...")
    submit = st.form_submit_button("Send", on_click=on_click_callback)

# End Conversation button
st.button("End Conversation", on_click=end_conversation)

# Display the final report if generated
if st.session_state.final_report:
    st.subheader("Final Report")
    st.text(st.session_state.final_report)
