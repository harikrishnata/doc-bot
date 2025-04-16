import os
import re
import streamlit as st
import requests
from typing import Optional, List

from langchain.llms.base import LLM
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# --- Custom DeepSeekLLM Implementation ---
class DeepSeekLLM(LLM):
    model: str = "deepseek-chat"
    api_key: str = "sk-4219a421c96d4698ac43d71aa0a6b3e1"
    base_url: str = "https://api.deepseek.com/v1/chat/completions"
    temperature: float = 0.7

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "deepseek"

# --- Initialize DeepSeek LLM ---
llm = DeepSeekLLM(temperature=0.7)

# --- Configuration & Briefs ---
BRIEFS = {
    "Introductory Brief": (
        "Welcome to the medical simulation chatbot. In this exercise, you will be interacting "
        "with a virtual patient as a medical student. First, you will take the patient’s history. "
        "Later, the scenario will shift: you will become a student doctor, and I will act as the examiner. "
        "If you stray off-topic, I will gently remind you of the context."
    ),
    "Alternate Brief Example": (
        "This is an alternative simulation briefing. You are required to act as a junior doctor. "
        "Start the consultation with a patient presenting with chest pain. Once the history-taking is complete, "
        "the interaction will transition to an exam mode where your performance will be evaluated."
    )
}

PATIENT_PROFILE = (
    "Patient Information:\n"
    "• 55-year-old man in the emergency department.\n"
    "• Presenting complaint: severe central chest pain for four hours.\n"
    "• History includes sudden onset, sharp pain, radiation to the back, and "
    "associated with concerns of heart attack.\n"
    "• Past medical history: poorly controlled hypertension, on Losartan, Amlodipine, and Indapamide.\n"
    "• Social history: ex-smoker, builder by profession.\n"
    "• Diagnosis: Aortic Dissection.\n"
)

# --- Initialize Session State for Memories ---
if "history_memory" not in st.session_state:
    st.session_state.history_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "examiner_memory" not in st.session_state:
    st.session_state.examiner_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Create ConversationChain with custom input key ---
def create_chain(prompt):
    return ConversationChain(
        llm=llm,
        memory=st.session_state.current_memory,
        prompt=PromptTemplate(
            input_variables=["chat_history", "human_input"],
            template=prompt
        ),
        input_key="human_input"
    )

# --- Prompt Templates with Self-Introductions ---
HISTORY_PROMPT = (
    "You are playing the role of a patient. Please begin the conversation by introducing yourself briefly before answering any questions.\n"
    f"{PATIENT_PROFILE}\n"
    "Answer questions as the patient in a realistic manner, providing details as necessary. "
    "If the user asks something off-topic or unrelated to the patient’s symptoms or history, kindly remind "
    "them to focus on the history-taking.\n"
    "Chat History:\n"
    "{chat_history}\n"
    "User (Medical Student): {human_input}\n"
    "Patient:"
)

EXAMINER_PROMPT = (
    "You are the examiner. Please begin by briefly introducing yourself and explaining your role before proceeding with follow-up questions.\n"
    "Your role is to assess the student doctor after the history-taking session. Ask follow-up questions or provide feedback to evaluate the thoroughness of the history-taking. "
    "If the discussion drifts away from the clinical context, guide the conversation back.\n"
    "Chat History:\n"
    "{chat_history}\n"
    "Examiner (You): {human_input}\n"
    "Examiner:"
)

# --- UI Layout ---
st.title("Medical History Chatbot Simulator")

# Sidebar for simulation setup
st.sidebar.header("Simulation Setup")
brief_choice = st.sidebar.selectbox("Select a Brief", list(BRIEFS.keys()))
simulation_stage = st.sidebar.radio("Select Stage", ("Brief", "History Taking", "Examiner"))

# Provision to add new briefs
with st.sidebar.expander("Add/Edit Briefs"):
    new_brief_title = st.text_input("New Brief Title")
    new_brief_text = st.text_area("New Brief Content")
    if st.button("Add Brief"):
        if new_brief_title.strip() and new_brief_text.strip():
            BRIEFS[new_brief_title] = new_brief_text
            st.success(f"Added brief: {new_brief_title}")
        else:
            st.error("Both title and content are required to add a new brief.")

# Reset conversation histories when switching stages
if "current_stage" not in st.session_state or st.session_state.current_stage != simulation_stage:
    st.session_state.current_stage = simulation_stage
    if simulation_stage == "History Taking":
        st.session_state.current_memory = st.session_state.history_memory
        st.session_state.conversation_chain = create_chain(HISTORY_PROMPT)
        # Automatically add patient introduction if no previous messages
        if not st.session_state.current_memory.buffer:
            st.session_state.current_memory.chat_memory.add_ai_message(
                "Hello, I'm John. I've been experiencing severe chest pain for the past four hours, and it's very concerning."
            )
    elif simulation_stage == "Examiner":
        st.session_state.current_memory = st.session_state.examiner_memory
        st.session_state.conversation_chain = create_chain(EXAMINER_PROMPT)
        # Automatically add examiner introduction if no previous messages
        if not st.session_state.current_memory.buffer:
            st.session_state.current_memory.chat_memory.add_ai_message(
                "Hello, I'm the examiner for today's session. I'll be asking follow-up questions to evaluate your history-taking skills."
            )
    else:
        st.session_state.current_memory = None
        st.session_state.conversation_chain = None

# --- Simulation Stages ---
if simulation_stage == "Brief":
    st.subheader("Simulation Brief")
    st.info(BRIEFS[brief_choice])
    st.write("Please review the brief. Once ready, switch to the 'History Taking' stage to begin your consultation.")

elif simulation_stage in ["History Taking", "Examiner"]:
    st.subheader(f"{simulation_stage} Stage")
    st.write("This is an interactive session. Type your message below and press Enter to send.")
    
    # Display conversation history if available
    if st.session_state.current_memory is not None:
        messages = st.session_state.current_memory.buffer
        if messages:
            st.markdown("**Conversation History:**")
            st.text_area("History", value=str(messages), height=300, key="history_display")
    
    # User input
    user_input = st.text_input("Your Input:", key="user_input")
    if st.button("Send"):
        if user_input.strip():
            response = st.session_state.conversation_chain.predict(human_input=user_input)
            st.write("**Bot Response:**")
            st.write(response)
        else:
            st.warning("Please enter a message.")

# --- Reset Conversation Button ---
if st.sidebar.button("Reset Conversation"):
    if simulation_stage == "History Taking":
        st.session_state.history_memory.clear()
    elif simulation_stage == "Examiner":
        st.session_state.examiner_memory.clear()
    st.success("Conversation history cleared.")
