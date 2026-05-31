import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv
from database import init_db, create_session, save_message, load_messages, load_all_sessions

load_dotenv()
init_db()

# Configure Global Settings for Bengali Support
Settings.llm = OpenAI(model="gpt-4o", temperature=0.3)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# --- Page Config ---
st.set_page_config(page_title="Deshi Chef AI", page_icon="🥘", layout="centered")
st.title("Deshi Chef")
st.markdown("আপনার ব্যক্তিগত AI শেফ। যেকোনো রেসিপির জন্য জিজ্ঞাসা করুন!")

# --- Setup LlamaIndex ---
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner("রেসিপি বই পড়া হচ্ছে... (Indexing recipes)..."):
        if not os.path.exists("./data"):
            st.error("Please create a folder named 'data' and put your Recipe PDF inside.")
            return None

        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)
        return index

# Initialize Index
index = load_data()

if index:
    bengali_system_prompt = """
    আপনি একজন বিশেষজ্ঞ বাংলাদেশী শেফ (বাবুর্চি)।

    নির্দেশাবলী:
    ১. উত্তর সর্বদা বাংলায় দেবেন (যদি না ব্যবহারকারী ইংরেজিতে চায়)।
    ২. রান্নার প্রতিটি ধাপ স্পষ্টভাবে ব্যাখ্যা করবেন।
    ৩. উপকরণের ক্ষেত্রে খাঁটি নাম ব্যবহার করবেন (যেমন: সয়াবিন তেলের বদলে 'সরিষার তেল', 'পাঁচ ফোড়ন')।
    ৪. ব্যবহারকারীর সাথে বন্ধুত্বপূর্ণ আচরণ করবেন।
    """

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        system_prompt=bengali_system_prompt
    )

    # --- Sidebar: session management ---
    with st.sidebar:
        st.header("Chat History")
        if st.button("New Chat"):
            st.session_state.session_id = create_session()
            st.session_state.messages = []

        past_sessions = load_all_sessions()
        if past_sessions:
            session_labels = {
                s["id"]: f"Session {s['id']} — {s['created_at'][:10]}"
                for s in past_sessions
            }
            selected_id = st.selectbox(
                "Load a past session",
                options=[s["id"] for s in past_sessions],
                format_func=lambda sid: session_labels[sid],
            )
            if st.button("Load"):
                st.session_state.session_id = selected_id
                st.session_state.messages = load_messages(selected_id)
                st.rerun()

    # --- Chat Interface ---
    GREETING = "আসসালামু আলাইকুম! আজ কী রান্না করতে চান?"

    if "session_id" not in st.session_state:
        st.session_state.session_id = create_session()

    if "messages" not in st.session_state:
        st.session_state.messages = load_messages(st.session_state.session_id)
        if not st.session_state.messages:
            save_message(st.session_state.session_id, "assistant", GREETING)
            st.session_state.messages = [{"role": "assistant", "content": GREETING}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("প্রশ্ন করুন (যেমন: কাচ্চি বিরিয়ানি কীভাবে রাঁধবো?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.session_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("শেফ চিন্তা করছেন..."):
                response = chat_engine.chat(prompt)
                reply = response.response
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                save_message(st.session_state.session_id, "assistant", reply)
