import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Global Settings for Bengali Support
Settings.llm = OpenAI(model="gpt-4o", temperature=0.3)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# --- Page Config ---
st.set_page_config(page_title="Bangladeshi Chef AI", page_icon="🥘", layout="centered")
st.title("Bangladeshi Chef")
st.markdown("আপনার ব্যক্তিগত AI শেফ। যেকোনো রেসিপির জন্য জিজ্ঞাসা করুন!")

# --- Setup LlamaIndex ---
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner("রেসিপি বই পড়া হচ্ছে... (Indexing recipes)..."):
        # Ensure your PDF is in the 'data' folder
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
    # Bengali System Prompt
    bengali_system_prompt = """
    আপনি একজন বিশেষজ্ঞ বাংলাদেশী শেফ (বাবুর্চি)। আপনার নাম 'রন্ধন ওস্তাদ'।
    
    নির্দেশাবলী:
    ১. উত্তর সর্বদা বাংলায় দেবেন (যদি না ব্যবহারকারী ইংরেজিতে চায়)।
    ২. রান্নার প্রতিটি ধাপ স্পষ্টভাবে ব্যাখ্যা করবেন।
    ৩. উপকরণের ক্ষেত্রে খাঁটি নাম ব্যবহার করবেন (যেমন: সয়াবিন তেলের বদলে 'সরিষার তেল', 'পাঁচ ফোড়ন')।
    ৪. ব্যবহারকারীর সাথে বন্ধুত্বপূর্ণ আচরণ করবেন।
    """

    chat_engine = index.as_chat_engine(
        chat_mode="context", 
        system_prompt=bengali_system_prompt
    )

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "আসসালামু আলাইকুম! আজ কী রান্না করতে চান?"}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("প্রশ্ন করুন (যেমন: কাচ্চি বিরিয়ানি কীভাবে রাঁধবো?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("শেফ চিন্তা করছেন..."):
                response = chat_engine.chat(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})