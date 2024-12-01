import sqlite3
import tensorflow as tf
from transformers import MarianTokenizer, TFMarianMTModel

tf.compat.v1.disable_eager_execution()

class Translator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-fr"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = TFMarianMTModel.from_pretrained(model_name)

    def translate(self, text, source_lang="en", target_lang="fr"):
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class MemoryManager:
    def __init__(self, db_name="chat_memory.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                user_id TEXT,
                input TEXT,
                response TEXT
            )
        ''')
        self.conn.commit()

    def save_interaction(self, user_id, user_input, response):
        self.cursor.execute(
            "INSERT INTO memory (user_id, input, response) VALUES (?, ?, ?)",
            (user_id, user_input, response)
        )
        self.conn.commit()

    def fetch_memory(self, user_id):
        self.cursor.execute("SELECT input, response FROM memory WHERE user_id=?", (user_id,))
        return self.cursor.fetchall()


class MultimodalChatbot:
    def __init__(self):
        self.translator = Translator()
        self.memory_manager = MemoryManager()

    def handle_interaction(self, user_id, user_input, user_lang="en"):
        
        if user_lang != "en":
            user_input = self.translator.translate(user_input, source_lang=user_lang, target_lang="en")

    
        response = f"I understood: {user_input}"

       
        if user_lang != "en":
            response = self.translator.translate(response, source_lang="en", target_lang=user_lang)

 
        self.memory_manager.save_interaction(user_id, user_input, response)

        
        return response

    def get_memory(self, user_id):
        
        return self.memory_manager.fetch_memory(user_id)


# Initialize the chatbot instance
chatbot = MultimodalChatbot()
user_id = "default_user"

# Streamlit App
st.title("Multimodal Chatbot with Translation and Memory")

# Language selection dropdown
user_lang = st.selectbox(
    "Select a language",
    ["en", "fr", "de", "es"],
    index=0
)

# User input textarea
user_input = st.text_area("Type your message here...")

# Submit button
if st.button("Send"):
    if user_input:
        response = chatbot.handle_interaction(user_id, user_input, user_lang)

        # Display chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.session_state.chat_history.append({"user": user_input, "bot": response})

        for chat in st.session_state.chat_history:
            st.write(f"**You**: {chat['user']}")
            st.write(f"**Bot**: {chat['bot']}")
    else:
        st.warning("Please type a message.")

# Show memory button
if st.button("Show Memory"):
    memory = chatbot.get_memory(user_id)
    if memory:
        for idx, (inp, resp) in enumerate(memory, start=1):
            st.write(f"{idx}. You: {inp} | Bot: {resp}")
    else:
        st.write("No memory saved yet.")

