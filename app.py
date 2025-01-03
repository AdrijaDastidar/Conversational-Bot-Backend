import os
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from flask_cors import CORS
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
from waitress import serve
from prompt import *

app = Flask(__name__, static_folder="src/static", template_folder="src/templates")
CORS(app, supports_credentials=True,)
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env file")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

model_name = "llama3-8b-8192"
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name, temperature=0.4, max_tokens=500)

conversational_memory_length = 5
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

def generate_response(text):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )
    
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    
    return conversation.predict(human_input=text)

@app.route("/res", methods=["OPTIONS", "POST"])
def res():
    if request.method == "OPTIONS":
        response = app.response_class(
            response="", status=204, headers={"Access-Control-Allow-Origin": "http://localhost:5173"}
        )
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    try:
        data = request.get_json()
        query = data.get("query")

        if not query:
            return jsonify({"error": "No query provided"}), 400

        bot_response = generate_response(query)
        return jsonify({"response": bot_response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)
