from flask import Flask, jsonify, request, render_template
import os
from dotenv import load_dotenv
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,  
)

retriver = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatmodel = Ollama(
    model="llama3",
    base_url="http://localhost:11434"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatmodel, prompt)
rag_chain = create_retrieval_chain(retriver,question_answer_chain)


@app.route('/')
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST", "GET"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("User Message:", input)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response['answer'])
    return str(response['answer'])

if __name__ == '__main__':
    app.run(host= "0.0.0.0", port = 8080, debug=True)

