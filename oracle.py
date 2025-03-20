from dotenv import load_dotenv
import os
import csv
import interface as intf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Carregar variáveis de ambiente
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Criar modelo do Google Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)

#  Carregar a base de informações do CSV corretamente
documents = []
with open("base.csv", encoding="utf-8") as base:
    reader = csv.reader(base, delimiter=";")  # Define o delimitador correto
    next(reader)  # Pular cabeçalho

    for row in reader:
        pergunta, resposta = row[1], row[2]  # Pegando apenas pergunta e resposta por colunas
        documents.append(Document(page_content=f"Pergunta: {pergunta}\nResposta: {resposta}"))

#  Criar embeddings do Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#  Criar o FAISS vectorstore corretamente
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Recupera os 3 mais relevantes

# Prompt para instrucionar a I.A
prompt = ChatPromptTemplate.from_messages([
        ("system", "Responda apenas com base nas informações fornecidas. " 
                   "Se a informação não estiver no contexto abaixo, diga que não sabe."),
        ("human", "{contexto}\n\nPergunta: {pergunta}")
    ])

# Iniciar a interface
intf.iniciar_chat(prompt, llm, retriever)
