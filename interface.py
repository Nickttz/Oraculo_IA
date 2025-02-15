import streamlit as st

def iniciar_chat(prompt, llm, retriever):
    st.title("🤖 Chat com I.A")
    st.subheader("Interaja com uma IA que responde perguntas sobre o universo de Dragon Ball Z.")
    pergunta = st.text_input("Digite aqui: ")
    
    if pergunta:
        result_docs = retriever.invoke(pergunta)
        
        contexto = ""
        for doc in result_docs:
            contexto += doc.page_content + "\n"
        
        final_prompt = prompt.format(contexto = contexto, pergunta = pergunta)
        resposta = llm.invoke(final_prompt)
        
        st.write(f"🤖: {resposta.content}")