import streamlit as st

def iniciar_chat(prompt, llm, retriever):
    # Define o título da aplicação na interface do Streamlit.
    st.title("🤖 Chat com I.A")
      
    # Exibe um subtítulo explicando a funcionalidade do chat.
    st.subheader("Interaja com uma IA que responde perguntas sobre o universo de Dragon Ball Z.")  

    # Cria um campo de entrada de texto para o usuário digitar a pergunta.
    pergunta = st.text_input("Digite aqui: ")  
    
    if pergunta:  
        # Usa o objeto 'retriever' para buscar documentos relevantes relacionados à pergunta.
        result_docs = retriever.invoke(pergunta)  

        # Inicializa uma string vazia para armazenar o contexto da resposta.
        contexto = ""
        for doc in result_docs:  
            # Concatena o conteúdo dos documentos recuperados para formar o contexto.
            contexto += doc.page_content + "\n"  

        # Formata o prompt final, inserindo o contexto e a pergunta do usuário.
        final_prompt = prompt.format(contexto = contexto, pergunta = pergunta)  

        # Usa o modelo de linguagem (LLM) para gerar uma resposta com base no prompt formatado.
        resposta = llm.invoke(final_prompt)  

        # Exibe a resposta da IA na interface do Streamlit.
        st.write(f"🤖: {resposta.content}")  
