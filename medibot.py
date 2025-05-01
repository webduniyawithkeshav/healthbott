import os
import shutil
import tempfile
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def process_uploaded_files(uploaded_files):
    all_texts = []
    temp_dir = tempfile.mkdtemp()
    file_summary = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_documents(documents)
        all_texts.extend(texts)

        # Store summary of the document (first 100 chars of the first text chunk)
        file_summary.append({"file": uploaded_file.name, "summary": texts[0].page_content[:100]})

    return all_texts, file_summary, temp_dir

def build_vectorstore(text_chunks, embedding_model):
    return FAISS.from_documents(text_chunks, embedding_model)

def main():
    st.title("Ask Healthbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    st.markdown("---")
    st.subheader("💬 Chat + 📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload your PDF files here to include in the chat (optional)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    use_uploaded_docs = st.checkbox("Use uploaded documents only", value=False)
    clear_chat_history = st.button("Clear Chat History")
    clear_state = st.button("Clear Uploaded Docs")

    if clear_chat_history:
        st.session_state.messages = []
        st.success("Chat history cleared.")

    if clear_state:
        st.session_state.pop('temp_db', None)
        st.success("Uploaded documents cleared.")

    prompt = st.chat_input("Ask your medical question here...")

    if uploaded_files:
        # Display the uploaded documents' summary
        _, file_summary, _ = process_uploaded_files(uploaded_files)
        st.subheader("Uploaded Documents Summary")
        for file_info in file_summary:
            st.write(f"**{file_info['file']}**: {file_info['summary']}...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
        Don't provide anything outside the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

            if use_uploaded_docs and uploaded_files:
                docs, file_summary, tmp_dir = process_uploaded_files(uploaded_files)
                vectorstore = build_vectorstore(docs, embedding_model)
                st.session_state['temp_db'] = vectorstore
                shutil.rmtree(tmp_dir)
            elif use_uploaded_docs and 'temp_db' in st.session_state:
                vectorstore = st.session_state['temp_db']
            else:
                vectorstore = get_vectorstore()

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGING_FACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            result_to_show = result + "\n\nSource Documents:\n"
            for doc in source_documents:
                result_to_show += f"**{doc.metadata['source']}**\n"

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
