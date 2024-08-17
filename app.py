from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

load_dotenv()

def main():
    st.title("Q/A Bot")

    # Step 1: Ask user to upload their PDF file
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = f"{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Create a database directory to store chroma db embeddings
        db_dir = "embedding_dir"

        # Check if embeddings already exist
        if not os.path.exists(db_dir):
            # Step 2: Open a modal to show progress of embeddings creation
            with st.spinner("Creating embeddings, please wait..."):
                loader = UnstructuredFileLoader(temp_file_path)
                documents = loader.load()

                # Split text to fix exceed embedding length issue
                text_splitter = CharacterTextSplitter(
                    chunk_size=2000, 
                    chunk_overlap=400
                )

                # Split text docs
                texts = text_splitter.split_documents(documents)

                # Create and persist embeddings
                embeddings = HuggingFaceEmbeddings()
                vector_db = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    persist_directory=db_dir
                )
            st.success("Embeddings created successfully!")

        # If embeddings already exist, load them
        else:
            st.write("Loading existing embeddings...")
            vector_db = Chroma(persist_directory=db_dir, embedding_function=HuggingFaceEmbeddings())

        # Step 3: Remove the data feed part and allow the user to ask questions
        st.write("You can now ask questions based on the uploaded document.")
        query = st.text_input("Enter your query:")

        if query:
            # Add llm
            llm = ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=0
            )

            # Add retriever to find most common vectors using vector similarity search
            retriever = vector_db.as_retriever()

            # Create a retrieval q/a chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )

            # Invoke llm with query
            response = qa_chain.invoke({"query": query})

            # Display response
            st.write("**Answer:**")
            st.write(response["result"])
            st.write("---")
            st.write(f'Source: {response["source_documents"][0].metadata["source"]}')

if __name__ == "__main__":
    main()