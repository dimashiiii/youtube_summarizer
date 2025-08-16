from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain import hub

from langchain_chroma import Chroma

import streamlit as st

from dotenv import load_dotenv
import os
load_dotenv()


import asyncio

# Create and set a new event loop for the current thread to avoid the RuntimeError
asyncio.set_event_loop(asyncio.new_event_loop())


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# # gemini 

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# # chromadb

vectorstore = Chroma(
    embedding_function=embedding_model,
    collection_name="youtube_db",
    persist_directory="./ys_chroma_db"
    )

# Youtube transcriber
ytt_api = YouTubeTranscriptApi()

#prompt for retrieval rag 
prompt = hub.pull("rlm/rag-prompt")


def retrieve(query: str):
    retrieved_docs = vectorstore.similarity_search(query)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    messages = prompt.invoke({"question": query, "context": docs_content})
    response = llm.invoke(messages)
    return response.content


def add_transcript_to_vectorstore(docs, video_id):  

    existing_docs = vectorstore.get(
        where={"video_id": video_id},
        include=['metadatas']
    )

    if existing_docs['ids']:
        print(f"Video {video_id} already exists in the collection.")
        return


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # chunk size (characters)
        chunk_overlap=100,  # chunk overlap (characters)
    )

    all_splits = text_splitter.split_documents([docs])

    vectorstore.add_documents(all_splits)

    print(f"Added transcript for video {video_id} to the collection.")


def extract_video_id(url: str) -> str:
    try:
        return url.split("v=")[1].split("&")[0]
    except IndexError:
        raise ValueError("Invalid YouTube URL format. Expected something like: https://www.youtube.com/watch?v=VIDEO_ID")


def main():

    st.set_page_config(page_title="YouTube Lesson Summarizer", layout="wide")
    st.title("📚 YouTube Lesson Summarizer")

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    if st.button("Summarize & Index"):
        if not youtube_url.strip():
            st.error("Please paste a YouTube URL")
        else:
            yt_video_id = extract_video_id(youtube_url)
            fetched_transcript = ytt_api.fetch(yt_video_id)
            formatted_transcript = TextFormatter().format_transcript(fetched_transcript)
            docs = Document(
                page_content=formatted_transcript,
                metadata={
                    "video_id": yt_video_id
                }
            )

            add_transcript_to_vectorstore(docs, yt_video_id)
            st.write("✅ Video uploaded into vectorstore")


    st.markdown("---")
    st.subheader("Ask a question about the uploaded videos")

    # Simple input + button
    user_query = st.text_input("Enter your query here:", placeholder="How many grams of sugar?", key="user_query")

    if st.button("Submit Query"):
        if not user_query or not user_query.strip():
            st.warning("Please type a query first.")
        else:
            # Show the query in the app
            st.write("### answer: ")
            response = retrieve(user_query)
            st.write(response)

            # Also print to terminal for debugging
            print("User query submitted:", response)

if __name__ == "__main__":
    main()
