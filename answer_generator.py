# answer_generator.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


def load_vector_db(db_path):
    """
    Load the saved FAISS vector database from disk.
    
    Args:
        db_path (str): Path to the saved FAISS database
        
    Returns:
        FAISS: Loaded vector database
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vector_db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print(f"✓ Vector database loaded from {db_path}")
    print(f"  - Total vectors: {vector_db.index.ntotal}")
    
    return vector_db

####text:

def retrieve_context_text(vector_db_t, question, top_k=10):
    """
    Retrieve relevant document chunks using similarity search.
    
    Args:
        vector_db (FAISS): Loaded FAISS database
        question (str): User's question to search for relevant context
        top_k (int): Number of top chunks to retrieve
        
    Returns:
        list: List of retrieved document chunks
    """
    retrieved_chunks = vector_db_t.similarity_search(question, k=top_k)
    
    print(f"✓ Retrieved {len(retrieved_chunks)} relevant chunks")
    
    return retrieved_chunks

def format_context(docs):
    """
    Formats retrieved chunks with metadata for readability.
    
    Args:
        docs (list): List of retrieved document chunks
        
    Returns:
        str: Formatted context string
    """
    formatted = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", "N/A")
        formatted.append(f"[Source: {source} | Page {page}]\n{d.page_content.strip()}")
    return "\n\n".join(formatted)


######## image :

def retrieve_context_image(vector_db_i, question, top_k=10):
    retrieved_chunks = vector_db_i.similarity_search(question, k=top_k)
    
    print(f"✓ Retrieved {len(retrieved_chunks)} relevant chunks")
    
    return retrieved_chunks


def image_format(retrieved_chunks):
    """
    Format retrieved image chunks with serial numbers.
    
    Args:
        retrieved_chunks (list): List of Document objects from retrieve_context_image
        
    Returns:
        tuple: (summary_doc, summary_format)
            - summary_doc: Dictionary mapping serial number to full Document object
            - summary_format: Beautified string with serial number and page_content
    """
    summary_doc = {}
    summary_entries = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        # Add beautified entry with serial number and page content
        summary_entries.append(f"[{i}]: {chunk.page_content}")
        
        # Map serial number to full document object
        summary_doc[i] = chunk
    
    # Join all entries with newlines
    summary_format = "\n".join(summary_entries)
    
    print(f"✓ Formatted {len(retrieved_chunks)} image summaries")
    
    return summary_doc, summary_format



#### llm

import json

def generate_answer(question, retrieved_chunks, summary_format, conversation_history):
    """
    Generate final answer using LLM with retrieved text context, image context, and conversation history.
    
    Args:
        question (str): User question
        retrieved_chunks (list): Retrieved text chunks from vector DB
        summary_format (str): Formatted image summaries with serial numbers
        conversation_history (list): List of message dicts that will be updated
        
    Returns:
        dict: JSON response with text_response and image_relevant list
    """
    # Check if we have retrieved chunks
    if not retrieved_chunks:
        return {
            "text_response": "I'm sorry, but the provided document does not contain enough information to answer that.",
            "image_relevant": []
        }
    
    # Format text context from retrieved chunks
    text_context = format_context(retrieved_chunks)

    # print('text_context')   # qc step
    # print(text_context)
    
    # Updated system prompt to handle both text and image context, and return JSON format
    SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions based ONLY on the provided context.

Rules:
- Answer the question using the text context and image context provided
- If images are relevant to the question, include their numbers in the response
- Return response as JSON with this exact format:
{
  "text_response": "your answer here",
  "image_relevant": [list of image numbers if relevant, empty list if none]
}
- If the answer is NOT in the context, set text_response to "I cannot answer this based on the provided documents"
- Do not make up information
- Be direct and factual"""

    # Current user message includes both text context and image context
    current_user_message = f"""
📘 **Text Context:**
{text_context}

🖼️ **Image Context:**
{summary_format}
---------------------
❓ **Question:**
{question}
---------------------
🧩 **Answer (Return ONLY valid JSON):**
""".strip()
    
    print('current_user_message:', current_user_message)
    print('\n')
    
    # Initialize LLM
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set!")
    
    # Build messages for LLM - start with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    print('messages (initial):', messages)
    print('\n')
    
    # Add conversation history (previous Q&A pairs)
    messages.extend(conversation_history)
    
    # Add current question with both text and image context
    messages.append({"role": "user", "content": current_user_message})
    
    print('messages (with history + current):', messages)
    print('\n')
    
    # Initialize LLM with JSON response format enforced
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key=api_key,
        temperature=0.7,
        max_tokens=1024,
        model_kwargs={"response_format": {"type": "json_object"}}  # Force JSON output
    )
    
    print('messages_to_llm:', messages)
    print('\n')
    
    # Get response from LLM
    response = llm.invoke(messages)
    print('llm_response:', response)
    print('\n')
    
    # Extract clean text output from LLM response
    if hasattr(response, "content"):
        answer_text = response.content.strip()
    elif isinstance(response, dict) and "content" in response:
        answer_text = response["content"].strip()
    else:
        answer_text = str(response).strip()
    
    # Parse the JSON response from LLM
    try:
        answer_json = json.loads(answer_text)
    except json.JSONDecodeError:
        # Fallback if LLM doesn't return valid JSON
        answer_json = {
            "text_response": answer_text,
            "image_relevant": []
        }
    
    print('conversation_history (before update):', conversation_history)
    print('\n')
    
    # Store the user question in conversation history (without the context to save tokens)
    conversation_history.append({"role": "user", "content": question})
    print('conversation_history (after user):', conversation_history)
    print('\n')
    
    # Store the full JSON response in conversation history for context continuity
    conversation_history.append({"role": "assistant", "content": json.dumps(answer_json)})
    print('conversation_history (after assistant - FINAL):', conversation_history)
    print('\n')
    
    print("✓ Answer generated successfully")
    print('answer_json_returned:', answer_json)
    print('\n')
    
    # Return the full JSON response with text_response and image_relevant fields
    return answer_json


def main(user_question, vector_db_path_text="faiss_index", vector_db_path_image="faiss_image", conversation_history=None):
    """
    Main function to orchestrate the complete retrieval + answer pipeline.
    
    Args:
        user_question (str): Question asked by the user
        vector_db_path_text (str): Path to the saved text FAISS vector database
        vector_db_path_image (str): Path to the saved image FAISS vector database
        conversation_history (list): Conversation history list (will be created if None)
        
    Returns:
        tuple: (answer_json, conversation_history, summary_doc)
            - answer_json: dict with text_response and image_relevant
            - conversation_history: updated conversation history list
            - summary_doc: dictionary mapping image numbers to Document objects
    """
    # Initialize conversation history if not provided
    if conversation_history is None:
        conversation_history = []
    
    print("=" * 60)
    print("Starting Pipeline 2: Retrieval and Answer Generation")
    print("=" * 60)
    
    # Step 1: Load vector database
    print("\n[1/3] Loading vector database...")
    vector_db_text = load_vector_db(vector_db_path_text)
    vector_db_image = load_vector_db(vector_db_path_image)
    
    
    # Step 2:
    # Retrieve context from text 
    print("\n[2/3] Retrieving relevant chunks...")
    retrieved_chunks_text = retrieve_context_text(vector_db_text, user_question, top_k=10)
    
    print('retrieved_chunks_text')
    print(retrieved_chunks_text)
    
    # Retrieve context from image
    print("\n[2/3] Retrieving relevant chunks...")
    retrieved_summary_img = retrieve_context_image(vector_db_image, user_question, top_k=10)
    
    # Step 3: format image . 
    summary_doc, summary_format = image_format(retrieved_summary_img)

    print('-----------------------')
    print('summary_doc')
    print(summary_doc)
    print('-----------------------')
    
    print('summary_format')
    print(summary_format)
    
    
    print("\n[3/3] Generating answer using LLM...")
    answer_json = generate_answer(user_question, retrieved_chunks_text, summary_format, conversation_history)
    
    print("\nPipeline 2 completed successfully!")
    print("=" * 60)
    
    # Return answer JSON, updated conversation history, AND summary_doc for image retrieval
    return answer_json, conversation_history, summary_doc

