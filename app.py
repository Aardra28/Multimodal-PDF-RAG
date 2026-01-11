# app.py
"""
Streamlit app that calls answer_generator.main() with conversation history support.

The app:
 - Takes user input
 - Maintains conversation history across messages
 - Passes history to answer_generator.main()
 - Displays the returned answer
 - Displays relevant images based on LLM response
"""

import streamlit as st
from pathlib import Path
from answer_generator import main as generate_answer_main


def init_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Display history: {role: 'user'|'assistant', text: str}
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []  # LLM history: {role: 'user'|'assistant', content: str}

    if "vector_db_path_text" not in st.session_state:
        st.session_state.vector_db_path_text = "faiss_index"
    
    if "vector_db_path_image" not in st.session_state:
        st.session_state.vector_db_path_image = "faiss_image"


def render_messages():
    """Display all past messages."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])
            
            # Display images if present in the message
            if "images" in msg and msg["images"]:
                st.markdown("**📷 Relevant Images:**")
                cols = st.columns(len(msg["images"]))
                for idx, img_path in enumerate(msg["images"]):
                    with cols[idx]:
                        try:
                            st.image(img_path, caption=f"Image {idx+1}", width=200)  # Changed here
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                            
def sidebar_controls():
    """Render sidebar with settings and controls."""
    st.sidebar.header("⚙️ Settings")

    st.session_state.vector_db_path_text = st.sidebar.text_input(
        "Text Vector DB Path",
        value=st.session_state.vector_db_path_text
    )
    
    st.session_state.vector_db_path_image = st.sidebar.text_input(
        "Image Vector DB Path",
        value=st.session_state.vector_db_path_image
    )

    # Check if DBs exist
    text_db_exists = Path(st.session_state.vector_db_path_text).exists()
    image_db_exists = Path(st.session_state.vector_db_path_image).exists()
    
    if not text_db_exists:
        st.sidebar.error("❌ Text Vector DB not found! Run main.py first.")
    else:
        st.sidebar.success("✅ Text Vector DB is available.")
    
    if not image_db_exists:
        st.sidebar.error("❌ Image Vector DB not found! Run main.py first.")
    else:
        st.sidebar.success("✅ Image Vector DB is available.")
    
    st.sidebar.markdown("---")
    
    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()
    
    # Show conversation stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Conversation Stats")
    st.sidebar.metric("Messages", len(st.session_state.messages))
    st.sidebar.metric("History Entries", len(st.session_state.conversation_history))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 About")
    st.sidebar.markdown("""
    This RAG system:
    - Loads PDFs from a folder
    - Extracts text and images
    - Chunks and embeds content
    - Stores in FAISS vector DBs
    - Retrieves relevant chunks and images
    - Generates answers using Groq LLM
    - Displays relevant images
    - Maintains conversation context
    """)


def main():
    st.set_page_config(
        page_title="PDF RAG Q&A", 
        page_icon="📄",
        layout="wide"
    )
    
    init_state()
    sidebar_controls()

    st.title("📄 PDF RAG Q&A System with Image Support")
    st.markdown("Ask questions about your PDF documents — powered by FAISS retrieval and Groq LLM (LLaMA 4 Scout 17B)")
    st.markdown("*Conversation history is maintained for follow-up questions. Relevant images will be displayed automatically.*")

    # Show history
    render_messages()

    # Chat input
    user_question = st.chat_input("Ask your question...")
    if user_question:

        # Save user message for display
        st.session_state.messages.append({"role": "user", "text": user_question})

        # Display user message immediately
        with st.chat_message("user"):
            st.write(user_question)

        # Assistant message block
        with st.chat_message("assistant"):

            with st.status("Processing your question..."):
                try:
                    # Call answer_generator.main() with conversation history
                    answer_json, updated_history, summary_doc = generate_answer_main(
                        user_question=user_question,
                        vector_db_path_text=st.session_state.vector_db_path_text,
                        vector_db_path_image=st.session_state.vector_db_path_image,
                        conversation_history=st.session_state.conversation_history
                    )
                    
                    # Update conversation history in session state
                    st.session_state.conversation_history = updated_history
                    
                    # Extract text response and image relevant list from JSON
                    text_response = answer_json.get("text_response", "")
                    image_relevant_list = answer_json.get("image_relevant", [])
                    
                    print('image_relevant_list:', image_relevant_list)
                    
                    # Extract image paths from summary_doc using the relevant image numbers
                    final_image_paths = []
                    for img_num in image_relevant_list:
                        if img_num in summary_doc:
                            image_path = summary_doc[img_num].metadata.get('image_path', '')
                            if image_path:
                                final_image_paths.append(image_path)
                                print(f'Image {img_num}: {image_path}')
                    
                except Exception as e:
                    text_response = f"❌ Error: {str(e)}"
                    final_image_paths = []
                    import traceback
                    st.error(traceback.format_exc())

            # Display the text answer
            st.write(text_response)
            
            # Display relevant images if any
            if final_image_paths:
                st.markdown("**📷 Relevant Images:**")
                cols = st.columns(len(final_image_paths))
                for idx, img_path in enumerate(final_image_paths):
                    with cols[idx]:
                        try:
                            st.image(img_path, caption=f"Image {idx+1}", width=200)  # Changed here
                        except Exception as e:
                            st.error(f"Error loading image {img_path}: {e}")
                            
                            
            # Save assistant message for display (with images)
            st.session_state.messages.append({
                "role": "assistant", 
                "text": text_response,
                "images": final_image_paths
            })


if __name__ == "__main__":
    main()
    
    