import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pymupdf4llm
from pathlib import Path
import re
from typing import List, Tuple
import base64
from huggingface_hub import InferenceClient
from langchain_core.documents import Document


def load_pdf_documents(folder_path):
    """
    Load all PDF documents from the specified folder and convert to markdown.
    
    Args:
        folder_path (str): Path to the folder containing PDF reports
        
    Returns:
        list: List of markdown texts (one per PDF file)
    """
    # Get all PDF files from the folder
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠ No PDF files found in {folder_path}")
        return []
    
    md_texts = []
    
    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            print(f"Processing: {pdf_file.name}")
            
            # Convert PDF to markdown with image extraction
            md_text = pymupdf4llm.to_markdown(
                doc=str(pdf_file),
                write_images=True,
                image_path="images",
                image_format="png",
                dpi=300
            )
            
            md_texts.append(md_text)
            print(f"  ✓ Converted {pdf_file.name} to markdown")
            
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {str(e)}")
            continue
    
    print(f"\n✓ Successfully processed {len(md_texts)} out of {len(pdf_files)} PDF documents")
    
    return md_texts


def find_image_references(md_text: str) -> List[Tuple[str, int, int]]:
    pattern = r'!\[\s*\]\(([^)]+)\)'
    matches = []
    
    for match in re.finditer(pattern, md_text):
        image_path = match.group(1)
        matches.append((image_path, match.start(), match.end()))
    
    return matches

def extract_char_context(
    text: str,
    start: int,
    end: int,
    window: int = 100
) -> Tuple[str, str]:
    before = text[max(0, start - window):start].strip()
    after = text[end:min(len(text), end + window)].strip()
    return before, after
    
def extract_image(md_texts):
    """
    Extract images and their surrounding context from markdown texts.
    
    Args:
        md_texts (list): List of markdown text strings
        
    Returns:
        list: List of dictionaries containing image paths and context for all PDFs
    """
    

    all_results = []
    
    # Process each markdown file one by one
    for md_text in md_texts:
        # Find all image references in current markdown
        image_refs = find_image_references(md_text)
        
        # Extract context for each image
        for img_path, start, end in image_refs:
            before, after = extract_char_context(md_text, start, end)
            
            all_results.append({
                "image_path": img_path,
                "context_before": before,
                "context_after": after
            })
    
    print('all_results_test',all_results)
    return all_results
 
def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
   
def generate_image_summary(all_results):
    """
    Generate summaries for images using LLM and create Document objects.
    
    Args:
        all_results (list): List of dictionaries containing image paths and context
        
    Returns:
        list: List of Document objects with summaries and metadata
    """
    
    # Initialize Hugging Face client
    client = InferenceClient(api_key="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")
    
    documents = []
    
    # Process each JSON one by one
    for result in all_results:
        # Extract information from JSON
        image_path = result["image_path"]
        context_before = result["context_before"]
        context_after = result["context_after"]
        
        try:
            print(f"Processing image: {image_path}")
            
            # Convert image to base64
            image_b64 = image_to_base64(image_path)
            
            # Call LLM with image and context
            completion = client.chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Analyze this image and provide a detailed interpretation based on the surrounding text context.

                                    CONTEXT BEFORE THE IMAGE:
                                    {context_before if context_before.strip() else "No preceding text available"}

                                    CONTEXT AFTER THE IMAGE:
                                    {context_after if context_after.strip() else "No following text available"}

                                    CRITICAL: Return ONLY plain text."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024
            )
            
            # Get summary from LLM
            summary = completion.choices[0].message.content
            
            # Create Document with page_content and metadata
            doc = Document(
                page_content=summary,
                metadata={"image_path": image_path}
            )
            
            documents.append(doc)
            print(f"  ✓ Generated summary for {image_path}")
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path}: {str(e)}")
            continue
    
    print(f"\n✓ Generated {len(documents)} image summaries")
    print(' ')
    print("documents_teee",documents)
    return documents
    

def chunk_documents(md_texts, chunk_size, chunk_overlap):
    """
    Convert markdown texts to documents and split into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        md_texts (list): List of markdown texts (output from load_pdf_documents)
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of text chunks
    """
 
    
    # Convert markdown texts to Document objects
    documents = []
    for md_text in md_texts:
        doc = Document(
            page_content=md_text,
            metadata={}
        )
        documents.append(doc)
    
    print(f"✓ Created {len(documents)} documents from markdown texts")
    
    # Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    print(f"✓ Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Return chunks
    return chunks

def create_vector_db_text(chunks, db_path="faiss_index"):
    """
    Create and save FAISS vector database from document chunks.
    
    Args:
        chunks (list): List of text chunks to embed
        db_path (str): Path where the FAISS database will be saved
        
    Returns:
        FAISS: The created vector database
    """
    # Initialize HuggingFaceEmbeddings with "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"✓ Embedding model loaded")
    
    # Create FAISS vector store from chunks using embeddings
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    print(f"✓ FAISS vector database created")
    
    # Save the vector database to disk
    vector_db.save_local(db_path)
    
    print(f"✓ Vector database saved to {db_path}")
    print(f"  - Index file: {db_path}/index.faiss")
    print(f"  - Pickle file: {db_path}/index.pkl")
    
    # Return the vector database
    return vector_db

def create_vector_db_image(image_documents, db_path="faiss_image"):
    """
    Create and save FAISS vector database from image summary documents.
    
    Args:
        image_documents (list): List of Document objects with image summaries
        db_path (str): Path where the FAISS database will be saved
        
    Returns:
        FAISS: The created vector database
    """
    # Initialize HuggingFaceEmbeddings with "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"✓ Embedding model loaded")
    
    # Create FAISS vector store from image documents using embeddings
    vector_db = FAISS.from_documents(image_documents, embeddings)
    
    print(f"✓ FAISS vector database created from {len(image_documents)} image summaries")
    
    # Save the vector database to disk
    vector_db.save_local(db_path)
    
    print(f"✓ Vector database saved to {db_path}")
    print(f"  - Index file: {db_path}/index.faiss")
    print(f"  - Pickle file: {db_path}/index.pkl")
    
    # Return the vector database
    return vector_db

def main(reports_folder_path, vector_db_text_path="faiss_index", vector_db_image_path="faiss_image"):
    """
    Main function to orchestrate the complete indexing pipeline.
    
    Args:
        reports_folder_path (str): Path to the folder containing laboratory reports
        vector_db_text_path (str): Path where the text FAISS database will be saved
        vector_db_image_path (str): Path where the image FAISS database will be saved
    """
    print("=" * 60)
    print("Starting Pipeline 1: Data Indexing")
    print("=" * 60)
    
    # Step 1: Load PDF documents
    print("\n[1/5] Loading PDF documents...")
    documents = load_pdf_documents(reports_folder_path)
    
    print('documents',documents)
    
    print('')
    
    # Step 2: Extract images
    print("\n[2/5] Extracting images...")
    images_info = extract_image(documents)
    
    print('images_main',images_info)
    
    print('\n')
    
    # Step 3: Generate image summaries
    print("\n[3/5] Generating image summaries...")
    image_summary = generate_image_summary(images_info)
    print('image_summary',image_summary)

    # Step 4: Chunk documents
    print("\n[4/5] Chunking documents...")
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=100)
    print('chunks',chunks)

    # Step 5: Create and save vector databases
    print("\n[5/5] Creating embeddings and vector databases...")
    
    print("  Creating text vector database...")
    vector_db_text = create_vector_db_text(chunks, vector_db_text_path)
    
    print("  Creating image vector database...")
    vector_db_image = create_vector_db_image(image_summary, vector_db_image_path)
    
    # Completion
    print("\nPipeline completed successfully!")
    print(f"Text vector database saved at: {vector_db_text_path}")
    print(f"Image vector database saved at: {vector_db_image_path}")
    print("=" * 60)
    
    
    
if __name__ == "__main__":
    # Example usage
    REPORTS_FOLDER = "Dataset"
    VECTOR_DB_TEXT_PATH = "faiss_index"
    VECTOR_DB_IMAGE_PATH = "faiss_image"
    
    # Call main function with appropriate parameters
    main(REPORTS_FOLDER, VECTOR_DB_TEXT_PATH, VECTOR_DB_IMAGE_PATH)