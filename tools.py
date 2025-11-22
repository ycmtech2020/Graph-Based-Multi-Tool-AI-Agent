# # import os
# # import io
# # import pytesseract
# # from PIL import Image
# # from typing import List
# # from langchain_core.documents import Document
# # from langchain_community.tools import WikipediaQueryRun
# # from langchain_community.utilities import WikipediaAPIWrapper
# # from langchain_community.document_loaders import PyPDFLoader
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_astradb import AstraDBVectorStore
# # from langchain_huggingface import HuggingFaceEmbeddings

# # # --- Tool Initialization (Run once on app start) ---
# # # NOTE: Tesseract requires system-level installation. Set the command path if necessary
# # # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # # Embeddings Model (using the one you chose)
# # EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # # Wikipedia Tool
# # WIKI_API_WRAPPER = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
# # WIKI_TOOL = WikipediaQueryRun(api_wrapper=WIKI_API_WRAPPER)

# # # Astra DB Vector Store (Initialized in main app)
# # ASTRA_VECTOR_STORE = None
# # COLLECTION_NAME = "pdf_rag_agent_collection"

# # def initialize_vector_store(astra_token: str, astra_db_id: str, astra_endpoint: str):
# #     """Initializes the global AstraDBVectorStore object."""
# #     global ASTRA_VECTOR_STORE
# #     # Note: cassio.init should be done in app.py
# #     ASTRA_VECTOR_STORE = AstraDBVectorStore(
# #         embedding=EMBEDDINGS,
# #         collection_name=COLLECTION_NAME,
# #         api_endpoint=astra_endpoint,
# #         token=astra_token,
# #     )
# #     print("Astra DB Vector Store Initialized.")
# #     return ASTRA_VECTOR_STORE.as_retriever()

# # # tools.py: inside initialize_vector_store()
# # # The function signature can be simplified since the credentials aren't needed.
# # # def initialize_vector_store():
# # #     """Initializes the global AstraDBVectorStore object using the globally set Cassio session."""
# # #     global ASTRA_VECTOR_STORE
    
# # #     ASTRA_VECTOR_STORE = AstraDBVectorStore(
# # #         embedding=EMBEDDINGS,
# # #         collection_name=COLLECTION_NAME,
# # #         # NOTE: Omit api_endpoint and token to use the session set by cassio.init()
# # #     )
# # #     print("Local Vector Store Initialized.")
# # #     return ASTRA_VECTOR_STORE.as_retriever()


# # def pdf_rag_tool(file_bytes: bytes) -> str:
# #     """
# #     Ingests a PDF file, chunks it, and stores the chunks in Astra DB.
# #     Returns a status message.
# #     """
# #     if ASTRA_VECTOR_STORE is None:
# #         return "Error: Astra DB Vector Store not initialized."
    
# #     # 1. Save file temporarily (LangChain Loaders need a path or file-like object)
# #     temp_pdf_path = "temp_uploaded_file.pdf"
# #     with open(temp_pdf_path, "wb") as f:
# #         f.write(file_bytes)

# #     # 2. Load and Split Documents
# #     loader = PyPDFLoader(temp_pdf_path)
# #     docs = loader.load()
    
# #     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
# #         chunk_size=1000, chunk_overlap=100
# #     )
# #     doc_splits = text_splitter.split_documents(docs)

# #     # 3. Add to Vector Store
# #     # NOTE: This will replace the existing collection content for simplicity.
# #     # For production, you might want to use a unique collection ID or a dedicated upsert logic.
# #     ASTRA_VECTOR_STORE.add_documents(doc_splits)

# #     # 4. Clean up
# #     os.remove(temp_pdf_path)
    
# #     return f"Successfully processed and indexed {len(doc_splits)} chunks from the PDF."


# # # --- Graph Node Functions ---

# # def retrieve(state):
# #     """Node for retrieving documents from the Astra DB Vector Store (PDF RAG)."""
# #     print("---RETRIEVE FROM PDF RAG---")
# #     question = state["question"]
    
# #     if ASTRA_VECTOR_STORE is None:
# #         return {"documents": [Document(page_content="Error: PDF RAG not available.")], "question": question}
    
# #     # Retrieval
# #     retriever = ASTRA_VECTOR_STORE.as_retriever()
# #     documents = retriever.invoke(question)
# #     return {"documents": documents, "question": question}


# # def wiki_search(state):
# #     """Node for performing a Wikipedia search."""
# #     print("---WIKIPEDIA SEARCH---")
# #     question = state["question"]

# #     # Wiki search
# #     result = WIKI_TOOL.invoke({"query": question})
    
# #     # Ensure the result is formatted as a list of Documents for consistency
# #     wiki_result_doc = Document(page_content=result)
    
# #     return {"documents": [wiki_result_doc], "question": question}


# # def tesseract_ocr_tool(state):
# #     """Node for performing OCR on an image (best tool for image-based text)."""
# #     print("---TESSERACT OCR TOOL---")
    
# #     # In a Streamlit app, the image should be passed via the state.
# #     # For this example, we'll assume a file path is provided in the question
# #     # or that the main app stores a temporary file and puts the path in state.
    
# #     # Simplified approach: Check for a placeholder "ocr_file_path" in the state
# #     ocr_file_path = state.get("ocr_file_path")
    
# #     if not ocr_file_path:
# #         return {"documents": [Document(page_content="Error: No image file path provided for OCR.")], "question": state["question"]}

# #     try:
# #         # 1. Open image
# #         img = Image.open(ocr_file_path)
        
# #         # 2. Perform OCR
# #         extracted_text = pytesseract.image_to_string(img)
        
# #         # 3. Format result
# #         ocr_result_doc = Document(page_content=f"OCR Result from image at {ocr_file_path}: {extracted_text}")
        
# #         return {"documents": [ocr_result_doc], "question": state["question"]}
        
# #     except Exception as e:
# #         return {"documents": [Document(page_content=f"OCR Error: {e}")], "question": state["question"]}


















# import os
# import io
# import pytesseract
# from PIL import Image
# from typing import List
# from langchain_core.documents import Document
# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# # import easyocr

# #from paddleocr import PaddleOCR
# # NEW IMPORTS for local RAG
# # from langchain_community.vectorstores import Chroma
# # tools.py
# # RECOMMENDED:
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings # Keep your original embeddings
# # Remove: from langchain_astradb import AstraDBVectorStore

# # --- Tool Initialization (Run once on app start) ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'  
# # Embeddings Model (using the one you chose)
# EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Wikipedia Tool (Unchanged)
# WIKI_API_WRAPPER = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
# WIKI_TOOL = WikipediaQueryRun(api_wrapper=WIKI_API_WRAPPER)

# # --- ChromaDB Configuration ---
# # Chroma DB Vector Store (Initialized in main app)
# CHROMA_DB_PATH = "./chroma_data"
# CHROMA_VECTOR_STORE = None
# COLLECTION_NAME = "pdf_rag_agent_collection"

# # Simplified function signature - no credentials needed!
# def initialize_vector_store(): 
#     """Initializes the global Chroma Vector Store object."""
#     global CHROMA_VECTOR_STORE
    
#     CHROMA_VECTOR_STORE = Chroma(
#         persist_directory=CHROMA_DB_PATH,
#         embedding_function=EMBEDDINGS,
#         collection_name=COLLECTION_NAME,
#     )
#     print(f"ChromaDB Vector Store Initialized locally at {CHROMA_DB_PATH}.")
#     # Return the existing store as the initial retriever for consistency
#     return CHROMA_VECTOR_STORE.as_retriever()


# def pdf_rag_tool(file_bytes: bytes) -> str:
#     """
#     Inverts a PDF file, chunks it, and stores the chunks in ChromaDB.
#     Returns a status message.
#     """
#     global CHROMA_VECTOR_STORE
    
#     if CHROMA_VECTOR_STORE is None:
#         return "Error: ChromaDB Vector Store not initialized."
    
#     # 1. Save file temporarily (LangChain Loaders need a path or file-like object)
#     temp_pdf_path = "temp_uploaded_file.pdf"
#     with open(temp_pdf_path, "wb") as f:
#         f.write(file_bytes)

#     # 2. Load and Split Documents (Unchanged)
#     loader = PyPDFLoader(temp_pdf_path)
#     docs = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=1000, chunk_overlap=100
#     )
#     doc_splits = text_splitter.split_documents(docs)

#     # 3. Add to Vector Store (Use Chroma's add_documents)
#     # NOTE: Chroma handles upserting/adding documents to the collection.
#     CHROMA_VECTOR_STORE.add_documents(doc_splits)

#     # 4. Clean up (Unchanged)
#     os.remove(temp_pdf_path)
    
#     return f"Successfully processed and indexed {len(doc_splits)} chunks from the PDF into ChromaDB."


# # --- Graph Node Functions ---

# def retrieve(state):
#     """Node for retrieving documents from the ChromaDB Vector Store (PDF RAG)."""
#     global CHROMA_VECTOR_STORE
#     print("---RETRIEVE FROM PDF RAG (CHROMA)---")
#     question = state["question"]
    
#     if CHROMA_VECTOR_STORE is None:
#         return {"documents": [Document(page_content="Error: PDF RAG (Chroma) not available.")], "question": question}
    
#     try:
#         # Retrieval logic that might be failing
#         retriever = CHROMA_VECTOR_STORE.as_retriever()
#         documents = retriever.invoke(question)
        
#         # ⚠️ Check if documents came back empty (optional, but good practice)
#         if not documents:
#             print("Warning: Retrieval returned 0 documents.")
            
#         return {"documents": documents, "question": question}
        
#     except Exception as e:
#         # 🟢 CRITICAL CHANGE: Return the actual error message
#         error_content = f"Retrieval Failed: {type(e).__name__}: {str(e)}"
#         print(error_content) # Print to terminal for debugging
#         return {"documents": [Document(page_content=error_content)], "question": question}

# # tools.py

# # ... (Definitions for WIKI_API_WRAPPER and WIKI_TOOL) ...

# def wiki_search(state):
#     """Node for performing a Wikipedia search."""
#     print("---WIKIPEDIA SEARCH---")
#     question = state["question"]

#     # Wiki search
#     result = WIKI_TOOL.invoke({"query": question})
    
#     # Ensure the result is formatted as a list of Documents for consistency
#     wiki_result_doc = Document(page_content=result, metadata={"source": "Wikipedia Search"}) # Added metadata for clarity
    
#     return {"documents": [wiki_result_doc], "question": question}

# # reader = easyocr.Reader(['en'])
# def tesseract_ocr_tool(state):
#     """Node for performing OCR on an image (best tool for image-based text)."""
#     print("---TESSERACT OCR TOOL---")
    
#     #OCR = PaddleOCR(use_angle_cls=True, lang='en')
#     # In a Streamlit app, the image should be passed via the state.
#     # For this example, we'll assume a file path is provided in the question
#     # or that the main app stores a temporary file and puts the path in state.
    
#     # Simplified approach: Check for a placeholder "ocr_file_path" in the state
#     ocr_file_path = state.get("ocr_file_path")
    
#     if not ocr_file_path:
#         return {"documents": [Document(page_content="Error: No image file path provided for OCR.")], "question": state["question"]}

#     try:
#         # 1. Open image
#         img = Image.open(ocr_file_path)
        
#         # 2. Perform OCR
#         extracted_text = pytesseract.image_to_string(img)
        
#         # 3. Format result
#         ocr_result_doc = Document(page_content=f"OCR Result from image at {ocr_file_path}: {extracted_text}")
        
#         return {"documents": [ocr_result_doc], "question": state["question"]}
        
#     except Exception as e:
#         return {"documents": [Document(page_content=f"OCR Error: {e}")], "question": state["question"]}
# # def tesseract_ocr_tool(state):
# #     print("---TESSERACT OCR TOOL---")

# #     ocr_file_path = state.get("ocr_file_path")

# #     if not ocr_file_path:
# #         return {
# #             "documents": [Document(page_content="Error: No image file path provided for OCR.")],
# #             "question": state["question"]
# #         }

# #     try:
# #         # EASY OCR extraction
# #         result = reader.readtext(ocr_file_path, detail=0)
# #         extracted_text = "\n".join(result)

# #         return {
# #             "documents": [Document(page_content=extracted_text)],
# #             "question": state["question"]
# #         }

# #     except Exception as e:
# #         return {
# #             "documents": [Document(page_content=f"OCR Error: {e}")],
# #             "question": state["question"]
# #         }
   



import os
import io
import pytesseract
from PIL import Image,ImageOps
from typing import List
from langchain_core.documents import Document
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader
import streamlit as st
from tempfile import NamedTemporaryFile
# NEW IMPORTS for local RAG
# from langchain_community.vectorstores import Chroma
# tools.py
# RECOMMENDED:
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Keep your original embeddings
# Remove: from langchain_astradb import AstraDBVectorStore

# --- Tool Initialization (Run once on app start) ---
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'  
# Embeddings Model (using the one you chose)
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Wikipedia Tool (Unchanged)
WIKI_API_WRAPPER = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
WIKI_TOOL = WikipediaQueryRun(api_wrapper=WIKI_API_WRAPPER)

# --- ChromaDB Configuration ---
# Chroma DB Vector Store (Initialized in main app)
CHROMA_DB_PATH = "./chroma_data"
CHROMA_VECTOR_STORE = None
COLLECTION_NAME = "multifile_rag_agent_collection"

# Simplified function signature - no credentials needed!
def initialize_vector_store(): 
    """Initializes the global Chroma Vector Store object."""
    global CHROMA_VECTOR_STORE
    
    CHROMA_VECTOR_STORE = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=EMBEDDINGS,
        collection_name=COLLECTION_NAME,
    )
    print(f"ChromaDB Vector Store Initialized locally at {CHROMA_DB_PATH}.")
    # Return the existing store as the initial retriever for consistency
    return CHROMA_VECTOR_STORE.as_retriever()

def load_csv(file_path: str):
    loader = CSVLoader(file_path)
    return loader.load()

def load_docx(file_path: str):
    loader = Docx2txtLoader(file_path)
    return loader.load()

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

def multi_file_rag_tool(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    global CHROMA_VECTOR_STORE
    
    if CHROMA_VECTOR_STORE is None:
        initialize_vector_store()

    total_chunks = 0

    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Save temporarily
        with NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Load documents per type
        if ext == ".pdf":
            docs = load_pdf(temp_path)
        elif ext == ".csv":
            docs = load_csv(temp_path)
        elif ext == ".docx":
            docs = load_docx(temp_path)
        else:
            # Skip unsupported file types
            os.remove(temp_path)
            continue


        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=100
        )
        doc_splits = text_splitter.split_documents(docs)

        # Add to vectorstore
        CHROMA_VECTOR_STORE.add_documents(doc_splits)
        total_chunks += len(doc_splits)

        os.remove(temp_path)

    return f"Successfully processed and indexed {total_chunks} chunks from the uploaded files into ChromaDB."


# def pdf_rag_tool(file_bytes: bytes) -> str:
#     """
#     Inverts a PDF file, chunks it, and stores the chunks in ChromaDB.
#     Returns a status message.
#     """
#     global CHROMA_VECTOR_STORE
    
#     if CHROMA_VECTOR_STORE is None:
#         return "Error: ChromaDB Vector Store not initialized."
    
#     # 1. Save file temporarily (LangChain Loaders need a path or file-like object)
#     temp_pdf_path = "temp_uploaded_file.pdf"
#     with open(temp_pdf_path, "wb") as f:
#         f.write(file_bytes)

#     # 2. Load and Split Documents (Unchanged)
#     loader = PyPDFLoader(temp_pdf_path)
#     docs = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=1000, chunk_overlap=100
#     )
#     doc_splits = text_splitter.split_documents(docs)

#     # 3. Add to Vector Store (Use Chroma's add_documents)
#     # NOTE: Chroma handles upserting/adding documents to the collection.
#     CHROMA_VECTOR_STORE.add_documents(doc_splits)

#     # 4. Clean up (Unchanged)
#     os.remove(temp_pdf_path)
    
#     return f"Successfully processed and indexed {len(doc_splits)} chunks from the PDF into ChromaDB."


# --- Graph Node Functions ---

#befor wilipedia remove

# def retrieve(state):
#     """Node for retrieving documents from the ChromaDB Vector Store (PDF RAG)."""
#     global CHROMA_VECTOR_STORE
#     print("---RETRIEVE FROM MULTI-FILE RAG (CHROMA)---")
#     question = state["question"]
    
#     if CHROMA_VECTOR_STORE is None:
#         return {"documents": [Document(page_content="Error: Multi-file RAG (Chroma) not available.")], "question": question}
    
#     try:
#         # Retrieval logic that might be failing
#         retriever = CHROMA_VECTOR_STORE.as_retriever()
#         documents = retriever.invoke(question)
        
#         # ⚠️ Check if documents came back empty (optional, but good practice)
#         if not documents:
#             print("Warning: Retrieval returned 0 documents.")
            
#         return {"documents": documents, "question": question}
        
#     except Exception as e:
#         # 🟢 CRITICAL CHANGE: Return the actual error message
#         error_content = f"Retrieval Failed: {type(e).__name__}: {str(e)}"
#         print(error_content) # Print to terminal for debugging
#         return {"documents": [Document(page_content=error_content)], "question": question}

# # tools.py

# # ... (Definitions for WIKI_API_WRAPPER and WIKI_TOOL) ...

# def wiki_search(state):
#     """Node for performing a Wikipedia search."""
#     print("---WIKIPEDIA SEARCH---")
#     question = state["question"]

#     # Wiki search
#     result = WIKI_TOOL.invoke({"query": question})
    
#     # Ensure the result is formatted as a list of Documents for consistency
#     wiki_result_doc = Document(page_content=result, metadata={"source": "Wikipedia Search"}) # Added metadata for clarity
    
#     return {"documents": [wiki_result_doc], "question": question}

# def tesseract_ocr_tool(state):
#     """Node for performing OCR on an image (best tool for image-based text)."""
#     print("---TESSERACT OCR TOOL---")
    
#     # In a Streamlit app, the image should be passed via the state.
#     # For this example, we'll assume a file path is provided in the question
#     # or that the main app stores a temporary file and puts the path in state.
    
#     # Simplified approach: Check for a placeholder "ocr_file_path" in the state
#     ocr_file_path = state.get("ocr_file_path")
    
#     if not ocr_file_path:
#         return {"documents": [Document(page_content="Error: No image file path provided for OCR.")], "question": state["question"]}

#     try:
#         # 1. Open image
#         img = Image.open(ocr_file_path)
#         # 🟢 Contrast Enhancement & Grayscale (Recommended for receipts)
#         img = img.convert('L') # Grayscale
#         # Simple binarization (optional, but helps remove background noise)
#         img = ImageOps.autocontrast(img)
#         threshold = 160 # A higher threshold works better for bills
#         img = img.point(lambda x: 0 if x < threshold else 255, '1').convert('L')
#         # 2. Perform OCR
#         custom_config = r'--oem 3 --psm 6' 
#         extracted_text = pytesseract.image_to_string(img, config=custom_config)
#         if not extracted_text.strip():
#             extracted_text = "[Warning] OCR did not detect any readable text. Try improving image quality."
#         # 3. Format result
#         ocr_result_doc = Document(
#         page_content=extracted_text, 
#         # 🟢 ADD METADATA FOR TRACEABILITY
#         metadata={"source": "Tesseract OCR"} 
#     )
        
#         return {"documents": [ocr_result_doc], "question": state["question"]}
        
#     except Exception as e:
#         return {"documents": [Document(page_content=f"OCR Error: {e}")], "question": state["question"]}



#Remove wikipedia
import os
import io
import pytesseract
from PIL import Image,ImageOps
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Keep your original embeddings
# Remove: from langchain_astradb import AstraDBVectorStore

# --- Tool Initialization (Run once on app start) ---
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'  
# Embeddings Model (using the one you chose)
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- ChromaDB Configuration ---
# Chroma DB Vector Store (Initialized in main app)
CHROMA_DB_PATH = "./chroma_data"
CHROMA_VECTOR_STORE = None
COLLECTION_NAME = "multifile_rag_agent_collection"

# Simplified function signature - no credentials needed!
def initialize_vector_store(): 
    """Initializes the global Chroma Vector Store object."""
    global CHROMA_VECTOR_STORE
    
    CHROMA_VECTOR_STORE = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=EMBEDDINGS,
        collection_name=COLLECTION_NAME,
    )
    print(f"ChromaDB Vector Store Initialized locally at {CHROMA_DB_PATH}.")
    # Return the existing store as the initial retriever for consistency
    return CHROMA_VECTOR_STORE.as_retriever()

def load_csv(file_path: str):
    loader = CSVLoader(file_path)
    return loader.load()

def load_docx(file_path: str):
    loader = Docx2txtLoader(file_path)
    return loader.load()

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

def multi_file_rag_tool(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    global CHROMA_VECTOR_STORE
    
    if CHROMA_VECTOR_STORE is None:
        initialize_vector_store()

    total_chunks = 0

    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Save temporarily
        with NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        # Load documents per type
        if ext == ".pdf":
            docs = load_pdf(temp_path)
        elif ext == ".csv":
            docs = load_csv(temp_path)
        elif ext == ".docx":
            docs = load_docx(temp_path)
        else:
            # Skip unsupported file types
            os.remove(temp_path)
            continue


        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=100
        )
        doc_splits = text_splitter.split_documents(docs)

        # Add to vectorstore
        CHROMA_VECTOR_STORE.add_documents(doc_splits)
        total_chunks += len(doc_splits)

        os.remove(temp_path)

    return f"Successfully processed and indexed {total_chunks} chunks from the uploaded files into ChromaDB."


def retrieve(state):
    """Node for retrieving documents from the ChromaDB Vector Store (PDF RAG)."""
    global CHROMA_VECTOR_STORE
    print("---RETRIEVE FROM MULTI-FILE RAG (CHROMA)---")
    question = state["question"]
    
    if CHROMA_VECTOR_STORE is None:
        return {"documents": [Document(page_content="Error: Multi-file RAG (Chroma) not available.")], "question": question}
    
    try:
        # Retrieval logic that might be failing
        retriever = CHROMA_VECTOR_STORE.as_retriever()
        documents = retriever.invoke(question)
        
        # ⚠️ Check if documents came back empty (optional, but good practice)
        if not documents:
            print("Warning: Retrieval returned 0 documents.")
            
        return {"documents": documents, "question": question}
        
    except Exception as e:
        # 🟢 CRITICAL CHANGE: Return the actual error message
        error_content = f"Retrieval Failed: {type(e).__name__}: {str(e)}"
        print(error_content) # Print to terminal for debugging
        return {"documents": [Document(page_content=error_content)], "question": question}

# tools.py

# ... (Definitions for WIKI_API_WRAPPER and WIKI_TOOL) ...

def tesseract_ocr_tool(state):
    """Node for performing OCR on an image (best tool for image-based text)."""
    print("---TESSERACT OCR TOOL---")
    
    # In a Streamlit app, the image should be passed via the state.
    # For this example, we'll assume a file path is provided in the question
    # or that the main app stores a temporary file and puts the path in state.
    
    # Simplified approach: Check for a placeholder "ocr_file_path" in the state
    ocr_file_path = state.get("ocr_file_path")
    
    if not ocr_file_path:
        return {"documents": [Document(page_content="Error: No image file path provided for OCR.")], "question": state["question"]}

    try:
        # 1. Open image
        img = Image.open(ocr_file_path)
        # 🟢 Contrast Enhancement & Grayscale (Recommended for receipts)
        img = img.convert('L') # Grayscale
        # Simple binarization (optional, but helps remove background noise)
        img = ImageOps.autocontrast(img)
        threshold = 160 # A higher threshold works better for bills
        img = img.point(lambda x: 0 if x < threshold else 255, '1').convert('L')
        # 2. Perform OCR
        custom_config = r'--oem 3 --psm 6' 
        extracted_text = pytesseract.image_to_string(img, config=custom_config)
        if not extracted_text.strip():
            extracted_text = "[Warning] OCR did not detect any readable text. Try improving image quality."
        # 3. Format result
        ocr_result_doc = Document(
        page_content=extracted_text, 
        # 🟢 ADD METADATA FOR TRACEABILITY
        metadata={"source": "Tesseract OCR"} 
    )
        
        return {"documents": [ocr_result_doc], "question": state["question"]}
        
    except Exception as e:
        return {"documents": [Document(page_content=f"OCR Error: {e}")], "question": state["question"]}








