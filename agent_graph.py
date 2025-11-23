import os
from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# New import for local memory (Correct for your langchain-core 1.0.7 version)
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver # New import hint
from tools import retrieve,tesseract_ocr_tool

# --- Graph State Definition ---

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[Document]
    ocr_file_path: str 
    session_id: str


# ===============================================
# --- ROUTING LOGIC (ADDED FROM COMMENTED BLOCK) ---
# ===============================================

class RouteQuery(BaseModel):
    """Route a user query to the most relevant tool/datasource."""
    datasource: Literal["multi_file_rag","ocr_tool", "rag_chat"]  = Field(
        ...,
        description=(
            "Given a user question, choose to route it to:\n"
            "1. 'multi_file_rag': For questions about uploaded PDF, CSV, Word content.\n"
            "2. 'ocr_tool': For extracting text from images (OCR).\n"
            "3. 'rag_chat': For follow-up conversational query."
        )
    )

def create_router(llm: ChatGroq):
    """Creates LLM-powered router chain supporting multi-file RAG."""
    
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system = """You are an expert router. Route the user question to the most relevant tool:
    - Use 'multi_file_rag' for questions specifically about uploaded document content (PDF, CSV, Word).
    - Use 'ocr_tool' if the user explicitly asks to 'read', 'extract', or 'transcribe' an image.
    - Use 'rag_chat' if the question is a follow-up conversational query.

    Only route to one datasource per question.
    """
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    return route_prompt | structured_llm_router

def route_question(state):
    """Route question to one of the tool nodes or the final generation node."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    
    # Initialize LLM for the router here
    groq_api_key = os.environ.get("GROQ_API_KEY")
    llm_router = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0)
    
    # Re-create router with the LLM
    structured_llm_router = llm_router.with_structured_output(RouteQuery)
    
    # Re-create the prompt for the router (simplified, since it's defined above)
    system =  system = """You are an expert router. Route the user question to the most relevant tool:
    - Use 'multi_file_rag' for questions specifically about uploaded document content (PDF, CSV, Word).
    - Use 'ocr_tool' if the user explicitly asks to 'read', 'extract text from', or 'transcribe' an image.
    - Use 'rag_chat' if the question is a follow-up to the current conversation (history).

    Only route to one datasource per question.
    """
    route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}"),])
    question_router = route_prompt | structured_llm_router
    
    source = question_router.invoke({"question": question})
    
    if source.datasource == "multi_file_rag":
        print("---ROUTE TO MULTI-FILE RAG RETRIEVAL---")
        return "retrieve"
    elif source.datasource == "ocr_tool":
        print("---ROUTE TO TESSERACT OCR TOOL---")
        return "ocr_tool"
    else: # Default or 'rag_chat' goes to generation
        print("---ROUTE TO GENERATION (CHATTING)---")
        return "generate" 


# ===============================================
# --- GENERATION NODE (FROM CURRENT CODE) ---
# ===============================================

# Global in-memory storage for chat history 
history_store = {} 

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves the correct chat history object for the session ID."""
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]


def generate(state):
    """
    Generate an answer using the retrieved documents and chat history.
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    session_id = state["session_id"]

    # if not documents or all(len(doc.page_content.strip()) == 0 for doc in documents):
    #     return {
    #         "generation": "Sorry, I can only answer questions based on the uploaded documents.",
    #         "question": question,
    #         "documents": documents
    #     }
    
    # Initialize LLM with a safe default model
    groq_api_key = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.2)
    
    # Local In-Memory History
    history = get_session_history(session_id)
    
    # Short-term memory (just the last few turns from history)
    short_term_history = history.messages[-5:]
    
    # Context
    context = "\n\n".join([doc.page_content for doc in documents])

    # Convert chat history for prompt
    chat_history = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in short_term_history])
    # 🟢 DYNAMIC PROMPT CREATION BASED ON CONTEXT SOURCE 🟢
    
    # This logic assumes any text retrieved after asking an OCR-type question needs cleaning.
    is_ocr_task = documents and state["question"].lower().strip().startswith(('read', 'extract', 'transcribe', 'what does the image say'))

    if is_ocr_task:
        print("---USING OCR EXTRACTION PROMPT---")
        print("---USING GENERIC OCR CLEANING PROMPT---")
        safe_context = context.replace("{", "{{").replace("}", "}}")
        system_instruction = f"""
        You are an expert data cleaning, formatting, and summarizing assistant. 
        The context provided below is **raw, noisy text** obtained from an Optical Character Recognition (OCR) scan.
        
        YOUR TASK is to **cleanse and structure** this raw text to make it legible, then answer the user's question.
        
        1.  **Cleanse:** Correct spelling errors, combine fragmented words, and remove all obvious noise/junk characters.
        2.  **Format:** Preserve the original structure (paragraphs, lists, or tables) as accurately as possible.
        3.  **Answer:** Use the cleaned and formatted text to provide a concise and clear answer to the user's question: '{question}'.
        
        ## Raw OCR Context
        {safe_context}
        """
        
    else:
        print("---USING STANDARD RAG PROMPT---")
        # Standard RAG prompt for PDF, Wikipedia, or simple chat
        system_instruction = f"""You are an expert helpful, graph-based AI assistant answering questions strictly based on the provided document context and chat history.
            If the answer is not contained in the context, politely respond that you do not know.
            Do not use your general knowledge to answer.
            Answer the user's question based on the provided context (from RAG) and chat history.
            
            ## Relevant Context
            {context}
            
            ## Chat History
            {chat_history}
            """

    # 🟢 Use the dynamically generated system instruction
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            ("human", "{question}"),
        ]
    )
    
    # RAG Chain
    rag_chain = prompt_template | llm
    
    generation = rag_chain.invoke(
        {"context": context, "question": question, "chat_history": chat_history}
    )
    
    # Save current turn to long-term memory
    history.add_user_message(question)
    history.add_ai_message(generation.content)

    return {"generation": generation.content, "question": question, "documents": documents}


# ===============================================
# --- BUILD GRAPH (FROM CURRENT CODE) ---
# ===============================================

def build_graph(checkpointer: InMemorySaver):
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    workflow.add_node("retrieve", retrieve) 
    workflow.add_node("ocr_tool", tesseract_ocr_tool)
    workflow.add_node("generate", generate)
    
    # Conditional Edges from START (Router)
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "retrieve": "retrieve",
            "ocr_tool": "ocr_tool",
            "generate": "generate",
        },
    )
    
    # Edges from Tool Nodes to Final Generation
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("ocr_tool", "generate")
    
    # Final Edge
    workflow.add_edge("generate", END)
    
    # Pass the checkpointer during compilation!


    return workflow.compile(checkpointer=checkpointer)

