import os 
from typing import TypedDict, Annotated, List, Literal
from langchain_ollama import OllamaEmbeddings
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START , END
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

def build_retriver(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800 , chunk_overlap = 100)
    chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(
        documents = chunks,
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs = {"k" : 4})



academic_retriver = build_retriver("academics_handbook.pdf")
fee_retriver = build_retriver("fee_structure.pdf")
llm = ChatOllama(model = 'qwen2.5:1.5b' , temperature=0.1)


class State(TypedDict):
    programme : str
    messages : Annotated[List , add_messages]
    query_type : str 
    retrieved_context : str 


def classifier_node(state : State) -> dict:
    """
        Look at the latest message and decide which path to take.
        Args: 
            state(State) : State class object
    """
    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type" : category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = academic_retriver.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriver.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state : State) -> dict:
    """Generate the final answer"""
    query = state['messages'][-1].content
    programme = state.get("programme" , "Unknown")
    context = state['retrieved_context']

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages" : [("ai" , response.content.strip())]}


def router_function(state : State) -> Literal['academic_rag' , 'fee_rag' , 'general']:
    """Route the query based on the query"""
    if state['query_type'] == "academic":
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"
    

graph = StateGraph(State)
graph.add_node("classifier" , classifier_node)
graph.add_node("academic_rag" , academic_rag_node)
graph.add_node("fee_rag" , fee_rag_node)
graph.add_node("general" , general_node)
graph.add_node("response" , response_node)

graph.add_edge(START , "classifier")
graph.add_conditional_edges("classifier" , router_function)
graph.add_edge("academic_rag" , "response")
graph.add_edge("fee_rag" , "response")
graph.add_edge("general" , "response")
graph.add_edge("response" , END)

app = graph.compile()

print("Welcome to the college Assistant \n\n")
print("which programm are you in?")
print("1. BCA\n2. BBA\n3. B.com (H)")

choice = input("\nEnter 1,2 or 3")
program_map = {
    "1" : "BCA",
    "2" : "BBA",
    "3" : "B.Com (H)"
}

student_programme = program_map.get(choice , "BCA")
print(f"Greate you are now set as a {student_programme} student.")

while True:
    user_query = input("You : ")

    if user_query.lower() in ['exit' , 'quit']:
        break

    result = app.invoke({
        "programme" : student_programme,
        "messages" : [{"role" : "human" , "content" : user_query}]
    })

    print(f"Assistant : {result['messages'][-1].content}")
