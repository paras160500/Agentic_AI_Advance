import os 
from typing import TypedDict

# State Creation

class pipelinestate(TypedDict):
    raw_input : str 
    edited_text : str 
    script_text : str 
    final_output : str 

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model = 'llama-3.3-70b-versatile' , temperature=0.7)

def editor_node(state : pipelinestate) -> dict:
    """
        Stage 1 : Cleans up grammer, remove typos and refines the tone.
    """
    prompt = f"""
        You are an expert copyeditor.Clean up the following raw text.
        Fix any grammatical errors spelling mistakes, and smooth out the transition flow
        while keeping the core message intact. Return only the edited text.\n\n
        Text : \n{state['raw_input']}
    """

    response = llm.invoke(prompt)
    
    return {"edited_text" : response.content}


def scriptwriter_node(state : pipelinestate) -> dict:
    """
        Stage 2 : Formats the clean text into an engaging video script style.
    """
    print("\n--- [Stage 2] Executing Scriptwriter Node ---")

    prompt = (
        f"""You are a charismatic youtube content creator. take this edited text and transform it into"
        a higly engaging , punchy conversational video script hook. Make it soung like a real person
        speaking passionately. return only the script content.\n\n
        Edited Text : \n{state['edited_text']}
        """
    )

    response = llm.invoke(prompt)
    
    return {"script_text" : response.content}


def translater_node(state : pipelinestate) -> dict:
    """
        Stage 3 : Translates the script into the desired language.
    """
    print("\n--- [Stage 3] Executing Translator Node ---")

    prompt = (
        f"""You are a professional translator. Translate the following script into the hinglish language
        while preserving its meaning, tone, and style. Return only the translated script.\n\n
        Script Text : \n{state['script_text']}
        """
    )

    response = llm.invoke(prompt)
    
    return {"final_output" : response.content}



# Making graph
from langgraph.graph import StateGraph, START , END

graph = StateGraph(pipelinestate)
graph.add_node("editor" , editor_node)
graph.add_node("scriptwriter" , scriptwriter_node)
graph.add_node("translator" , translater_node)

graph.add_edge(START , "editor")
graph.add_edge("editor" , "scriptwriter")
graph.add_edge("scriptwriter" , "translator")
graph.add_edge("translator" , END)

app = graph.compile()

result = app.invoke({
    "raw_input" : "Ai Agents are the future of tech.They can plan think and act on their own. Langgraph helps you build these agents with proper control and memory."
})

# output 
print(result['final_output'])

