import os 



# First Way to create State using Typedict 
from typing import List, TypedDict

# Creating State class 
class State(TypedDict):
    topic : str 
    summary : str 
    score : str 

#---------------------------------------------------------------------

# Second way to create State using Pydantic
from pydantic import Field, BaseModel, field_validator

class State(BaseModel):
    topic : str 
    score : int 
    summary : str = ""

    @field_validator
    def score_positive(cls,v):
        if v < 0:
            raise ValueError("Score must be positive")
        
#---------------------------------------------------------------------

# Python Dataclasses : Use not much in reallife 
from dataclasses import dataclass , field

@dataclass
class State:
    topic : str = ""
    summary : str = ""
    messages : list = field(default_factory=list)

#---------------------------------------------------------------------

from langgraph.graph import MessagesState

class State(MessagesState):
    # messages_file is already included with the add_messages reducer
    user_name : str 
    language : str 
