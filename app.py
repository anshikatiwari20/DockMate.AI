# Import necessary modules and setup for FastAPI, LangGraph, and LangChain
from fastapi import FastAPI  # FastAPI framework for creating the web application
from httpcore import request
from pydantic import BaseModel  # BaseModel for structured data data models
from typing import List  # List type hint for type annotations
from langchain_tavily import TavilySearch  # TavilySearchResults tool for handling search results from Tavily
import os  # os module for environment variable handling
from dotenv import load_dotenv  # For loading .env file
from langgraph.prebuilt import create_react_agent  # Function to create a ReAct agent
from langchain_groq import ChatGroq  # ChatGroq class for interacting with LLMs
import uvicorn  # Import Uvicorn server for running the FastAPI app


load_dotenv()  # Load environment variables from a .env file

# Retrieve API keys from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
# Ensure API keys are set
if not groq_api_key or not tavily_api_key:
    raise ValueError("GROQ_API_KEY or TAVILY_API_KEY not found in environment variables!")

# Set Tavily API key in environment for TavilySearch
os.environ["TAVILY_API_KEY"] = tavily_api_key

# Predefined list of supported model names
MODEL_NAMES = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b"
]

# Initialize the TavilySearchResults tool with a specified maximum number of results.
tool_tavily = TavilySearch(max_results=2)  # Allows retrieving up to 2 results


# Combine the TavilySearchResults and ExecPython tools into a list.
tools = [tool_tavily, ]

# FastAPI application setup with a title
app = FastAPI(title='LangGraph AI Agent')

# Define the request schema using Pydantic's BaseModel
class RequestState(BaseModel):
    system_prompt: str  # System prompt for initializing the model
    model_name: str  # Name of the model to use for processing the request
    messages: List[str]  # List of messages in the chat

# Define an endpoint for handling chat requests
@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with the chatbot using LangGraph and tools.
    Dynamically selects the model specified in the request.
    """
    if request.model_name not in MODEL_NAMES:
        # Return an error response if the model name is invalid
        return {"error": "Invalid model name. Please select a valid model."}

    # Initialize the LLM with the selected model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=request.model_name)

    # Create a ReAct agent using the selected LLM and tools
    agent = create_react_agent(llm, tools=tools)

    # Create the initial state for processing
    state = {"messages": [request.system_prompt] + request.messages}


    # Process the state using the agent
    try:
        # Invoke the agent
        result = agent.invoke(state)
    except Exception as e:
        return {"error": str(e)}

    # Return the result as the response
    return result

# Run the application if executed as the main script
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)  # Start the app on localhost with port 8000