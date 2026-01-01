# ============================================
# LangChain AI Agent using Groq API 
# Domain: IT Support
# ============================================

import os

# -----------------------------
# Set Groq API Key
# -----------------------------
os.environ["GROQ_API_KEY"] = "" 

# -----------------------------
# Imports
# -----------------------------
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

# -----------------------------
# Initialize Groq LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# -----------------------------
# Define Tools
# -----------------------------
def check_system_status(_):
    return "All IT systems are operational."

def reset_user_password(username: str):
    return f"Password for user '{username}' has been reset successfully."

def troubleshoot_issue(issue: str):
    return f"Steps to fix '{issue}': Restart device, check cables, reconnect network."

tools = [
    Tool(
        name="SystemStatus",
        func=check_system_status,
        description="Check current IT system status",
        return_direct=True  
    ),
    Tool(
        name="ResetPassword",
        func=reset_user_password,
        description="Reset a user's password"
    ),
    Tool(
        name="Troubleshoot",
        func=troubleshoot_issue,
        description="Troubleshoot IT-related problems"
    )
]

# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -----------------------------
# ReAct Prompt
# -----------------------------
prompt = PromptTemplate.from_template(
    """You are an IT support assistant.

You have access to the following tools:
{tools}

Tool names:
{tool_names}

Use the following format:

Question: the input question
Thought: your reasoning
Action: the tool name from [{tool_names}]
Action Input: the input to the tool
Observation: the result
Final Answer: the final response

Question: {input}
Thought:{agent_scratchpad}
"""
)

# -----------------------------
# Create ReAct Agent
# -----------------------------
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# -----------------------------
# Agent Executor
# -----------------------------
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=3,              
    early_stopping_method="force"
)

# -----------------------------
# Run Agent
# -----------------------------
print("\n--- AI IT Support Agent (Groq + ReAct) ---\n")

while True:
    user_input = input("User: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Agent: Goodbye!")
        break

    response = agent_executor.invoke({"input": user_input})
    print("Agent:", response["output"])
