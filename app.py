# app.py

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import Tool
from datetime import datetime

def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression provided as a string.

    Args:
        expression (str): The mathematical expression to evaluate.

    Returns:
        str: The result of the evaluation or an error message.
    """
    try:
        # Evaluate the expression using eval (for demo purposes, use with caution)
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Check if GITHUB_TOKEN exists in environment variables
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ Error: GITHUB_TOKEN not found in environment variables.")
        print("💡 Please create a .env file with the following content:")
        print("   GITHUB_TOKEN=your_personal_access_token")
        print("🔗 You can generate a token at: https://github.com/settings/tokens")
        return  # Exit early if no token is found

    print("✅ GITHUB_TOKEN loaded successfully! 🎉")

    # Create a ChatOpenAI instance
    chat = ChatOpenAI(
        model="openai/gpt-4o",
        temperature=0,
        base_url="https://models.github.ai/inference",
        api_key=github_token
    )

    print("🤖 ChatOpenAI instance created successfully! 🎉")

    # Create a tools list with a Tool object for the calculator
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Use this tool to evaluate mathematical expressions. Provide the expression as a string, and it will return the result."
        )
    ]

    print("🛠️ Tools initialized successfully! 🎉")

    # Create an agent using create_agent
    agent_executor = create_agent(
        model=chat,
        tools=tools,
        debug=True  # Enable verbose output
    )

    print("🤖 Agent created successfully! 🎉")

    # Test query
    test_query = "What is 25 * 4 + 10?"
    print(f"📝 Sending query to agent: {test_query}")

    try:
        # Explicitly construct the messages array
        messages = [HumanMessage(content=test_query)]
        result = agent_executor.invoke({"messages": messages})

        # Extract the final AI message
        messages = result["messages"]
        final_message = messages[-1].content

        print("💬 Agent Output:", final_message)
        #result = agent_executor.invoke({"messages": messages})
        #print("💬 Agent Output:", result['output'])
    except Exception as e:
        print("❌ Error while executing agent query:", str(e))

    print("✅ Query processed successfully! 🎉" )

if __name__ == "__main__":
    main()
