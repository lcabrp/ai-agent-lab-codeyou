# app.py

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain_core.tools import Tool
from datetime import datetime
import time

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

def get_current_time(_: str) -> str:
    """
    Returns the current date and time as a formatted string.

    Args:
        _ (str): A required parameter for the Tool interface (not used).

    Returns:
        str: The current date and time in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def reverse_string(input_string: str) -> str:
    """
    Reverses a given string.

    Args:
        input_string (str): The string to reverse.

    Returns:
        str: The reversed string.
    """
    return input_string[::-1]

def get_weather(date: str) -> str:
    """
    Returns weather information for a given date.
    
    Args:
        date (str): The date in the format 'YYYY-MM-DD'.
    
    Returns:
        str: Weather information for the given date.
    """
    try:
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Return weather based on whether the date matches today
        if date == today:
            return "Sunny, 72°F"
        else:
            return "Rainy, 55°F"
    except Exception as e:
        return f"Error retrieving weather: {str(e)}"

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

    # Create a tools list with Tool objects for the calculator, current time, reverse string, and weather
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Use this tool to evaluate mathematical expressions. Provide the expression as a string, and it will return the result."
        ),
        Tool(
            name="get_current_time",
            func=get_current_time,
            description="Use this tool to get the current date and time. It returns the current timestamp in the format 'YYYY-MM-DD HH:MM:SS'."
        ),
        Tool(
            name="reverse_string",
            func=reverse_string,
            description="Reverses a string. Input should be a single string."
        ),
        Tool(
            name="get_weather",
            func=get_weather,
            description="Get weather information for a specific date. Input should be a date string in the format 'YYYY-MM-DD'. Use this tool to retrieve weather data for a given date."
        )
    ]

    print("🛠️ Tools initialized successfully! 🎉")
    # System message to guide the agent behavior
    system_prompt = (
        "You are a professional and helpful AI assistant. "
        "Provide concise and direct responses. "
        "Use the available tools when needed to answer questions accurately."
    )
    # Create an agent using create_agent
    agent_executor = create_agent(
        model=chat,
        tools=tools,
        debug=False,  # Enable verbose output
        system_prompt=system_prompt
    )

    print("🤖 Agent created successfully! 🎉")

    

    # List of test queries
    test_queries = [
        "What time is it right now?",
        "What is 25 * 4 + 10?",
        "Reverse the string 'Hello World'",
        "What's the weather like today?"
    ]

    print("Running example queries:")
    print()

    for query in test_queries:
        print("\n📝 Query:", query)
        print("─" * 50)
        try:
            messages = [
                #SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            result = agent_executor.invoke({"messages": messages})
            time.sleep(1)  # Add a small delay to avoid hitting rate limits
            # Extract the final response from the messages list
            final_message = result['messages'][-1].content
            print("\n✅ Result:", final_message)
        except Exception as e:
            print("❌ Error while processing query:", str(e))
        print()

    print("\n" + "─" * 50)
    print("🎉 Agent demo complete!")

if __name__ == "__main__":
    main()
