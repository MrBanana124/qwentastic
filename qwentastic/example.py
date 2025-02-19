from qwentastic import qwen_init, qwen_data, qwen_prompt
import requests
import random
import datetime

def main():
    # Initialize with a specific model
    print("Initializing Qwen 7B model...")
    qwen_init("Qwen/Qwen1.5-7B-Chat")

    # Define custom functions
    functions = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_random_number",
                "description": "Generate a random number between min and max",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min": {
                            "type": "integer",
                            "description": "Minimum value"
                        },
                        "max": {
                            "type": "integer",
                            "description": "Maximum value"
                        }
                    },
                    "required": ["min", "max"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]

    # Implement the functions
    def get_weather(location: str):
        # This is a mock implementation. In real use, you would use an API key
        return {
            "temperature": 72,
            "condition": "sunny",
            "location": location
        }

    def get_random_number(min: int, max: int):
        return {"number": random.randint(min, max)}

    def get_current_time():
        return {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    # Map function names to their implementations
    function_map = {
        "get_weather": get_weather,
        "get_random_number": get_random_number,
        "get_current_time": get_current_time
    }

    # Set up Qwen with the custom functions
    qwen_data(
        """You are a helpful AI assistant with access to various functions.
        Use these functions when appropriate to provide accurate information.""",
        functions=functions,
        function_map=function_map
    )

    # Example prompts demonstrating function usage
    examples = [
        "What's the weather like in San Francisco?",
        "Give me a random number between 1 and 100.",
        "What time is it right now?",
        """I need both the current time and a random number between 1 and 10.
        Please format them nicely in your response."""
    ]

    # Get responses
    for i, prompt in enumerate(examples, 1):
        print(f"\n=== Example {i} ===")
        print(f"Prompt: {prompt}")
        response = qwen_prompt(prompt)
        print(f"Response: {response}")

    # Example of complex function usage
    complex_prompt = """I need some information:
    1. Get the current time
    2. Get the weather in New York
    3. Generate a random number between that city's temperature and 100
    Please explain what you're doing at each step."""
    
    print("\n=== Complex Function Usage ===")
    print(f"Prompt: {complex_prompt}")
    response = qwen_prompt(complex_prompt)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
