from qwentastic import model_init, model_data, model_prompt
import requests
import random
import datetime

def main():
    # Example 1: Using Qwen with basic functions
    print("\n=== Example 1: Qwen with Basic Functions ===")
    model_init("Qwen/Qwen1.5-7B-Chat")

    # Define basic functions that don't need API keys
    basic_functions = [
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

    def get_random_number(min: int, max: int):
        return {"number": random.randint(min, max)}

    def get_current_time():
        return {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    basic_function_map = {
        "get_random_number": get_random_number,
        "get_current_time": get_current_time
    }

    model_data(
        "You are a helpful assistant with access to basic functions.",
        functions=basic_functions,
        function_map=basic_function_map
    )

    response = model_prompt(
        "Give me a random number between 1 and 100 and tell me what time it is."
    )
    print(f"Response: {response}\n")

    # Example 2: Using DeepSeek with API-dependent functions
    print("\n=== Example 2: DeepSeek with Weather API ===")
    model_init("deepseek-ai/deepseek-coder-6.7b-instruct")

    # Define functions that need API keys
    api_functions = [
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
            },
            "requires_api_key": True,
            "api_key_name": "openweather_api_key"
        }
    ]

    def get_weather(location: str, api_key: str):
        """Real implementation would use the API key to fetch weather data"""
        return {
            "temperature": 72,
            "condition": "sunny",
            "location": location,
            "note": "This is a mock response. Real implementation would use the API key."
        }

    api_function_map = {
        "get_weather": get_weather
    }

    api_keys = {
        "openweather_api_key": "your-api-key-here"  # Replace with real API key
    }

    model_data(
        "You are a weather assistant.",
        functions=api_functions,
        function_map=api_function_map,
        api_keys=api_keys
    )

    response = model_prompt(
        "What's the weather like in San Francisco?"
    )
    print(f"Response: {response}\n")

    # Example 3: Using Llama with Multiple Functions
    print("\n=== Example 3: Llama with Multiple Functions ===")
    model_init("meta-llama/Llama-2-7b-chat-hf")

    # Combine all functions
    all_functions = basic_functions + api_functions
    all_function_map = {**basic_function_map, **api_function_map}

    model_data(
        """You are a helpful assistant with access to various functions.
        Use these functions to provide accurate and up-to-date information.""",
        functions=all_functions,
        function_map=all_function_map,
        api_keys=api_keys
    )

    response = model_prompt("""
    I need some information:
    1. Get the current time
    2. Get the weather in New York
    3. Generate a random number between 1 and 10
    Please explain what you're doing at each step.
    """)
    print(f"Response: {response}\n")

if __name__ == "__main__":
    main()
