# Qwentastic 🚀

A powerful yet simple interface for running Qwen locally. This package provides an elegant way to interact with any Qwen 1.5 model through three intuitive functions and supports custom function calling.

## 🌟 Features

- **Simple One-Liner Interface**: Just three functions to remember
  - `qwen_init()`: Choose your Qwen model
  - `qwen_data()`: Set context and register custom functions
  - `qwen_prompt()`: Get AI responses
- **Multiple Model Support**: 
  - Qwen 1.5 14B
  - Qwen 1.5 7B
  - Qwen 1.5 4B
  - Qwen 1.5 1.8B
  - Qwen 1.5 0.5B
- **Custom Function Calling**: 
  - Define your own functions in OpenAI format
  - Automatic function execution
  - Support for complex function chaining
  - Easy integration with external APIs
- **Smart Hardware Optimization**:
  - Automatic GPU detection and selection
  - Multi-GPU support with optimal device selection
  - Fallback to CPU when needed

## 📦 Installation

```bash
pip install qwentastic
```

## 🚀 Quick Start

```python
from qwentastic import qwen_init, qwen_data, qwen_prompt

# Initialize with your chosen model
qwen_init("Qwen/Qwen1.5-7B-Chat")

# Define custom functions
functions = [{
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
}]

# Implement function logic
def get_weather(location: str):
    return {
        "temperature": 72,
        "condition": "sunny",
        "location": location
    }

# Register functions with Qwen
qwen_data(
    "You are a helpful AI assistant.",
    functions=functions,
    function_map={"get_weather": get_weather}
)

# Use the functions
response = qwen_prompt("What's the weather like in San Francisco?")
print(response)
```

## 💻 System Requirements

Requirements vary by model:

### Qwen 1.5 14B
- RAM: 32GB minimum
- GPU: 24GB+ VRAM recommended
- Storage: 30GB free space

### Qwen 1.5 7B
- RAM: 16GB minimum
- GPU: 16GB+ VRAM recommended
- Storage: 15GB free space

### Qwen 1.5 4B/1.8B/0.5B
- RAM: 8GB minimum
- GPU: 8GB+ VRAM recommended
- Storage: 8GB free space

Common Requirements:
- Python >= 3.8
- CUDA-capable GPU recommended (but not required)
- accelerate >= 0.27.0 (automatically installed)

## 🔧 Function Calling Guide

### Defining Functions

Functions are defined using the OpenAI function calling format:

```python
functions = [{
    "type": "function",
    "function": {
        "name": "function_name",
        "description": "What the function does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Parameter description"
                },
                "param2": {
                    "type": "integer",
                    "description": "Parameter description"
                }
            },
            "required": ["param1"]
        }
    }
}]
```

### Implementing Functions

Functions are implemented as regular Python functions and mapped to their definitions:

```python
def function_name(param1: str, param2: int = 0):
    # Function implementation
    return {"result": "some result"}

function_map = {
    "function_name": function_name
}
```

### Registering Functions

Functions are registered using qwen_data():

```python
qwen_data(
    "System prompt here",
    functions=functions,
    function_map=function_map
)
```

### Complex Function Usage

Functions can be chained and used in complex ways:

```python
# Define multiple functions
functions = [
    weather_function,
    time_function,
    calculator_function
]

# Register all functions
qwen_data(
    "You can use these functions together",
    functions=functions,
    function_map=function_map
)

# Use multiple functions in one prompt
response = qwen_prompt("""
    1. Get the current time
    2. Get the weather in New York
    3. Calculate the temperature in Fahrenheit
""")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - feel free to use this in your projects!

## ⚠️ Important Notes

- First run requires internet connection for model download
- Model files are cached in the HuggingFace cache directory
- GPU acceleration requires CUDA support
- CPU inference is supported but significantly slower

## 🔍 Troubleshooting

Common issues and solutions:

1. **Out of Memory**:
   - Try a smaller model (e.g., switch from 14B to 7B)
   - Close other GPU-intensive applications
   - Switch to CPU if needed

2. **Slow Inference**:
   - Check GPU utilization
   - Consider using a smaller model
   - Ensure CUDA is properly installed

3. **Function Calling Issues**:
   - Verify function definitions match OpenAI format
   - Check function implementations handle all cases
   - Ensure required parameters are provided

## 📚 Citation

If you use this in your research, please cite:

```bibtex
@software{qwentastic,
  title = {Qwentastic: Simple Interface for Qwen 1.5},
  author = {Jacob Kuchinsky},
  year = {2024},
  url = {https://github.com/MrBanana124/qwentastic}
}
