# Qwen Package

A simple interface for running Qwen locally with two easy-to-use functions.

## Installation

```bash
pip install qwen-package
```

## Usage

```python
from qwen_package import qwen_data, qwen_prompt

# Set background/purpose for Qwen
qwen_data("You are a helpful coding assistant")

# Get responses from Qwen
response = qwen_prompt("How do I use Python decorators?")
print(response)
```

## Features

- Simple one-liner interface
- Efficient model loading (loads only once)
- Maintains state between calls
- Easy to use in any Python project

## Requirements

- Python >= 3.8
- torch >= 2.0.0
- transformers >= 4.32.0

## License

MIT License
