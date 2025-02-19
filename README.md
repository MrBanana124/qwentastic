# Qwentastic 🚀

A powerful yet simple interface for running Qwen locally. This package provides an elegant way to interact with any Qwen 1.5 model through just three intuitive functions.

## 🌟 Features

- **Simple One-Liner Interface**: Just three functions to remember
  - `qwen_init()`: Choose your Qwen model
  - `qwen_data()`: Set context and purpose
  - `qwen_prompt()`: Get AI responses
- **Multiple Model Support**: 
  - Qwen 1.5 14B
  - Qwen 1.5 7B
  - Qwen 1.5 4B
  - Qwen 1.5 1.8B
  - Qwen 1.5 0.5B
- **Efficient Model Management**: 
  - Singleton pattern ensures model loads only once
  - Automatic resource management
  - State persistence between calls
- **Smart Memory Handling**:
  - Optimized with accelerate for better performance
  - Automatic device detection and optimization
  - Efficient model switching
- **Production Ready**:
  - Thread-safe implementation
  - Error handling and recovery
  - Detailed logging

## 📦 Installation

```bash
pip install qwentastic
```

## 🚀 Quick Start

```python
from qwentastic import qwen_init, qwen_data, qwen_prompt

# Initialize with your chosen model (defaults to 14B if not specified)
qwen_init("Qwen/Qwen1.5-7B-Chat")  # Choose a smaller model for faster responses

# Set the AI's purpose/context
qwen_data("You are a Python expert focused on writing clean, efficient code")

# Get responses
response = qwen_prompt("How do I implement a decorator in Python?")
print(response)

# Switch to a different model mid-session
qwen_init("Qwen/Qwen1.5-4B-Chat")  # Switch to an even smaller model
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

## ⚡ Performance Notes

First run will:
1. Download the selected Qwen model
2. Cache it locally for future use
3. Initialize the model (may take a few minutes)

Subsequent runs will be much faster as the model is cached.

## 🔧 Advanced Usage

### Model Selection

```python
from qwentastic import qwen_init, qwen_data, qwen_prompt

# Available models:
qwen_init("Qwen/Qwen1.5-14B-Chat")  # Highest quality, slower
qwen_init("Qwen/Qwen1.5-7B-Chat")   # Good balance
qwen_init("Qwen/Qwen1.5-4B-Chat")   # Faster
qwen_init("Qwen/Qwen1.5-1.8B-Chat") # Very fast
qwen_init("Qwen/Qwen1.5-0.5B-Chat") # Fastest
```

### Temperature Control

```python
# More deterministic responses
response = qwen_prompt("Write a function", temperature=0.3)

# More creative responses
response = qwen_prompt("Write a story", temperature=0.9)
```

### Memory Management

The package automatically handles model loading and unloading. You can switch models at any time using `qwen_init()`. The old model will be properly unloaded to free up memory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - feel free to use this in your projects!

## ⚠️ Important Notes

- First run requires internet connection for model download
- Model files are cached in the HuggingFace cache directory
- GPU acceleration requires CUDA support
- CPU inference is supported but significantly slower
- Uses accelerate for optimal performance

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

## 📚 Citation

If you use this in your research, please cite:

```bibtex
@software{qwentastic,
  title = {Qwentastic: Simple Interface for Qwen 1.5},
  author = {Jacob Kuchinsky},
  year = {2024},
  url = {https://github.com/MrBanana124/qwentastic}
}
