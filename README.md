# Qwentastic 🚀

A powerful yet simple interface for running Qwen locally. This package provides an elegant way to interact with the Qwen 1.5 14B model through just two intuitive functions.

## 🌟 Features

- **Simple One-Liner Interface**: Just two functions to remember
  - `qwen_data()`: Set context and purpose
  - `qwen_prompt()`: Get AI responses
- **Efficient Model Management**: 
  - Singleton pattern ensures model loads only once
  - Automatic resource management
  - State persistence between calls
- **Smart Memory Handling**:
  - Optimized for both CPU and GPU environments
  - Automatic device detection and optimization
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
from qwentastic import qwen_data, qwen_prompt

# Set the AI's purpose/context
qwen_data("You are a Python expert focused on writing clean, efficient code")

# Get responses
response = qwen_prompt("How do I implement a decorator in Python?")
print(response)
```

## 💻 System Requirements

- Python >= 3.8
- RAM: 16GB minimum (32GB recommended)
- Storage: 30GB free space for model files
- CUDA-capable GPU recommended (but not required)

### Hardware Recommendations
- **CPU**: Modern multi-core processor
- **GPU**: NVIDIA GPU with 12GB+ VRAM (for optimal performance)
- **RAM**: 32GB for smooth operation
- **Storage**: SSD recommended for faster model loading

## ⚡ Performance Notes

First run will:
1. Download the Qwen 1.5 14B model (~30GB)
2. Cache it locally for future use
3. Initialize the model (may take a few minutes)

Subsequent runs will be much faster as the model is cached.

## 🔧 Advanced Usage

### Custom Temperature

```python
from qwentastic import qwen_data, qwen_prompt

# Set creative context
qwen_data("You are a creative storyteller")

# Get more creative responses with higher temperature
response = qwen_prompt(
    "Write a short story about a robot learning to paint",
    temperature=0.8  # More creative (default is 0.7)
)
```

### Memory Management

The package automatically handles model loading and unloading. The model stays in memory until your program exits, optimizing for repeated use while being memory-efficient.

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
   - Try reducing batch size
   - Close other GPU-intensive applications
   - Switch to CPU if needed

2. **Slow Inference**:
   - Check GPU utilization
   - Ensure CUDA is properly installed
   - Consider hardware upgrades for better performance

## 📚 Citation

If you use this in your research, please cite:

```bibtex
@software{qwentastic,
  title = {Qwentastic: Simple Interface for Qwen 1.5},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/qwentastic}
}
