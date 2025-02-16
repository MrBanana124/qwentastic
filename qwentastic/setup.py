from setuptools import setup, find_packages

setup(
    name="qwen_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.32.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple interface for running Qwen locally",
    long_description="""
    This package provides a simple interface for running Qwen locally with two main functions:
    - qwen_data(): Set background information and purpose
    - qwen_prompt(): Send prompts and get responses
    """,
    keywords="qwen, ai, language model",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
