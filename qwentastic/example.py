from qwentastic import qwen_init, qwen_data, qwen_prompt

def main():
    # Initialize with a specific model (defaults to 14B if not specified)
    print("Initializing Qwen 7B model...")
    qwen_init("Qwen/Qwen1.5-7B-Chat")

    # Set the background/purpose for Qwen
    qwen_data("You are a Python coding assistant focused on providing clear, concise explanations.")

    # Example prompts with different temperatures
    examples = [
        ("What is a list comprehension in Python?", 0.7),  # Default temperature
        ("Write a creative story about coding in Python", 0.9),  # Higher temperature for more creative output
    ]

    # Get responses
    for prompt, temp in examples:
        print(f"\nPrompt: {prompt}")
        print(f"Temperature: {temp}")
        response = qwen_prompt(prompt, temperature=temp)
        print(f"Response: {response}")

    # Switch to a different model mid-session
    print("\nSwitching to 4B model...")
    qwen_init("Qwen/Qwen1.5-4B-Chat")
    
    # Try another prompt with the new model
    prompt = "Explain the difference between a list and a tuple in Python"
    print(f"\nPrompt: {prompt}")
    response = qwen_prompt(prompt)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
