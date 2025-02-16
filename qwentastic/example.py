from qwentastic import qwen_data, qwen_prompt

def main():
    # Set the background/purpose for Qwen
    qwen_data("You are a Python coding assistant focused on providing clear, concise explanations.")

    # Example prompts
    prompts = [
        "What is a list comprehension in Python?",
        "How do I handle exceptions in Python?"
    ]

    # Get responses
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = qwen_prompt(prompt)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
