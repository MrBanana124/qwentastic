import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Literal

# Available Qwen models
QwenModel = Literal[
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-0.5B-Chat",
]

class QwenManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QwenManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not QwenManager._initialized:
            self.model_name = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = None
            self.tokenizer = None
            self.system_prompt = "You are a helpful AI assistant."
            QwenManager._initialized = True

    def initialize_model(self, model_name: QwenModel = "Qwen/Qwen1.5-14B-Chat"):
        """Initialize or switch to a different Qwen model"""
        print(f"Using device: {self.device}")
        self.model_name = model_name
        
        print(f"Loading {model_name} tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"Loading {model_name} model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        print("Model loaded and ready!")

    def generate_response(self, user_input: str, max_length: int = 2048, temperature: float = 0.7) -> str:
        """Generate a response to user input"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call qwen_init() first.")

        full_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        
        return response

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt"""
        self.system_prompt = prompt

# Global instance
_manager = None

def _get_manager() -> QwenManager:
    global _manager
    if _manager is None:
        _manager = QwenManager()
    return _manager

def qwen_init(model: QwenModel = "Qwen/Qwen1.5-14B-Chat") -> None:
    """
    Initialize or switch to a specific Qwen model.
    
    Args:
        model: The Qwen model to use. Available options:
            - "Qwen/Qwen1.5-14B-Chat" (default)
            - "Qwen/Qwen1.5-7B-Chat"
            - "Qwen/Qwen1.5-4B-Chat"
            - "Qwen/Qwen1.5-1.8B-Chat"
            - "Qwen/Qwen1.5-0.5B-Chat"
    """
    manager = _get_manager()
    manager.initialize_model(model)

def qwen_data(background: str) -> None:
    """
    Set the background information and purpose for Qwen.
    
    Args:
        background (str): The system prompt or background information for Qwen
    """
    manager = _get_manager()
    manager.set_system_prompt(background)

def qwen_prompt(prompt: str, max_length: int = 2048, temperature: float = 0.7) -> str:
    """
    Send a prompt to Qwen and get the response.
    
    Args:
        prompt (str): The input prompt for Qwen
        max_length (int, optional): Maximum length of the generated response. Defaults to 2048.
        temperature (float, optional): Sampling temperature. Higher values make output more random. Defaults to 0.7.
        
    Returns:
        str: Qwen's response to the prompt
        
    Raises:
        RuntimeError: If qwen_init() hasn't been called to initialize a model
    """
    manager = _get_manager()
    return manager.generate_response(prompt, max_length, temperature)
