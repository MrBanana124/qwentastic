import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QwenManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not QwenManager._initialized:
            self.model_name = "Qwen/Qwen1.5-14B-Chat"
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            self.system_prompt = "You are a helpful AI assistant."
            print("Model loaded and ready!")
            QwenManager._initialized = True

    def generate_response(self, user_input, max_length=2048, temperature=0.7):
        """Generate a response to user input"""
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

    def set_system_prompt(self, prompt):
        """Update the system prompt"""
        self.system_prompt = prompt

# Global instance
_manager = None

def _get_manager():
    global _manager
    if _manager is None:
        _manager = QwenManager()
    return _manager

def qwen_data(background: str) -> None:
    """
    Set the background information and purpose for Qwen.
    
    Args:
        background (str): The system prompt or background information for Qwen
    """
    manager = _get_manager()
    manager.set_system_prompt(background)

def qwen_prompt(prompt: str) -> str:
    """
    Send a prompt to Qwen and get the response.
    
    Args:
        prompt (str): The input prompt for Qwen
        
    Returns:
        str: Qwen's response to the prompt
    """
    manager = _get_manager()
    return manager.generate_response(prompt)
