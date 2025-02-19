import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Any
import json

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
            self.functions = []
            self.function_map = {}
            QwenManager._initialized = True

    def initialize_model(self, model_name: str = "Qwen/Qwen1.5-14B-Chat"):
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
        ).to(self.device)
        
        print("Model loaded and ready!")

    def register_functions(self, functions: List[Dict[str, Any]], function_map: Dict[str, callable]):
        """Register available functions and their implementations"""
        self.functions = functions
        self.function_map = function_map

    def _extract_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Extract function calls from the model's response"""
        function_calls = []
        try:
            # Look for function calls in the format: <function_call>{"name": "...", "arguments": {...}}</function_call>
            while "<function_call>" in text and "</function_call>" in text:
                start = text.find("<function_call>") + len("<function_call>")
                end = text.find("</function_call>")
                if start > 0 and end > start:
                    function_text = text[start:end].strip()
                    function_data = json.loads(function_text)
                    function_calls.append(function_data)
                    text = text[end + len("</function_call>"):]
        except json.JSONDecodeError:
            pass
        return function_calls

    def _execute_function(self, function_call: Dict[str, Any]) -> str:
        """Execute a function call and return the result"""
        try:
            name = function_call["name"]
            arguments = function_call.get("arguments", {})
            if name in self.function_map:
                result = self.function_map[name](**arguments)
                return json.dumps(result)
            return f"Error: Function {name} not found"
        except Exception as e:
            return f"Error executing function: {str(e)}"

    def generate_response(self, user_input: str, max_length: int = 2048, temperature: float = 0.7) -> str:
        """Generate a response to user input"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call qwen_init() first.")

        # Create the system message with function definitions
        system_message = self.system_prompt + "\n\nAvailable functions:\n"
        if self.functions:
            system_message += json.dumps(self.functions, indent=2)

        # Create the full prompt
        full_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        
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

        # Look for and execute function calls
        function_calls = self._extract_function_calls(response)
        for call in function_calls:
            result = self._execute_function(call)
            response = response.replace(
                f'<function_call>{json.dumps(call)}</function_call>',
                f'<function_result>{result}</function_result>'
            )

        return response

    def set_system_prompt(self, prompt: str, functions: List[Dict[str, Any]] = None, function_map: Dict[str, callable] = None):
        """Update the system prompt and available functions"""
        self.system_prompt = prompt
        if functions is not None:
            self.functions = functions
        if function_map is not None:
            self.function_map = function_map

# Global instance
_manager = None

def _get_manager() -> QwenManager:
    global _manager
    if _manager is None:
        _manager = QwenManager()
    return _manager

def qwen_init(model: str = "Qwen/Qwen1.5-14B-Chat") -> None:
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

def qwen_data(background: str, functions: List[Dict[str, Any]] = None, function_map: Dict[str, callable] = None) -> None:
    """
    Set the background information and register available functions for Qwen.
    
    Args:
        background (str): The system prompt or background information for Qwen
        functions (List[Dict[str, Any]], optional): List of function definitions in OpenAI format
        function_map (Dict[str, callable], optional): Dictionary mapping function names to their implementations
        
    Example:
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
        
        function_map = {
            "get_weather": lambda location: {"temperature": 72, "condition": "sunny"}
        }
        
        qwen_data("You are a weather assistant.", functions, function_map)
    """
    manager = _get_manager()
    manager.set_system_prompt(background, functions, function_map)

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
