import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Any, Literal
import json

# Available model types
ModelType = Literal[
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-0.5B-Chat",
    "deepseek-ai/deepseek-coder-33b-instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-7b-chat-hf",
]

class ModelManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelManager._initialized:
            self.model_name = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = None
            self.tokenizer = None
            self.system_prompt = "You are a helpful AI assistant."
            self.functions = []
            self.function_map = {}
            self.api_keys = {}
            ModelManager._initialized = True

    def initialize_model(self, model_name: ModelType = "Qwen/Qwen1.5-7B-Chat"):
        """Initialize or switch to a different model"""
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

    def set_api_keys(self, api_keys: Dict[str, str]):
        """Set API keys for functions that need them"""
        self.api_keys = api_keys

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
            
            # Add API key to arguments if function needs it
            function_def = next((f for f in self.functions if f["function"]["name"] == name), None)
            if function_def and function_def.get("requires_api_key"):
                api_key_name = function_def.get("api_key_name", f"{name}_api_key")
                if api_key_name in self.api_keys:
                    arguments["api_key"] = self.api_keys[api_key_name]
                else:
                    return f"Error: {name} requires API key '{api_key_name}' which was not provided"
            
            if name in self.function_map:
                result = self.function_map[name](**arguments)
                return json.dumps(result)
            return f"Error: Function {name} not found"
        except Exception as e:
            return f"Error executing function: {str(e)}"

    def generate_response(self, user_input: str, max_length: int = 2048, temperature: float = 0.7) -> str:
        """Generate a response to user input"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call model_init() first.")

        # Create the system message with function definitions
        system_message = self.system_prompt + "\n\nAvailable functions:\n"
        if self.functions:
            system_message += json.dumps(self.functions, indent=2)

        # Create the full prompt based on model type
        if "Qwen" in self.model_name:
            full_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
        elif "deepseek" in self.model_name:
            full_prompt = f"### System:\n{system_message}\n\n### User:\n{user_input}\n\n### Assistant:\n"
        elif "llama" in self.model_name:
            full_prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_input} [/INST]"
        else:
            full_prompt = f"{system_message}\n\nUser: {user_input}\nAssistant:"
        
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
        
        # Extract assistant's response based on model type
        if "Qwen" in self.model_name:
            response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        elif "deepseek" in self.model_name:
            response = response.split("### Assistant:\n")[-1].split("### User:")[0].strip()
        elif "llama" in self.model_name:
            response = response.split("[/INST]")[-1].split("</s>")[0].strip()
        else:
            response = response.split("Assistant:")[-1].strip()

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

def _get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager

def model_init(model: ModelType = "Qwen/Qwen1.5-7B-Chat") -> None:
    """
    Initialize or switch to a specific model.
    
    Args:
        model: The model to use. Available options:
            Qwen Models:
            - "Qwen/Qwen1.5-14B-Chat"
            - "Qwen/Qwen1.5-7B-Chat"
            - "Qwen/Qwen1.5-4B-Chat"
            - "Qwen/Qwen1.5-1.8B-Chat"
            - "Qwen/Qwen1.5-0.5B-Chat"
            
            DeepSeek Models:
            - "deepseek-ai/deepseek-coder-33b-instruct"
            - "deepseek-ai/deepseek-coder-6.7b-instruct"
            
            Llama Models:
            - "meta-llama/Llama-2-70b-chat-hf"
            - "meta-llama/Llama-2-13b-chat-hf"
            - "meta-llama/Llama-2-7b-chat-hf"
    """
    manager = _get_manager()
    manager.initialize_model(model)

def model_data(
    background: str,
    functions: List[Dict[str, Any]] = None,
    function_map: Dict[str, callable] = None,
    api_keys: Dict[str, str] = None
) -> None:
    """
    Set the background information and register available functions.
    
    Args:
        background (str): The system prompt or background information
        functions (List[Dict[str, Any]], optional): List of function definitions in OpenAI format
        function_map (Dict[str, callable], optional): Dictionary mapping function names to implementations
        api_keys (Dict[str, str], optional): Dictionary of API keys for functions that need them
        
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
            },
            "requires_api_key": True,
            "api_key_name": "openweather_api_key"
        }]
        
        function_map = {
            "get_weather": lambda location, api_key: get_weather_data(location, api_key)
        }
        
        api_keys = {
            "openweather_api_key": "your-api-key-here"
        }
        
        model_data(
            "You are a weather assistant.",
            functions=functions,
            function_map=function_map,
            api_keys=api_keys
        )
    """
    manager = _get_manager()
    manager.set_system_prompt(background, functions, function_map)
    if api_keys:
        manager.set_api_keys(api_keys)

def model_prompt(prompt: str, max_length: int = 2048, temperature: float = 0.7) -> str:
    """
    Send a prompt to the model and get the response.
    
    Args:
        prompt (str): The input prompt
        max_length (int, optional): Maximum length of the generated response. Defaults to 2048.
        temperature (float, optional): Sampling temperature. Higher values make output more random. Defaults to 0.7.
        
    Returns:
        str: Model's response to the prompt
        
    Raises:
        RuntimeError: If model_init() hasn't been called to initialize a model
    """
    manager = _get_manager()
    return manager.generate_response(prompt, max_length, temperature)

# Aliases for backward compatibility
qwen_init = model_init
qwen_data = model_data
qwen_prompt = model_prompt
