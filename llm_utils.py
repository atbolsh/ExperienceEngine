"""
LLM utilities providing both local Qwen3 0.6B for text-only operations
and GPT-5 for vision-based operations.

The local model is loaded via HuggingFace and will be downloaded automatically
on first use (approximately 1.2GB).

Set FORCE_CPU=1 environment variable to use CPU mode (avoids CUDA issues).
"""

import os
from typing import Optional, List, Any
from functools import lru_cache

# Check for CPU-only mode BEFORE importing torch
# This must happen before any torch import to take effect
_force_cpu = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
if _force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Global instances for caching
_local_llm = None
_local_pipeline = None
_vision_llm = None


def _load_model_on_device(model_id, tokenizer, device, torch_dtype):
    """
    Helper to load model on specified device.
    
    Args:
        model_id: HuggingFace model ID
        tokenizer: Loaded tokenizer
        device: 'cuda' or 'cpu'
        torch_dtype: torch.float16 or torch.float32
        
    Returns:
        Loaded model and pipeline
    """
    from transformers import AutoModelForCausalLM, pipeline
    import torch
    
    if device == "cuda":
        # Use device_map for CUDA - don't specify device in pipeline
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        # When using device_map, don't specify device in pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=tokenizer.pad_token_id,
            # No device argument when using device_map
        )
    else:
        # For CPU: load without device_map, then move to CPU explicitly
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to("cpu")
        
        # Create pipeline with explicit CPU device
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
            pad_token_id=tokenizer.pad_token_id,
            device="cpu",
        )
    
    return model, pipe


def get_local_llm():
    """
    Get the local Qwen3 0.6B model for text-only operations.
    Uses HuggingFace transformers with LangChain integration.
    
    Note: Qwen3 0.6B is a small model and may not support complex tool calling.
    The agent.py is configured to use a structured chat approach that works
    better with smaller models.
    
    Set environment variable FORCE_CPU=1 to skip CUDA and use CPU directly.
    
    Returns:
        A LangChain-compatible Chat LLM instance
    """
    global _local_llm, _local_pipeline
    
    if _local_llm is not None:
        return _local_llm
    
    print("[LLM] Loading local Qwen3-0.6B model...")
    print("[LLM] First load will download ~1.2GB model files...")
    
    try:
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
        from transformers import AutoTokenizer
        import torch
        
        model_id = "Qwen/Qwen3-0.6B"
        
        # Load tokenizer first (same for CPU/GPU)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Ensure pad token is set (Qwen models may not have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check if we should force CPU mode
        force_cpu = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
        
        # Determine device
        use_cuda = torch.cuda.is_available() and not force_cpu
        
        if use_cuda:
            print("[LLM] Attempting to use CUDA device...")
            try:
                model, pipe = _load_model_on_device(
                    model_id, tokenizer, "cuda", torch.float16
                )
                print("[LLM] CUDA device loaded successfully")
            except (RuntimeError, Exception) as cuda_error:
                # Check if this is a CUDA compatibility error
                error_str = str(cuda_error).lower()
                if "cuda" in error_str or "kernel" in error_str or "accelerator" in error_str:
                    print(f"[LLM] CUDA error detected: {cuda_error}")
                    print("[LLM] Falling back to CPU mode...")
                    # Clear any partial CUDA state and disable CUDA
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
                    # Disable CUDA for this process to prevent further issues
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    model, pipe = _load_model_on_device(
                        model_id, tokenizer, "cpu", torch.float32
                    )
                    print("[LLM] CPU fallback loaded successfully")
                else:
                    raise
        else:
            if force_cpu:
                print("[LLM] FORCE_CPU=1 set, using CPU mode")
                # Also set environment to prevent any CUDA usage
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            else:
                print("[LLM] CUDA not available, using CPU (inference will be slower)")
            model, pipe = _load_model_on_device(
                model_id, tokenizer, "cpu", torch.float32
            )
        
        _local_pipeline = pipe
        
        # Wrap in LangChain
        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        
        # Create chat model wrapper for instruction-tuned interface
        _local_llm = ChatHuggingFace(llm=hf_pipeline)
        
        print("[LLM] Qwen3-0.6B loaded successfully")
        return _local_llm
        
    except ImportError as e:
        print(f"[LLM] Error: Required packages not installed. Run:")
        print("  pip install langchain-huggingface transformers torch accelerate")
        raise ImportError(
            "Local LLM requires: langchain-huggingface, transformers, torch, accelerate. "
            f"Original error: {e}"
        )
    except Exception as e:
        print(f"[LLM] Error loading local model: {e}")
        raise


def get_local_pipeline():
    """
    Get the raw HuggingFace pipeline for the local model.
    Useful for direct text generation without the LangChain wrapper.
    
    Returns:
        HuggingFace text-generation pipeline
    """
    global _local_pipeline
    if _local_pipeline is None:
        # Initialize via get_local_llm which sets up the pipeline
        get_local_llm()
    return _local_pipeline


def get_vision_llm():
    """
    Get the GPT-5 model for vision/image operations.
    
    Returns:
        A LangChain ChatOpenAI instance configured for vision
    """
    global _vision_llm
    
    if _vision_llm is not None:
        return _vision_llm
    
    from langchain_openai import ChatOpenAI
    
    _vision_llm = ChatOpenAI(
        model="gpt-5",
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    print("[LLM] GPT-5 vision model configured")
    return _vision_llm


def get_text_llm():
    """
    Alias for get_local_llm() - returns the text-only LLM.
    Use this for all text-only operations in the main engine and tools.
    """
    return get_local_llm()


def reset_llm_cache():
    """
    Reset the cached LLM instances.
    Useful for testing or if you need to reload models.
    """
    global _local_llm, _local_pipeline, _vision_llm
    _local_llm = None
    _local_pipeline = None
    _vision_llm = None
    print("[LLM] LLM cache cleared")


def generate_text(prompt: str, max_tokens: int = 512) -> str:
    """
    Direct text generation using the local model.
    Useful for simple text completion tasks.
    
    Args:
        prompt: The text prompt to complete
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text string
    """
    pipe = get_local_pipeline()
    result = pipe(prompt, max_new_tokens=max_tokens)
    return result[0]['generated_text']

