"""
LLM utilities: local Qwen3-0.6B for text and local Qwen2-VL for vision.

Set FORCE_CPU=1 to avoid CUDA. Override vision model with VISION_MODEL_ID (default Qwen2-VL-2B-Instruct).
"""

import base64
import io
import os
from typing import Any, List, Optional, Tuple

# Check for CPU-only mode BEFORE importing torch
_force_cpu = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
if _force_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Global instances for caching
_local_llm = None
_local_pipeline = None
_vision_llm = None


def _clear_stale_max_length_on_pipeline(pipe) -> None:
    """
    Hugging Face merges the text-generation pipeline's max_new_tokens with the model's
    GenerationConfig, which often still carries the library default max_length=20.
    That triggers: "Both max_new_tokens and max_length seem to have been set".
    We rely on max_new_tokens only for the agent; clear max_length on the pipeline copy.
    """
    gc = getattr(pipe, "generation_config", None)
    if gc is None:
        return
    try:
        mnt = getattr(gc, "max_new_tokens", None)
        if mnt is not None and mnt > 0:
            gc.max_length = None
    except Exception:
        pass


def _load_model_on_device(model_id, tokenizer, device, torch_dtype):
    """
    Helper to load causal LM on specified device (used by Qwen3 text model).
    """
    from transformers import AutoModelForCausalLM, pipeline
    import torch

    # Qwen3 recommended: temperature=0.6, top_p=0.95, top_k=20 for thinking mode.
    gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.1,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **gen_kwargs)
        _clear_stale_max_length_on_pipeline(pipe)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to("cpu")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu", **gen_kwargs)
        _clear_stale_max_length_on_pipeline(pipe)

    return model, pipe


def get_local_llm():
    """
    Local Qwen3 0.6B for text-only agent loop (LangChain ChatHuggingFace).
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
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        force_cpu = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
        use_cuda = torch.cuda.is_available() and not force_cpu

        if use_cuda:
            print("[LLM] Attempting to use CUDA device...")
            try:
                model, pipe = _load_model_on_device(
                    model_id, tokenizer, "cuda", torch.float16
                )
                print("[LLM] CUDA device loaded successfully")
            except (RuntimeError, Exception) as cuda_error:
                error_str = str(cuda_error).lower()
                if "cuda" in error_str or "kernel" in error_str or "accelerator" in error_str:
                    print(f"[LLM] CUDA error detected: {cuda_error}")
                    print("[LLM] Falling back to CPU mode...")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
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
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            else:
                print("[LLM] CUDA not available, using CPU (inference will be slower)")
            model, pipe = _load_model_on_device(
                model_id, tokenizer, "cpu", torch.float32
            )

        _local_pipeline = pipe
        hf_pipeline = HuggingFacePipeline(pipeline=pipe)
        _local_llm = ChatHuggingFace(llm=hf_pipeline)
        print("[LLM] Qwen3-0.6B loaded successfully")
        return _local_llm

    except ImportError as e:
        print("[LLM] Error: Required packages not installed. Run:")
        print("  pip install langchain-huggingface transformers torch accelerate")
        raise ImportError(
            "Local LLM requires: langchain-huggingface, transformers, torch, accelerate. "
            f"Original error: {e}"
        ) from e
    except Exception as e:
        print(f"[LLM] Error loading local model: {e}")
        raise


def get_local_pipeline():
    """Raw HuggingFace text-generation pipeline for Qwen3."""
    global _local_pipeline
    if _local_pipeline is None:
        get_local_llm()
    return _local_pipeline


def _data_url_to_pil(url: str):
    from PIL import Image

    if not url.startswith("data:"):
        raise ValueError("Expected data: image URL")
    _, b64part = url.split(",", 1)
    raw = base64.b64decode(b64part)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _langchain_messages_to_prompt_and_image(
    messages: List[Any],
) -> Tuple[str, Any]:
    """Extract text + PIL image from LangChain HumanMessage (OpenAI-style content blocks)."""
    from langchain_core.messages import HumanMessage

    texts: List[str] = []
    pil_image = None

    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue
        c = msg.content
        if isinstance(c, str):
            texts.append(c)
            continue
        if isinstance(c, list):
            for part in c:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    texts.append(part.get("text") or "")
                elif ptype == "image_url":
                    u = part.get("image_url")
                    if isinstance(u, dict):
                        u = u.get("url") or ""
                    if isinstance(u, str) and u.startswith("data:"):
                        pil_image = _data_url_to_pil(u)

    prompt = "\n".join(t for t in texts if t).strip()
    return prompt, pil_image


class Qwen2VLVisionChat:
    """
    Minimal ChatOpenAI-like object: invoke([HumanMessage]) -> object with .content str.
    """

    def __init__(self, model, processor, device: str, max_new_tokens: int):
        self.model = model
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens

    def invoke(self, messages: List[Any]):
        from qwen_vl_utils import process_vision_info
        import torch

        prompt, pil_image = _langchain_messages_to_prompt_and_image(messages)
        if pil_image is None:
            return type("R", (), {"content": "Error: No image found in message for vision model."})()

        qwen_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt or "Describe the image."},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        tgt = self.device if self.device != "auto" else next(self.model.parameters()).device
        inputs = inputs.to(tgt)

        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        in_len = inputs["input_ids"].shape[1]
        new_tokens = gen[:, in_len:]
        out = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text_out = (out[0] if out else "").strip()
        return type("R", (), {"content": text_out})()


def _load_qwen2_vl_vision_chat() -> Qwen2VLVisionChat:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_id = os.environ.get("VISION_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct").strip()
    max_new = int(os.environ.get("VISION_MAX_NEW_TOKENS", "512"))
    force_cpu = os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes")
    use_cuda = torch.cuda.is_available() and not force_cpu

    print(f"[LLM] Loading vision model {model_id!r} (first run downloads weights)...")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if use_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        device = str(next(model.parameters()).device)
    else:
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = model.to("cpu")
        device = "cpu"

    print(f"[LLM] Qwen2-VL ready (device={device})")
    return Qwen2VLVisionChat(model, processor, device, max_new_tokens=max_new)


def get_vision_llm():
    """
    Local Qwen2-VL-Instruct (default 2B) for image understanding.

    Expects ``invoke([HumanMessage])`` with OpenAI-style content: text + data:image/...;base64,...

    Env:
      VISION_MODEL_ID — HuggingFace id (default ``Qwen/Qwen2-VL-2B-Instruct``)
      VISION_MAX_NEW_TOKENS — default 512
    """
    global _vision_llm

    if _vision_llm is not None:
        return _vision_llm

    try:
        _vision_llm = _load_qwen2_vl_vision_chat()
    except ImportError as e:
        print(
            "[LLM] Vision model needs: pip install qwen-vl-utils transformers>=4.45 torch accelerate"
        )
        raise ImportError(f"Vision LLM import failed: {e}") from e

    return _vision_llm


def get_text_llm():
    """Alias for get_local_llm()."""
    return get_local_llm()


def reset_llm_cache():
    """Clear cached LLM / vision model instances."""
    global _local_llm, _local_pipeline, _vision_llm
    _local_llm = None
    _local_pipeline = None
    _vision_llm = None
    print("[LLM] LLM cache cleared")


def generate_text(prompt: str, max_tokens: int = 512) -> str:
    """Direct text generation with local Qwen3 pipeline."""
    pipe = get_local_pipeline()
    result = pipe(prompt, max_new_tokens=max_tokens)
    return result[0]["generated_text"]
