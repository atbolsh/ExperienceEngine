"""
Nous-format agent for Qwen3.

Bypasses LangChain's agent framework entirely.  Uses Qwen3's native chat
template (apply_chat_template with tools= parameter) and the <tool_call> /
<tool_response> XML protocol that the model was actually trained on.

The entire assistant turn is one continuous token stream.  When the model emits
</tool_call>, generation is suspended, the tool is run, <tool_response>...</tool_response>
tokens are spliced into the sequence, and generation resumes from there.
The model sees tool results as part of its own reasoning — no message-role switching.

Thinking (<think>...</think>) is left enabled and printed in full.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from active_environment import get_active_environment_name
from tools import create_tools, initialize_vector_store_manager, initialize_env
from tools.memory_schema import list_memory_folders
from llm_utils import get_pipeline_and_tokenizer

_VERBOSE = None


def _is_verbose() -> bool:
    global _VERBOSE
    if _VERBOSE is None:
        _VERBOSE = os.environ.get("AGENT_VERBOSE", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
    return _VERBOSE


# ── regex ────────────────────────────────────────────────────────────
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TRAILING_THINK_RE = re.compile(r"<think>(.*)$", re.DOTALL)


# ── StoppingCriteria ─────────────────────────────────────────────────

def _make_tool_call_stopper(tokenizer, prompt_len: int):
    """Return a StoppingCriteriaList that halts generation at </tool_call>."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopAtToolCallEnd(StoppingCriteria):
        def __init__(self, tok, p_len):
            super().__init__()
            self.tok = tok
            self.p_len = p_len

        def __call__(self, input_ids, scores, **kwargs):
            new_ids = input_ids[0][self.p_len :]
            if new_ids.shape[0] < 4:
                return False
            tail = self.tok.decode(new_ids[-40:], skip_special_tokens=True)
            return "</tool_call>" in tail

    return StoppingCriteriaList([_StopAtToolCallEnd(tokenizer, prompt_len)])


# ── prompt / blurb helpers ───────────────────────────────────────────

def _load_prompt_text() -> str:
    path = Path(__file__).resolve().parent / "prompts" / "global_prompt.txt"
    return path.read_text(encoding="utf-8")


def _load_environment_blurb() -> str:
    env_name = get_active_environment_name()
    name_to_file = {
        "car_environment": "car_blurb.txt",
        "game_environment": "game_blurb.txt",
    }
    blurb_file = name_to_file.get(env_name, "game_blurb.txt")
    blurb_path = Path(__file__).resolve().parent / "prompts" / blurb_file
    try:
        text = blurb_path.read_text(encoding="utf-8")
        print(f"[NousAgent] Loaded environment blurb: {blurb_file}")
        return text
    except FileNotFoundError:
        print(f"Warning: Blurb file not found: {blurb_path}")
        return ""


def _load_optional_text(path: Path, label: str) -> str:
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                print(f"[NousAgent] Loaded {label}")
                return text
    except Exception as e:
        print(f"Warning loading {label}: {e}")
    return ""


def _build_system_prompt() -> str:
    prompt_text = _load_prompt_text()
    env_blurb = _load_environment_blurb()

    formatted = prompt_text.format(
        cwd=os.getcwd(),
        environment_blurb=env_blurb,
    )

    ctx = _load_optional_text(
        Path(__file__).parent / "prompts" / "context_prompt.md",
        "context_prompt.md",
    )
    mem = _load_recent_episodic()

    extras: list[str] = []
    if ctx:
        extras.append(f"Learned context:\n{ctx}")
    if mem:
        extras.append(f"Previous session:\n{mem}")
    if extras:
        formatted += "\n\n" + "\n\n".join(extras)

    return formatted


def _load_recent_episodic() -> str:
    try:
        ep = Path(__file__).parent / "episodic"
        if not ep.exists():
            return ""
        memories = list_memory_folders(ep)
        if not memories:
            return ""
        most_recent = max(memories, key=lambda m: m.folder_path.stat().st_mtime)
        print(f"[NousAgent] Loaded episodic memory: {most_recent.folder_name}")
        result = f"=== RECENT EPISODIC MEMORY ===\n{most_recent.folder_name}\n\n"
        result += most_recent.text_content
        if most_recent.image_files:
            result += f"\n({len(most_recent.image_files)} images: {', '.join(most_recent.image_names)})"
        result += "\n=== END ===\n"
        return result
    except Exception as e:
        print(f"Warning: episodic memory: {e}")
        return ""


# ── tool schema conversion ───────────────────────────────────────────

def _tool_to_schema(tool) -> dict:
    """Convert a LangChain Tool / StructuredTool to the JSON-schema dict
    expected by Qwen3's apply_chat_template(tools=...)."""
    schema: dict = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (tool.description or "").strip(),
            "parameters": {"type": "object", "properties": {}},
        },
    }
    if tool.args_schema is not None:
        try:
            raw = tool.args_schema.schema()
            params: dict = {"type": "object"}
            if "properties" in raw:
                params["properties"] = raw["properties"]
            if "required" in raw:
                params["required"] = raw["required"]
            schema["function"]["parameters"] = params
        except Exception:
            pass
    return schema


# ── chat history wrapper ─────────────────────────────────────────────

class ChatHistory:
    """Minimal chat history with .clear() for main.py compat."""

    def __init__(self):
        self.messages: List[Dict[str, str]] = []

    def clear(self):
        self.messages.clear()


# ── agent ─────────────────────────────────────────────────────────────

class NousAgent:
    """
    Agentic loop using Qwen3's native <tool_call> / <tool_response> protocol.

    The entire assistant reply is a single continuous generation.  When the
    model emits </tool_call>, we pause, run the tool, splice
    <tool_response>…</tool_response> tokens into the sequence, and resume
    model.generate() from there.  No message-role switching — the model sees
    tool results as part of its own stream.

    Public interface:
        agent.invoke({"input": "..."}, config=...) -> {"output": "..."}
    """

    MAX_NEW_TOKENS = int(os.environ.get("AGENT_MAX_NEW_TOKENS", "8192"))

    def __init__(
        self,
        system_prompt: str,
        tools: list,
        pipeline,
        tokenizer,
        max_tool_rounds: int = 15,
    ):
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_schemas = [_tool_to_schema(t) for t in tools]
        self.tool_map = {t.name: t for t in tools}
        self.pipeline = pipeline
        self.model = pipeline.model
        self.tokenizer = tokenizer
        self.max_tool_rounds = max_tool_rounds
        self.chat_history = ChatHistory()
        self._device = next(self.model.parameters()).device

    # ── parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_tool_calls(text: str) -> List[Tuple[str, dict]]:
        results = []
        for m in _TOOL_CALL_RE.finditer(text):
            try:
                obj = json.loads(m.group(1))
                name = obj.get("name", "")
                args = obj.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"input": args}
                results.append((name, args))
            except json.JSONDecodeError:
                print(f"  [parse] Malformed tool_call JSON: {m.group(1)[:200]}")
        return results

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Strip thinking, tool_call, and tool_response blocks → visible answer."""
        text = _TOOL_CALL_RE.sub("", text)
        text = _TOOL_RESPONSE_RE.sub("", text)
        text = _THINK_RE.sub("", text)
        text = _TRAILING_THINK_RE.sub("", text)
        return text.strip()

    # ── tool execution ───────────────────────────────────────────────

    def _run_tool(self, name: str, args: dict) -> str:
        tool = self.tool_map.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self.tool_map)}"
        try:
            if args:
                return str(tool.invoke(args))
            else:
                return str(tool.invoke(""))
        except Exception as e:
            return f"Error running {name}: {e}"

    # ── main loop ────────────────────────────────────────────────────

    def invoke(
        self,
        input_dict: dict,
        config: Optional[dict] = None,
    ) -> dict:
        import torch

        user_input = input_dict["input"]
        verbose = _is_verbose()

        # Build the initial prompt via chat template
        messages: List[dict] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.chat_history.messages)
        messages.append({"role": "user", "content": user_input})

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            enable_thinking=True,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize once — this is our running token sequence
        seq = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(
            self._device
        )
        prompt_len = seq.shape[1]

        truncated = False

        for tool_round in range(self.max_tool_rounds):
            # Generate from current sequence, stop at </tool_call> or EOS
            gen_start = seq.shape[1]
            stop = _make_tool_call_stopper(self.tokenizer, gen_start)

            with torch.inference_mode():
                out = self.model.generate(
                    input_ids=seq,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    repetition_penalty=1.1,
                    stopping_criteria=stop,
                )

            n_new = out.shape[1] - gen_start
            truncated = n_new >= self.MAX_NEW_TOKENS
            new_text = self.tokenizer.decode(
                out[0][gen_start:], skip_special_tokens=True
            )

            if verbose:
                print(new_text, end="", flush=True)
                if truncated:
                    print(
                        f"\n*** TRUNCATED at {self.MAX_NEW_TOKENS} tokens — "
                        f"set AGENT_MAX_NEW_TOKENS higher ***"
                    )

            # Check for tool calls in the newly generated chunk
            tool_calls = self._parse_tool_calls(new_text)

            if not tool_calls:
                # No tool call — generation is finished.
                break

            # Splice tool responses into the token stream
            seq = out
            for name, args in tool_calls:
                result = self._run_tool(name, args)
                injection = f"\n<tool_response>\n{result}\n</tool_response>\n"

                if verbose:
                    print(injection, end="", flush=True)

                inj_ids = self.tokenizer(
                    injection, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(self._device)
                seq = torch.cat([seq, inj_ids], dim=1)

        # Everything after the original prompt is the assistant's turn:
        # <think>…<tool_call>…</tool_call><tool_response>…</tool_response>…</think> answer
        full_assistant = self.tokenizer.decode(
            seq[0][prompt_len:], skip_special_tokens=True
        )

        if verbose:
            print()  # newline after streaming output

        answer = self._extract_answer(full_assistant)
        if not answer:
            answer = "(Model produced only internal reasoning with no visible answer.)"
        if truncated:
            answer += (
                f"\n\n[generation truncated at {self.MAX_NEW_TOKENS} tokens"
                " — set AGENT_MAX_NEW_TOKENS higher]"
            )

        self.chat_history.messages.append({"role": "user", "content": user_input})
        self.chat_history.messages.append({"role": "assistant", "content": answer})
        return {"output": answer}


# ── factory ──────────────────────────────────────────────────────────

def create_conversational_agent():
    """Create a NousAgent with tools, memory, and environment.

    Returns (agent, chat_history) — same interface as langchain_agent.
    """
    print("Initializing vector stores...")
    initialize_vector_store_manager()
    initialize_env()

    pipeline, tokenizer = get_pipeline_and_tokenizer()
    tools = create_tools()
    system_prompt = _build_system_prompt()

    agent = NousAgent(
        system_prompt=system_prompt,
        tools=tools,
        pipeline=pipeline,
        tokenizer=tokenizer,
    )

    print(f"[NousAgent] Ready ({len(tools)} tools, thinking=enabled)")
    return agent, agent.chat_history
