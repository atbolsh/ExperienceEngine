"""
Nous-format agent for Qwen3.

Bypasses LangChain's agent framework entirely.  Uses Qwen3's native chat
template (apply_chat_template with tools= parameter) and the <tool_call> /
<tool_response> XML protocol that the model was actually trained on.

Generation stops at </tool_call> via a custom StoppingCriteria, the tool is
executed, the result is inserted as a {"role": "tool"} message, the prompt is
re-tokenized, and generation resumes.  This avoids the hallucinated-observation
problem that plagues generate-then-parse loops.

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
            # Only decode new tokens (skip the prompt).
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

    Generation uses model.generate() directly (not the pipeline wrapper) so we
    can inject a StoppingCriteria that halts at </tool_call>.

    Public interface mirrors the LangChain path so main.py needs no changes:
        agent.invoke({"input": "..."}, config={"configurable": {"session_id": "..."}})
        -> {"output": "..."}
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list,
        pipeline,
        tokenizer,
        max_iterations: int = 30,
    ):
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_schemas = [_tool_to_schema(t) for t in tools]
        self.tool_map = {t.name: t for t in tools}
        self.pipeline = pipeline
        self.model = pipeline.model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.chat_history = ChatHistory()
        self._device = next(self.model.parameters()).device

    MAX_NEW_TOKENS = int(os.environ.get("AGENT_MAX_NEW_TOKENS", "8192"))

    # ── generation ───────────────────────────────────────────────────

    def _generate(self, messages: List[dict]) -> Tuple[str, bool]:
        """Build prompt via apply_chat_template, generate with stop-at-tool_call.

        Returns (text, truncated) where truncated is True if generation hit
        max_new_tokens without a natural stop.
        """
        import torch

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            enable_thinking=True,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        stop = _make_tool_call_stopper(self.tokenizer, prompt_len)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.1,
                stopping_criteria=stop,
            )

        new_tokens = output_ids[0][prompt_len:]
        n_generated = new_tokens.shape[0]
        truncated = n_generated >= self.MAX_NEW_TOKENS
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, truncated

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
        """Strip <think> blocks and <tool_call> blocks, return visible answer."""
        text = _TOOL_CALL_RE.sub("", text)
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
        user_input = input_dict["input"]
        verbose = _is_verbose()

        messages: List[dict] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.chat_history.messages)
        messages.append({"role": "user", "content": user_input})

        for iteration in range(self.max_iterations):
            raw, truncated = self._generate(messages)

            if verbose:
                print(f"\n--- generation {iteration} ---\n{raw}\n--- end ---")
                if truncated:
                    print(
                        f"  *** TRUNCATED: hit max_new_tokens={self.MAX_NEW_TOKENS}. "
                        f"Increase with AGENT_MAX_NEW_TOKENS env var. ***"
                    )
                print()

            tool_calls = self._parse_tool_calls(raw)

            if not tool_calls:
                answer = self._extract_answer(raw)
                if not answer:
                    answer = "(Model produced only internal reasoning with no visible answer.)"
                if truncated:
                    answer += (
                        f"\n\n[generation truncated at {self.MAX_NEW_TOKENS} tokens"
                        " — set AGENT_MAX_NEW_TOKENS higher]"
                    )

                self.chat_history.messages.append(
                    {"role": "user", "content": user_input}
                )
                self.chat_history.messages.append(
                    {"role": "assistant", "content": answer}
                )
                return {"output": answer}

            # Append the assistant turn (with tool_call tags intact so
            # apply_chat_template can format it properly on the next call).
            messages.append({"role": "assistant", "content": raw})

            for name, args in tool_calls:
                result = self._run_tool(name, args)

                if verbose:
                    print(f"  <tool_call> {name}({json.dumps(args, ensure_ascii=False)})")
                    print(f"  <tool_response>\n  {result}\n  </tool_response>\n")

                messages.append({"role": "tool", "content": result, "name": name})

        answer = "Reached maximum tool-calling iterations."
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
