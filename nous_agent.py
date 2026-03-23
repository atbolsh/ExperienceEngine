"""
Nous-format agent for Qwen3.

Bypasses LangChain's agent framework entirely.  Uses Qwen3's native chat
template (apply_chat_template with tools= parameter) and the <tool_call> /
<tool_response> XML protocol that the model was actually trained on.

Thinking (<think>…</think>) is left enabled — the model's internal reasoning
improves output quality and is printed when AGENT_VERBOSE=1 (default).
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
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
        _VERBOSE = os.environ.get("AGENT_VERBOSE", "1").strip().lower() not in ("0", "false", "no")
    return _VERBOSE

# ── regex for parsing model output ───────────────────────────────────
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TRAILING_THINK_RE = re.compile(r"<think>(.*)$", re.DOTALL)


# ── helpers shared with langchain_agent.py ───────────────────────────
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


# ── chat history wrapper (exposes .clear() for main.py compat) ───────

class ChatHistory:
    """Minimal chat history with the same .clear() API that main.py expects."""

    def __init__(self):
        self.messages: List[Dict[str, str]] = []

    def clear(self):
        self.messages.clear()


# ── agent ─────────────────────────────────────────────────────────────

class NousAgent:
    """
    Agentic loop using Qwen3's native <tool_call> / <tool_response> protocol.

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
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.chat_history = ChatHistory()

    # ── generation ───────────────────────────────────────────────────

    def _generate(self, messages: List[dict]) -> str:
        """Format messages with the Qwen3 chat template and generate."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            enable_thinking=True,
            add_generation_prompt=True,
            tokenize=False,
        )
        outputs = self.pipeline(prompt, return_full_text=False)
        return outputs[0]["generated_text"]

    # ── parsing ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_thinking(text: str) -> Tuple[str, str]:
        """Return (thinking_content, remainder_after_stripping_think_blocks)."""
        thinks: list[str] = []
        for m in _THINK_RE.finditer(text):
            thinks.append(m.group(1).strip())
        cleaned = _THINK_RE.sub("", text)
        # Handle unclosed trailing <think>
        trail = _TRAILING_THINK_RE.search(cleaned)
        if trail:
            thinks.append(trail.group(1).strip())
            cleaned = _TRAILING_THINK_RE.sub("", cleaned)
        return "\n".join(thinks), cleaned.strip()

    @staticmethod
    def _parse_tool_calls(text: str) -> List[Tuple[str, dict]]:
        """Extract all <tool_call> blocks, return list of (name, args)."""
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
                if _is_verbose():
                    print(f"  [parse] Malformed tool_call JSON: {m.group(1)[:120]}")
        return results

    @staticmethod
    def _strip_tool_calls(text: str) -> str:
        """Remove <tool_call> blocks, leaving only the plain-text answer."""
        return _TOOL_CALL_RE.sub("", text).strip()

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

        # Build message list: system + history + new user turn
        messages: List[dict] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.chat_history.messages)
        messages.append({"role": "user", "content": user_input})

        for iteration in range(self.max_iterations):
            raw = self._generate(messages)
            thinking, visible = self._extract_thinking(raw)

            if verbose and thinking:
                print(f"  <think> {thinking[:500]}{'...' if len(thinking) > 500 else ''} </think>")

            tool_calls = self._parse_tool_calls(visible)

            if not tool_calls:
                # No tool call — this is the final answer.
                answer = self._strip_tool_calls(visible).strip()
                if not answer and thinking:
                    answer = "(The model produced only internal reasoning with no visible answer.)"

                # Persist in chat history
                self.chat_history.messages.append({"role": "user", "content": user_input})
                self.chat_history.messages.append({"role": "assistant", "content": answer})
                return {"output": answer}

            # Append the full assistant turn (including tool_call tags) for context
            messages.append({"role": "assistant", "content": raw})

            for name, args in tool_calls:
                if verbose:
                    print(f"  > Tool: {name}({json.dumps(args, ensure_ascii=False)[:200]})")

                result = self._run_tool(name, args)

                if verbose:
                    print(f"  < Result: {result[:300]}{'...' if len(result) > 300 else ''}")

                messages.append({"role": "tool", "content": result, "name": name})

        # Exhausted iterations
        answer = "I reached the maximum number of tool-calling steps. Here is what I have so far."
        self.chat_history.messages.append({"role": "user", "content": user_input})
        self.chat_history.messages.append({"role": "assistant", "content": answer})
        return {"output": answer}


# ── factory (matches langchain_agent.py's create_conversational_agent) ─

def create_conversational_agent():
    """Create a NousAgent with tools, memory, and environment.

    Returns (agent, chat_history) with the same interface as langchain_agent.
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
