"""
Output parser for Qwen3 + ReAct.

Qwen3 emits <think>...</think> blocks before its visible reply.
The stock ReActSingleInputOutputParser sees the angle brackets as
parseable content and chokes. This parser:

  1. Strips <think>…</think> (greedy, possibly multi-block).
  2. Strips any stray Observation: ... blocks the model hallucinated
     (only the system should append observations).
  3. Delegates the cleaned text to the normal ReAct regex parser.
"""

from __future__ import annotations

import re
from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

try:
    from langchain.agents.agent import AgentOutputParser
except ImportError:
    from langchain_classic.agents.agent import AgentOutputParser

# ── regex constants ──────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TRAILING_THINK_RE = re.compile(r"<think>.*", re.DOTALL)

_ACTION_RE = re.compile(
    r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)",
    re.DOTALL,
)
_FINAL_ANSWER = "Final Answer:"

_HALLUCINATED_OBS_RE = re.compile(
    r"\n*Observation\s*:.*",
    re.DOTALL,
)


class Qwen3ReActOutputParser(AgentOutputParser):
    """Parse ReAct output, tolerating Qwen3 <think> blocks."""

    def get_format_instructions(self) -> str:
        return (
            "Thought: your reasoning\n"
            "Action: tool_name\n"
            "Action Input: single line\n"
            "— OR —\n"
            "Thought: your reasoning\n"
            "Final Answer: reply to the user\n"
        )

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _strip_thinking(text: str) -> str:
        text = _THINK_RE.sub("", text)
        text = _TRAILING_THINK_RE.sub("", text)
        return text.strip()

    @staticmethod
    def _strip_hallucinated_observations(text: str) -> str:
        return _HALLUCINATED_OBS_RE.sub("", text).strip()

    # ── main parse ───────────────────────────────────────────────────
    def parse(self, raw: str) -> Union[AgentAction, AgentFinish]:
        text = self._strip_thinking(raw)
        text = self._strip_hallucinated_observations(text)

        if not text:
            raise OutputParserException(
                "LLM produced only <think> content with no action or answer.",
                observation="You must output either:\nThought: ...\nAction: <tool>\nAction Input: <input>\n\nor:\nThought: ...\nFinal Answer: <reply>",
                llm_output=raw,
                send_to_llm=True,
            )

        action_match = _ACTION_RE.search(text)
        has_final = _FINAL_ANSWER in text

        if action_match and has_final:
            # Both present — take the action (model can answer after tool result).
            pass

        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip().strip('"')
            return AgentAction(action, action_input, raw)

        if has_final:
            answer = text.split(_FINAL_ANSWER)[-1].strip()
            return AgentFinish({"output": answer}, raw)

        # Neither found — send a corrective hint back to the model.
        raise OutputParserException(
            f"Could not parse LLM output: `{text}`",
            observation=(
                "Format error. You MUST use exactly one of:\n"
                "Thought: ...\nAction: <tool_name>\nAction Input: <input>\n\n"
                "or:\nThought: ...\nFinal Answer: <your answer>"
            ),
            llm_output=raw,
            send_to_llm=True,
        )

    @property
    def _type(self) -> str:
        return "qwen3-react"
