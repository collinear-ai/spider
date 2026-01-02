from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager

@dataclass
class ParsedTurn:
    content: str
    reasoning: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    parser_fallback: bool

def parse_assistant_turn(
    *,
    messages: Sequence[Dict[str, Any]],
    assistant_text: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Any = None,
    tool_parser_name: Optional[str] = None,
    reasoning_parser_name: Optional[str] = None,
    tokenizer: Any = None,
    token_ids: Optional[Sequence[int]] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> ParsedTurn:
    parser_fallback = False
    content = assistant_text or ""
    reasoning = None
    tool_calls = None

    request = ChatCompletionRequest(
        model="local",
        messages=list(messages),
        tools=tools,
        tool_choice=tool_choice,
    )

    if reasoning_parser_name and tokenizer is not None:
        try:
            parser_cls = ReasoningParserManager.get_reasoning_parser(
                reasoning_parser_name,
            )
            parser = parser_cls(tokenizer, chat_template_kwargs=chat_template_kwargs)
            reasoning, content = parser.extract_reasoning(content, request=request)
        except Exception:
            parser_fallback = True
    elif reasoning_parser_name:
        parser_fallback = True

    if tool_parser_name and tools and tool_choice not in ("none", None):
        if tokenizer is None:
            parser_fallback = True
        elif tool_parser_name == "openai" and token_ids is None:
            parser_fallback = True
        else:
            try:
                parser_cls = ToolParserManager.get_tool_parser(tool_parser_name)
                parser = parser_cls(tokenizer)
                try:
                    info = parser.extract_tool_calls(
                        content,
                        request=request,
                        token_ids=token_ids,
                    )
                except TypeError:
                    info = parser.extract_tool_calls(content, request=request)
                tool_calls = list(info.tool_calls) if info.tool_calls else None
                if info.content is not None:
                    content = info.content
            except Exception:
                parser_fallback = True

    return ParsedTurn(
        content=content or "",
        reasoning=reasoning,
        tool_calls=tool_calls,
        parser_fallback=parser_fallback,
    )