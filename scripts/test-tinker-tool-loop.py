import json, os
import logging
logging.basicConfig(level=logging.INFO)

import tinker

from server.executor import _run_prompt_with_tools
from spider.config import GenerationConfig, JobConfig, ModelConfig, OutputConfig, SourceConfig
from scripts.web_search import _fetch_schema, _search_schema, fetch_page, web_search

def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Run a web search and return relevant results.",
                "parameters": _search_schema(),
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_page",
                "description": "Download a web page and return a cleaned text snippet.",
                "parameters": _fetch_schema(),
            }
        }
    ]

def _job_config(model_name, system_prompt):
    return JobConfig(
        model=ModelConfig(
            provider="tinker",
            name=model_name,
            parameters={
                "system_prompt": system_prompt or ""
            }
        ),
        source=SourceConfig(dataset="demo", split="train", field=None),
        generation=GenerationConfig(
            max_turns=4,
            parameters={
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.95,
            }
        ),
        output=OutputConfig(),
    )

def main():
    base_model = "Qwen/Qwen3-8B"
    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(
        base_model=base_model
    )
    
    job = _job_config(
        base_model,
        system_prompt="You are a web search assistant. You MUST use tools and cite sources and never rely on memory."
    )
    tool_registry = {
        "web_search": web_search,
        "fetch_page": fetch_page,
    }

    transcript, token_ids, logprobs, reward_mask = _run_prompt_with_tools(
        backend=sampling_client,
        job=job,
        prompt="In the video https://www.youtube.com/watch?v=L1vXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?",
        tool_defs=_tool_defs(),
        tool_registry=tool_registry,
        turn_limit=job.generation.max_turns,
        job_id="test-tinker-tool-loop",
        include_logprobs=True,
    )

    print("=== Transcript ===")
    for msg in transcript:
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        print(f"{role}: {content[:40].replace('\n', '\\n')}...")
        if tool_calls:
            print("  tool_calls:", json.dumps(tool_calls, indent=2))

    print("\n=== Logprobs ===")
    print(f"Total tokens: {len(token_ids)}")
    print(f"Total logprobs: {len(logprobs)} (first 5) {logprobs[:5]}")
    print(f"Reward mask positives: {sum(reward_mask)} / {len(reward_mask)}")

if __name__ == "__main__":
    main()