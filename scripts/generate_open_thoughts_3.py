import json, os
from typing import Iterable, Dict, Any, List, Optional

from openai import OpenAI

from spider.client import SpiderClient
from spider.config import AppConfig

# == sample post processor ==

LANG_MARKER = "```python"

def _extract_code_block(text):
    lower_text = text.lower()
    start_marker = lower_text.rfind(LANG_MARKER)
    if start_marker == -1:
        return None
    code_start = start_marker + len(LANG_MARKER)
    closing_marker = text.find("```", code_start)
    if closing_marker == -1:
        return None
    snippet = text[code_start:closing_marker].lstrip("\r\n").rstrip()
    return snippet

def filter_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    params:
    -- row (dict): a single rollout record with a "completion" field (str)

    return:
    -- enriched record (dict): the updated record with arbitrary fields added / edited
    -- None: if the record is unwanted
    """
    completion = row.get("completion")
    if not isinstance(completion, str):
        return None
    snippet = _extract_code_block(completion)
    if snippet is None:
        return None
    enriched = dict(row)
    enriched["code"] = snippet  
    return enriched

# == sample pre processor == 

PROMPT_GPT_DIFFICULTY = """You will be given a code problem. Your job is to grade the difficulty level from 1–10 according to the ICPC standard.
Here is the standard:
A 10-point scale for ICPC problems could be structured as follows, where level 1 represents the easiest problems and level 10 represents the most challenging:
Level 1: Basic implementation problems requiring simple input/output handling and straightforward calculations. Typically solvable with a single loop or basic conditional statements. Examples include summing numbers or finding the maximum in an array.
Level 2: Problems involving basic data structures like arrays and strings, requiring simple algorithms like linear search or basic sorting. May include simple mathematical concepts like prime numbers or basic geometry.
Level 3: Problems requiring knowledge of standard algorithms like binary search, complete sorting algorithms, or basic graph traversal (DFS/BFS). May include simple dynamic programming problems with clear state transitions.
Level 4: Problems combining multiple basic concepts, requiring careful implementation and moderate optimization. Includes medium-difficulty dynamic programming problems and basic graph algorithms like shortest paths.
Level 5: Problems requiring solid understanding of data structures like segment trees, binary indexed trees, or disjoint set unions. May include more complex graph algorithms like minimum spanning trees or network flow basics.
Level 6: Advanced dynamic programming problems with non-obvious state representations. Problems requiring combination of multiple algorithms or data structures. May include basic game theory or basic number theory concepts.
Level 7: Problems requiring advanced algorithmic knowledge like heavy-light decomposition, suffix arrays, or advanced geometric algorithms. Includes complex optimization problems and harder network flow applications.
Level 8: Problems requiring deep mathematical insights combined with complex algorithmic implementations. May include advanced number theory, complex geometric algorithms, or problems requiring multiple non-obvious observations.
Level 9: Problems requiring extensive knowledge of advanced algorithms and mathematical concepts, often needing multiple key insights to solve. May include advanced string algorithms like suffix automata, or complex mathematical optimizations.
Level 10: The most challenging problems, often requiring novel approaches or insights not covered in standard competitive programming material. These problems might combine multiple advanced concepts in non-obvious ways, require complex proofs for correctness, or need highly optimized implementations to meet strict time limits.
This scale corresponds roughly to the difficulty progression you might see from early regional contests (levels 1–4) through regional finals (levels 4–7) to world finals problems (levels 7–10).
Problem to be labeled: {question}."""

DIFFICULTY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "difficulty",
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Difficulty score from 1 to 10"
                }
            },
            "required": ["score"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

PROMPT_TEMPLATE = """You are a helpful programmer. You are given a programming question below.
Question: {prompt}

First reason through the problem. Then provide your final code in backticks. 
"""

_gpt_client = None

def get_client():
    global _gpt_client
    if _gpt_client is None:
        _gpt_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _gpt_client

def judge_difficulty(client, question):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": PROMPT_GPT_DIFFICULTY.format(question=question)},
        ],
        response_format=DIFFICULTY_SCHEMA,
    )
    msg = resp.choices[0].message.content
    return int(json.loads(msg)["score"])

def build_prompt(row: Dict[str, Any]):
    """
    param: row (dict): a single prompt record

    return:
    -- prompt (str): the transformed prompt to be sent to the rollout model
    -- None: if the row is unwanted
    """
    client = get_client()
    question = row.get("input")
    difficulty = judge_difficulty(client, question)
    if difficulty < 5:
        return None
    return PROMPT_TEMPLATE.format(prompt=question)

# == main function for client call ==

def main() -> None:
    config = AppConfig.load("config/open_thoughts_3.yaml")
    secrets = {"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]}
    with SpiderClient(
        config=config, 
        env=secrets,
        pre_processor=build_prompt,
        post_processor=filter_row
    ) as client:
        submission = client.submit_job()
        job_id = submission["job_id"]
        print(f"Job submitted: {job_id}")

        status = client.poll_job(job_id, interval=5.0, timeout=600, wait_for_completion=True)
        print(f"Final status: {status['status']}")

        if status["status"] == "completed":
            artifact_path = client.download_result(job_id, destination="artifacts/test-remote.json")
            print(f"Artifacts saved to {artifact_path}")
        else:
            print("Messages: ", status.get("messages", []))
            if status.get("error"):
                print("Error: ", status["error"])

if __name__ == "__main__":
    main()