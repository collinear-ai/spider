import os
import json
from typing import Any, Dict, List, Optional

from spider.client import SpiderClient
from spider.config import AppConfig, RuntimeDependencyConfig

# Every tool def and processor def here is identical to scripts/generate_tool_search_nemo.py

TAVILY_ENDPOINT = "https://api.tavily.com/search"

def search(query):
    import httpx
    api_key = os.environ["TAVILY_API_KEY"]
    payload = {
        "query": query,
        "max_results": 10,
        "include_images": False,
        "include_answer": False,
    }

    with httpx.Client(timeout=15.0) as client:
        response = client.post(
            TAVILY_ENDPOINT,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])[: payload["max_results"]]
    return {
        "query": query,
        "results": [
            {
                "title": item.get("title"),
                "snippet": item.get("content"),
                "url": item.get("url"),
            }
            for item in results
        ]
    }

def browse(url):
    import httpx
    from bs4 import BeautifulSoup

    def _jina_proxy(target_url):
        return "https://r.jina.ai/http://" + target_url.replace(
            "https://", ""
        ).replace(
            "http://", ""
        )

    original_url = url
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SpiderBot/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    with httpx.Client(timeout=20.0) as client:
        try:
            response = client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                url = _jina_proxy(original_url)
                response = client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
            else:
                raise
        
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    words = text.split()
    trimmed = " ".join(words[:10000])
    return {
        "url": url,
        "source_url": original_url,
        "content": trimmed,
    }

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Natural language search query."},
    },
    "required": ["query"]
}

BROWSE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "HTTP/HTTPS URL to download."},
    },
    "required": ["url"],
}

def pre_processor(row):
    responses_create_params = row.get("responses_create_params", {})
    instructions = responses_create_params.get("instructions", "")
    input_str = responses_create_params.get("input", "")
    updated = dict(row)
    updated["prompt"] = instructions + "\n\n" + input_str
    return updated

def main():
    config = AppConfig.load("config/train_on_policy_tool_search_nemo.yaml")
    config.job.ensure_runtime().add_packages(
        "httpx",
        "beautifulsoup4",
    )

    with SpiderClient(
        config=config, 
        env=("TAVILY_API_KEY", "HF_TOKEN", "TINKER_API_KEY", "WANDB_API_KEY"), 
        pre_processor=pre_processor,
    ) as client:
        client.add_tool(
            description="Search Google for a query and return up to 10 search results.",
            json_schema=SEARCH_SCHEMA,
            func=search,
        )
        client.add_tool(
            description="Returns the cleaned content of a webpage. If the page is too long, it will be truncated to 10,000 words.",
            json_schema=BROWSE_SCHEMA,
            func=browse,
        )

        submission = client.submit_job()
        status = client.poll_job(submission["job_id"], interval=5.0, wait_for_completion=True)

        if status["status"] == "completed":
            client.download_result(submission["job_id"], destination="artifacts/generate_tool_search.json")
        else:
            raise RuntimeError(status.get("error") or status.get("messages"))

if __name__ == "__main__":
    main()