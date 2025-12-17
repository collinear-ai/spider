import os
from typing import Any, Dict, List, Optional

from spider.client import SpiderClient
from spider.config import AppConfig, RuntimeDependencyConfig

TAVILY_ENDPOINT = "https://api.tavily.com/search"

def web_search(query, max_results):
    import httpx
    api_key = os.environ["TAVILY_API_KEY"]
    payload = {
        "query": query,
        "max_results": max(1, min(max_results, 10)),
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

def fetch_page(url, max_chars):
    import httpx
    from bs4 import BeautifulSoup
    
    with httpx.Client(timeout=20.0) as client:
        response = client.get(url, follow_redirects=True)
        response.raise_for_status()
        
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    snippet = text[: max_chars]
    return {
        "url": url,
        "content": snippet,
    }

def _search_schema():
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query."},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
                "description": "Number of search hits to return (1-10)."
            }
        },
        "required": ["query"]
    }

def _fetch_schema():
    return {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "HTTP/HTTPS URL to download."},
            "max_chars": {
                "type": "integer",
                "minimum": 200,
                "maximum": 8000,
                "default": 2000,
                "description": "Maximum characters of cleaned text to return."
            }
        },
        "required": ["url"]
    }

def main():
    config = AppConfig.load("config/generate_gaia_search.yaml")
    runtime = config.job.runtime or RuntimeDependencyConfig()
    runtime.packages = ["httpx", "beautifulsoup4"]
    config.job.runtime = runtime

    secrets = {
        "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
        "HF_TOKEN": os.environ["HF_TOKEN"]
    }

    with SpiderClient(config=config, env=secrets) as client:
        client.add_tool(
            description="Run a web search and return relevant results.",
            json_schema=_search_schema(),
            func=web_search,
        )
        client.add_tool(
            description="Download a web page and return a cleaned text snippet.",
            json_schema=_fetch_schema(),
            func=fetch_page,
        )

        submission = client.submit_job()
        status = client.poll_job(submission["job_id"], interval=5.0, wait_for_completion=True)

        if status["status"] == "completed":
            client.download_result(submission["job_id"], destination="artifacts/generate_gaia_search.json")
        else:
            raise RuntimeError(status.get("error") or status.get("messages"))

if __name__ == "__main__":
    main()