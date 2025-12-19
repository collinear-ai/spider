import os
from typing import Any, Dict, List, Optional

from spider.client import SpiderClient
from spider.config import AppConfig, RuntimeDependencyConfig

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
    
    with httpx.Client(timeout=20.0) as client:
        response = client.get(url, follow_redirects=True)
        response.raise_for_status()
        
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    words = text.split()
    trimmed = " ".join(words[:10000])
    return {
        "url": url,
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
    return instructions + "\n\n" + input_str

def main():
    config = AppConfig.load("config/generate_tool_search_nemo.yaml")
    runtime = config.job.runtime or RuntimeDependencyConfig()
    runtime.packages = ["httpx", "beautifulsoup4"]
    config.job.runtime = runtime

    secrets = {
        "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_HOME": os.environ["HF_HOME"],
    }

    with SpiderClient(
        config=config, 
        env=secrets, 
        pre_processor=pre_processor
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