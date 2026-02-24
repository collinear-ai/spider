from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class PublicReadonlyMCPServer:
    key: str
    domain: str
    display_name: str
    mcp_url: str
    mcp_url_env: str
    description: str
    transport: str  # "remote_http" | "stdio"
    auth: str  # "none" | "api_key" | "oauth" | "smithery_key"
    tool_count: Optional[int] = None
    stdio_command: Optional[Tuple[str, ...]] = None
    stdio_install: Optional[str] = None
    notes: Optional[str] = None
    default_headers: Mapping[str, str] = field(default_factory=dict)
    sources: Tuple[str, ...] = field(default_factory=tuple)


_DEFAULT_USER_AGENT = "spider-mcp-support/1.0"

PUBLIC_READONLY_MCP_SERVERS: Dict[str, PublicReadonlyMCPServer] = {
    "arxiv-paper-mcp": PublicReadonlyMCPServer(
        key="arxiv-paper-mcp",
        domain="paper_retrieval",
        display_name="arXiv Paper MCP",
        mcp_url="http://127.0.0.1:9802/mcp",
        mcp_url_env="MCP_ARXIV_PAPER_URL",
        description="arXiv paper search/retrieval endpoint (self-hostable/bridgeable).",
        transport="stdio",
        auth="none",
        tool_count=4,
        stdio_command=("uvx", "arxiv-paper-mcp"),
        stdio_install="uvx arxiv-paper-mcp (requires Python >=3.11; uvx manages its own venv)",
        sources=("https://github.com/daheepk/arxiv-paper-mcp",),
    ),
    "clinicaltrials-mcp-server": PublicReadonlyMCPServer(
        key="clinicaltrials-mcp-server",
        domain="clinical_data_retrieval",
        display_name="ClinicalTrials MCP Server",
        mcp_url="http://127.0.0.1:9803/mcp",
        mcp_url_env="MCP_CLINICALTRIALS_URL",
        description="ClinicalTrials public data retrieval endpoint (self-hostable/bridgeable).",
        transport="stdio",
        auth="none",
        tool_count=17,
        stdio_command=("node", "build/index.js"),
        stdio_install="mkdir -p workloads/mcp_support/local_servers && cd workloads/mcp_support/local_servers && (test -d ClinicalTrials-MCP-Server || git clone https://github.com/Augmented-Nature/ClinicalTrials-MCP-Server.git) && cd ClinicalTrials-MCP-Server && npm install && npm run build",
        notes="Must be cloned from source and built; no published npm package.",
        sources=("https://github.com/Augmented-Nature/ClinicalTrials-MCP-Server",),
    ),
    "context7": PublicReadonlyMCPServer(
        key="context7",
        domain="developer_docs_search",
        display_name="Context7",
        mcp_url="https://mcp.context7.com/mcp",
        mcp_url_env="MCP_CONTEXT7_URL",
        description="Library documentation resolution and targeted docs querying.",
        transport="remote_http",
        auth="none",
        tool_count=2,
        sources=("https://context7.com/docs/mcp-server",),
    ),
    "deepwiki": PublicReadonlyMCPServer(
        key="deepwiki",
        domain="codebase_knowledge",
        display_name="DeepWiki",
        mcp_url="https://mcp.deepwiki.com/mcp",
        mcp_url_env="MCP_DEEPWIKI_URL",
        description="Repository-level code and architecture exploration.",
        transport="remote_http",
        auth="none",
        tool_count=3,
        sources=("https://docs.devin.ai/work-with-devin/deepwiki-mcp",),
    ),
    "exa-search": PublicReadonlyMCPServer(
        key="exa-search",
        domain="web_search_research",
        display_name="Exa Search",
        mcp_url="https://mcp.exa.ai/mcp",
        mcp_url_env="MCP_EXA_SEARCH_URL",
        description="Hosted web search/research endpoint; fully functional without auth.",
        transport="remote_http",
        auth="none",
        tool_count=3,
        notes="No auth required for listing or invocation. Validated read-only calls succeed.",
        sources=("https://exa.ai/mcp",),
    ),
    "financialdatasets": PublicReadonlyMCPServer(
        key="financialdatasets",
        domain="financial_data_api",
        display_name="Financial Datasets",
        mcp_url="https://mcp.financialdatasets.ai/api",
        mcp_url_env="MCP_FINANCIAL_DATASETS_URL",
        description="Financial datasets MCP endpoint (API key required).",
        transport="remote_http",
        auth="api_key",
        tool_count=13,
        notes="Requires X-API-Key header; connection fails without it.",
        default_headers={"X-API-Key": ""},
        sources=("https://mcp.financialdatasets.ai/api",),
    ),
    "google-maps": PublicReadonlyMCPServer(
        key="google-maps",
        domain="geo_routing_places",
        display_name="Google Maps",
        mcp_url="https://mcp.open-mcp.org/api/server/google-maps@latest/mcp",
        mcp_url_env="MCP_GOOGLE_MAPS_URL",
        description="Geocoding, directions, places, distance matrix, and street view APIs.",
        transport="remote_http",
        auth="api_key",
        tool_count=18,
        notes="Tool listing accessible without auth; invocation requires KEY env var (Google Maps API key).",
        sources=("https://mcp.open-mcp.org/server/google-maps",),
    ),
    "leetcode": PublicReadonlyMCPServer(
        key="leetcode",
        domain="coding_problem_retrieval",
        display_name="LeetCode MCP",
        mcp_url="http://127.0.0.1:9805/mcp",
        mcp_url_env="MCP_LEETCODE_URL",
        description="LeetCode problem/discussion endpoint (self-hostable/bridgeable).",
        transport="stdio",
        auth="none",
        tool_count=9,
        stdio_command=("npx", "-y", "@jinzcdev/leetcode-mcp-server", "--site", "global"),
        stdio_install="npx -y @jinzcdev/leetcode-mcp-server",
        sources=("https://github.com/jinzcdev/leetcode-mcp-server",),
    ),
    "open-weather": PublicReadonlyMCPServer(
        key="open-weather",
        domain="weather_global",
        display_name="OpenWeather",
        mcp_url="https://mcp.open-mcp.org/api/server/open-weather@latest/mcp",
        mcp_url_env="MCP_OPEN_WEATHER_URL",
        description="Global weather lookup using public OpenWeather-backed APIs.",
        transport="remote_http",
        auth="none",
        tool_count=2,
        sources=("https://mcp.open-mcp.org/server/open-weather",),
    ),
    "pubmed": PublicReadonlyMCPServer(
        key="pubmed",
        domain="biomedical_literature",
        display_name="PubMed MCP",
        mcp_url="http://127.0.0.1:9807/mcp",
        mcp_url_env="MCP_PUBMED_URL",
        description="PubMed literature retrieval endpoint (self-hostable/bridgeable).",
        transport="stdio",
        auth="none",
        tool_count=16,
        stdio_command=("node", "build/index.js"),
        stdio_install="mkdir -p workloads/mcp_support/local_servers && cd workloads/mcp_support/local_servers && (test -d PubMed-MCP-Server || git clone https://github.com/Augmented-Nature/PubMed-MCP-Server.git) && cd PubMed-MCP-Server && npm install && npm run build",
        notes="Must be cloned from source and built; no published npm package.",
        sources=("https://github.com/Augmented-Nature/PubMed-MCP-Server",),
    ),
    "scientific-computation-mcp": PublicReadonlyMCPServer(
        key="scientific-computation-mcp",
        domain="scientific_compute",
        display_name="Scientific Computation MCP",
        mcp_url="http://127.0.0.1:9808/mcp",
        mcp_url_env="MCP_SCI_COMP_URL",
        description="Scientific computation endpoint (requires Smithery key).",
        transport="stdio",
        auth="smithery_key",
        tool_count=26,
        stdio_command=("npx", "-y", "@smithery/cli@latest", "run",
                        "@Aman-Amith-Shastry/scientific_computation_mcp"),
        stdio_install="export SMITHERY_API_KEY=YOUR_SMITHERY_API_KEY && npx -y @smithery/cli@latest run @Aman-Amith-Shastry/scientific_computation_mcp",
        notes="Requires SMITHERY_API_KEY and interactive Smithery OAuth on first connection.",
        sources=("https://github.com/Aman-Amith-Shastry/scientific_computation_mcp",),
    ),
    "tavily": PublicReadonlyMCPServer(
        key="tavily",
        domain="web_search_research_api",
        display_name="Tavily",
        mcp_url="https://mcp.tavily.com/mcp/",
        mcp_url_env="MCP_TAVILY_URL",
        description="Search/research MCP endpoint (API key required).",
        transport="remote_http",
        auth="api_key",
        tool_count=5,
        notes="Requires Authorization: Bearer <key> header; connection fails without it.",
        sources=("https://docs.tavily.com/documentation/mcp",),
    ),
    "time-mcp": PublicReadonlyMCPServer(
        key="time-mcp",
        domain="time_timezone_utility",
        display_name="Time MCP",
        mcp_url="http://127.0.0.1:9809/mcp",
        mcp_url_env="MCP_TIME_URL",
        description="Time/timezone utility endpoint (self-hostable/bridgeable).",
        transport="stdio",
        auth="none",
        tool_count=6,
        stdio_command=("node", "dist/index.js"),
        stdio_install="mkdir -p workloads/mcp_support/local_servers && cd workloads/mcp_support/local_servers && (test -d mcp-server-http-time || git clone https://github.com/Cam10001110101/mcp-server-http-time.git) && cd mcp-server-http-time && npm install && npm run build",
        notes="npm package @cbuk100011/mcp-time is unpublished (404); must clone from source.",
        sources=("https://github.com/Cam10001110101/mcp-server-http-time",),
    ),
    "weather-mcp": PublicReadonlyMCPServer(
        key="weather-mcp",
        domain="weather_retrieval",
        display_name="Weather MCP",
        mcp_url="http://127.0.0.1:9810/mcp",
        mcp_url_env="MCP_WEATHER_URL",
        description="Weather information endpoint (self-hostable/bridgeable).",
        transport="stdio",
        auth="none",
        tool_count=8,
        stdio_command=("python", "-m", "mcp_weather_server"),
        stdio_install="pip install mcp_weather_server",
        sources=("https://github.com/isdaniel/mcp_weather_server",),
    ),
    "wikipedia": PublicReadonlyMCPServer(
        key="wikipedia",
        domain="encyclopedia_retrieval",
        display_name="Wikipedia MCP",
        mcp_url="http://127.0.0.1:9811/mcp",
        mcp_url_env="MCP_WIKIPEDIA_URL",
        description="Wikipedia public knowledge retrieval endpoint (self-hostable/bridgeable).",
        transport="stdio",
        auth="none",
        tool_count=10,
        stdio_command=("node", "dist/index.js"),
        stdio_install="mkdir -p workloads/mcp_support/local_servers && cd workloads/mcp_support/local_servers && (test -d wikipedia-mcp-server || git clone https://github.com/1999AZZAR/wikipedia-mcp-server.git) && cd wikipedia-mcp-server && npm install && npm install --save-dev @cloudflare/workers-types && npm run build",
        notes="Must be cloned from source; build requires adding @cloudflare/workers-types as devDep.",
        sources=("https://github.com/1999AZZAR/wikipedia-mcp-server",),
    ),
}


def list_public_readonly_mcp_servers() -> List[PublicReadonlyMCPServer]:
    return sorted(PUBLIC_READONLY_MCP_SERVERS.values(), key=lambda item: item.key)


def get_public_readonly_mcp_server(key: str) -> PublicReadonlyMCPServer:
    server = PUBLIC_READONLY_MCP_SERVERS.get(key)
    if server is None:
        available = ", ".join(sorted(PUBLIC_READONLY_MCP_SERVERS))
        raise KeyError(f"Unknown public read-only MCP server '{key}'. Available: {available}")
    return server


def ensure_default_mcp_headers(default_headers: Mapping[str, str]) -> None:
    if not default_headers:
        return

    raw = os.environ.get("MCP_HEADERS_JSON", "").strip()
    parsed: Dict[str, str] = {}
    if raw:
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("MCP_HEADERS_JSON must decode to a JSON object.")
        parsed = {str(k): str(v) for k, v in loaded.items()}

    changed = False
    for header, value in default_headers.items():
        if header not in parsed:
            parsed[header] = value
            changed = True

    if changed:
        os.environ["MCP_HEADERS_JSON"] = json.dumps(parsed)
