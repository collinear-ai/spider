from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple
from urllib.parse import urlparse


@dataclass(frozen=True)
class PublicReadonlyMCPServer:
    key: str
    domain: str
    display_name: str
    mcp_url: str
    mcp_url_env: str
    description: str
    transport: str
    auth: str
    tool_count: Optional[int] = None
    stdio_command: Optional[Tuple[str, ...]] = None
    stdio_install: Optional[str] = None
    local_source_subdir: Optional[str] = None
    required_env_vars: Tuple[str, ...] = field(default_factory=tuple)
    auth_env_headers: Mapping[str, str] = field(default_factory=dict)
    auth_header_prefixes: Mapping[str, str] = field(default_factory=dict)
    runtime_packages: Tuple[str, ...] = field(default_factory=tuple)
    bootstrap_commands: Tuple[str, ...] = field(default_factory=tuple)
    notes: Optional[str] = None
    default_headers: Mapping[str, str] = field(default_factory=dict)
    sources: Tuple[str, ...] = field(default_factory=tuple)


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
        stdio_install="uvx arxiv-paper-mcp",
        runtime_packages=("uv",),
        bootstrap_commands=("uvx arxiv-paper-mcp",),
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
        stdio_install="git clone https://github.com/Augmented-Nature/ClinicalTrials-MCP-Server.git && npm install && npm run build",
        local_source_subdir="workloads/mcp_support/local_servers/ClinicalTrials-MCP-Server",
        runtime_packages=("node", "npm"),
        bootstrap_commands=(
            "mkdir -p workloads/mcp_support/local_servers",
            "cd workloads/mcp_support/local_servers && (test -d ClinicalTrials-MCP-Server || git clone https://github.com/Augmented-Nature/ClinicalTrials-MCP-Server.git)",
            "cd workloads/mcp_support/local_servers/ClinicalTrials-MCP-Server && npm install && npm run build",
        ),
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
        required_env_vars=("FINANCIAL_API_KEY",),
        auth_env_headers={"X-API-Key": "FINANCIAL_API_KEY"},
        default_headers={"X-API-Key": ""},
        notes="Requires X-API-Key header; connection fails without it.",
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
        required_env_vars=("GOOGLE_MAPS_API_KEY",),
        auth_env_headers={"FORWARD_VAR_KEY": "GOOGLE_MAPS_API_KEY"},
        notes="Tool listing accessible without auth; invocation requires Google Maps API key.",
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
        runtime_packages=("node", "npx"),
        bootstrap_commands=("npx -y @jinzcdev/leetcode-mcp-server",),
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
        stdio_install="git clone https://github.com/Augmented-Nature/PubMed-MCP-Server.git && npm install && npm run build",
        local_source_subdir="workloads/mcp_support/local_servers/PubMed-MCP-Server",
        runtime_packages=("node", "npm"),
        bootstrap_commands=(
            "mkdir -p workloads/mcp_support/local_servers",
            "cd workloads/mcp_support/local_servers && (test -d PubMed-MCP-Server || git clone https://github.com/Augmented-Nature/PubMed-MCP-Server.git)",
            "cd workloads/mcp_support/local_servers/PubMed-MCP-Server && npm install && npm run build",
        ),
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
        stdio_command=(
            "npx",
            "-y",
            "@smithery/cli@latest",
            "run",
            "@Aman-Amith-Shastry/scientific_computation_mcp",
            "--key",
            "<SMITHERY_API_KEY>",
        ),
        stdio_install="npx -y @smithery/cli@latest run @Aman-Amith-Shastry/scientific_computation_mcp --key <SMITHERY_API_KEY>",
        required_env_vars=("SMITHERY_API_KEY",),
        runtime_packages=("node", "npx"),
        bootstrap_commands=("npx -y @smithery/cli@latest run @Aman-Amith-Shastry/scientific_computation_mcp --key <SMITHERY_API_KEY>",),
        notes="Requires SMITHERY_API_KEY.",
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
        required_env_vars=("TAVILY_API_KEY",),
        auth_env_headers={"Authorization": "TAVILY_API_KEY"},
        auth_header_prefixes={"Authorization": "Bearer "},
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
        stdio_install="git clone https://github.com/Cam10001110101/mcp-server-http-time.git && npm install && npm run build",
        local_source_subdir="workloads/mcp_support/local_servers/mcp-server-http-time",
        runtime_packages=("node", "npm"),
        bootstrap_commands=(
            "mkdir -p workloads/mcp_support/local_servers",
            "cd workloads/mcp_support/local_servers && (test -d mcp-server-http-time || git clone https://github.com/Cam10001110101/mcp-server-http-time.git)",
            "cd workloads/mcp_support/local_servers/mcp-server-http-time && npm install && npm run build",
        ),
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
        stdio_install="python -m pip install mcp_weather_server",
        runtime_packages=("mcp_weather_server",),
        bootstrap_commands=("python -m pip install mcp_weather_server",),
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
        stdio_install="git clone https://github.com/1999AZZAR/wikipedia-mcp-server.git && npm install && npm install --save-dev @cloudflare/workers-types && npm run build",
        local_source_subdir="workloads/mcp_support/local_servers/wikipedia-mcp-server",
        runtime_packages=("node", "npm"),
        bootstrap_commands=(
            "mkdir -p workloads/mcp_support/local_servers",
            "cd workloads/mcp_support/local_servers && (test -d wikipedia-mcp-server || git clone https://github.com/1999AZZAR/wikipedia-mcp-server.git)",
            "cd workloads/mcp_support/local_servers/wikipedia-mcp-server && npm install && npm install --save-dev @cloudflare/workers-types && npm run build",
        ),
        notes="Must be cloned from source; build requires @cloudflare/workers-types.",
        sources=("https://github.com/1999AZZAR/wikipedia-mcp-server",),
    ),
}
_RUNTIME_READY: Set[str] = set()
_PLACEHOLDER_RE = re.compile(r"<([A-Z0-9_]+)>")


def list_public_readonly_mcp_servers() -> List[PublicReadonlyMCPServer]:
    return sorted(PUBLIC_READONLY_MCP_SERVERS.values(), key=lambda item: item.key)


def get_public_readonly_mcp_server(key: str) -> PublicReadonlyMCPServer:
    server = PUBLIC_READONLY_MCP_SERVERS.get(key)
    if server is None:
        available = ", ".join(sorted(PUBLIC_READONLY_MCP_SERVERS))
        raise KeyError(f"Unknown public read-only MCP server '{key}'. Available: {available}")
    return server


def is_localhost_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def server_requires_local_proxy(server: PublicReadonlyMCPServer) -> bool:
    return bool(server.stdio_command and is_localhost_url(server.mcp_url))


def resolve_stdio_cwd(server: PublicReadonlyMCPServer, repo_root: Optional[Path] = None) -> Optional[Path]:
    if not server.local_source_subdir:
        return None
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    return root / server.local_source_subdir


def headers_for_server(server: PublicReadonlyMCPServer, environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = environ or os.environ
    headers: Dict[str, str] = {}
    for header, env_var in server.auth_env_headers.items():
        value = str(env.get(env_var, "")).strip()
        if not value:
            continue
        prefix = server.auth_header_prefixes.get(header, "")
        headers[header] = f"{prefix}{value}"
    for header, value in server.default_headers.items():
        headers.setdefault(str(header), str(value))
    return headers


def validate_required_env_vars(server: PublicReadonlyMCPServer, environ: Optional[Mapping[str, str]] = None) -> None:
    env = environ or os.environ
    missing = [name for name in server.required_env_vars if not str(env.get(name, "")).strip()]
    if missing:
        raise RuntimeError(f"missing required env vars for {server.key}: {', '.join(missing)}")


def ensure_server_runtime_ready(server: PublicReadonlyMCPServer, repo_root: Optional[Path] = None) -> None:
    if not server_requires_local_proxy(server):
        validate_required_env_vars(server)
        return
    if server.key in _RUNTIME_READY:
        validate_required_env_vars(server)
        return
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    commands = tuple(cmd for cmd in server.bootstrap_commands if cmd.strip())
    if not commands and server.stdio_install and server.stdio_install.strip():
        commands = (server.stdio_install,)
    def expand(cmd: str) -> str:
        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            value = str(os.environ.get(var, "")).strip()
            if not value:
                raise RuntimeError(f"missing env var for command placeholder: {var} ({server.key})")
            return value
        return _PLACEHOLDER_RE.sub(repl, cmd)
    for cmd in commands:
        subprocess.run(expand(cmd), shell=True, check=True, cwd=str(root), env=os.environ.copy())
    _RUNTIME_READY.add(server.key)
    validate_required_env_vars(server)


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
