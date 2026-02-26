# Public Read-Only MCP servers

## Constraints

This section consolidates the earlier iteration filters (former Iteration 1/2/3) into stable constraints used for selection.

- Constraint 1: Transport and protocol compatibility
  - Must be a remote MCP endpoint over streamable HTTP/HTTPS.
  - Must work with the current stack:
    - `workloads/mcp_support/launcher.py` for URL/env wiring.
    - `workloads/mcp_support/tool_schemas.py` for `tool_config_from_server(...)` schema discovery.
  - Why this constraint exists:
    - Dataset generation scripts need drop-in endpoints that can be introspected and invoked without custom adapters.

- Constraint 2: No local database connector setup
  - Selected servers must not require users to configure a local DB connection (for example self-hosted Postgres/MySQL wiring) to be usable.
  - API-key/OAuth authentication is acceptable if the endpoint remains remotely hosted and directly callable.
  - Why this constraint exists:
    - The generation flow should remain lightweight and reproducible across environments without per-user infra setup.

- Constraint 3: No personal-account state dependency for baseline utility
  - Prefer endpoints that provide public/read-only value immediately, or at least expose a clear API-key path without needing custom tenant bootstrap.
  - Endpoints that are auth-gated are kept only if they still fit the “remote MCP + no DB setup” requirement.
  - Why this constraint exists:
    - The workload target is scalable dataset generation; server setup friction must stay low.

- Constraint 4: Production-useful agentic surface area
  - Coverage should include realistic operations (research, docs retrieval, geospatial, market data, cloud ops, etc.).
  - Tool surfaces should be broad enough to support multi-step trajectories.
  - Why this constraint exists:
    - The goal is valuable production-like agent behavior, not toy tool invocation.

- Constraint 5: Verifiability and recency
  - Endpoint viability is validated via live probing (`tool_config_from_server`) when possible.
  - If an endpoint is auth-gated, probing without credentials is expected to return `401`; tool count is marked `Unknown` until authenticated.
  - Why this constraint exists:
    - Avoid stale directory-only assumptions and keep server metadata tied to current observed behavior.

## Servers

- `arxiv-paper-mcp`
  - URL: `http://127.0.0.1:9802/mcp`
  - Auth: No auth
  - Tool count: 4
  - Probe status: OK (via `uvx arxiv-paper-mcp` + stdio_proxy)
  - Tools: `scrape_recent_category_papers`, `search_papers`, `get_paper_info`, `analyze_trends`
  - Details: Research-paper retrieval over arXiv for literature discovery, metadata lookup, and abstract-driven evidence gathering tasks.
  - Source: `https://github.com/daheepk/arxiv-paper-mcp`

- `context7`
  - URL: `https://mcp.context7.com/mcp`
  - Auth: No auth
  - Tool count: 2
  - Probe status: OK (direct streamable HTTP probe)
  - Tools: `resolve_library_id`, `query_docs`
  - Details: Minimal but high-signal developer-doc workflow: resolve the correct package/library identifier first, then query relevant docs snippets. Useful when an agent must disambiguate similarly named libraries and fetch implementation details quickly for coding/support tasks.
  - Source: `https://context7.com/docs/mcp-server`

- `deepwiki`
  - URL: `https://mcp.deepwiki.com/mcp`
  - Auth: No auth
  - Tool count: 3
  - Probe status: OK (direct streamable HTTP probe)
  - Tools: `read_wiki_structure`, `read_wiki_contents`, `ask_question`
  - Details: Repository-centric analysis endpoint that exposes structural exploration plus content retrieval and question answering over code/wiki context. Useful for onboarding, architecture mapping, and answering “where/how is X implemented?” tasks in unfamiliar codebases.
  - Source: `https://docs.devin.ai/work-with-devin/deepwiki-mcp`

- `financialdatasets`
  - URL: `https://mcp.financialdatasets.ai/api`
  - Auth: API key (header: `X-API-Key`)
  - Tool count: 13
  - Probe status: OK (direct streamable HTTP probe with X-API-Key header)
  - Tools: `getAvailableCryptoTickers`, `getCryptoPriceSnapshot`, `getCryptoPrices`, `getBalanceSheet`, `getIncomeStatement`, `getCashFlowStatement`, `getFinancialMetrics`, `getNews`, `getFilings`, `getFilingItems`, `getAvailableFilingItems`, `getCompanyFacts`, `getSegmentedRevenues`
  - Details: Authenticated financial-data MCP for pulling structured market/fundamental datasets. Useful for analytics tasks such as factor comparisons, period-over-period metric extraction, screening pipelines, and report generation without direct database setup.
  - Source: `https://mcp.financialdatasets.ai/api`

- `google-maps`
  - URL: `https://mcp.open-mcp.org/api/server/google-maps@latest/mcp`
  - Auth: API key
  - Tool count: 18
  - Probe status: OK (tool listing accessible without auth; invocation requires KEY env var)
  - Tools: `expandSchema`, `geolocate`, `directions`, `elevation`, `geocode`, `timezone`, `snaptoroads`, `nearestroads`, `distancematrix`, `placedetails`, `findplacefromtext`, `nearbysearch`, `textsearch`, `placephoto`, `queryautocomplete`, `autocomplete`, `streetview`, `streetviewmetadata`
  - Details: Comprehensive geospatial stack covering address/coordinate conversion, routing, travel-time matrix estimation, place discovery/details, road snapping, and street-view context. Useful for dispatch planning, travel assistant flows, and geo-enrichment in customer-support operations.
  - Source: `https://mcp.open-mcp.org/server/google-maps`

- `leetcode`
  - URL: `http://127.0.0.1:9805/mcp`
  - Auth: No auth
  - Tool count: 9
  - Probe status: OK (via `npx -y @jinzcdev/leetcode-mcp-server --site global` + stdio_proxy)
  - Tools: `get_daily_challenge`, `get_problem`, `search_problems`, `get_user_profile`, `get_recent_submissions`, `get_recent_ac_submissions`, `get_user_contest_ranking`, `list_problem_solutions`, `get_problem_solution`
  - Details: Coding-problem retrieval and associated context useful for programming-practice agents and benchmark-style problem navigation.
  - Source: `https://github.com/jinzcdev/leetcode-mcp-server`

- `open-weather`
  - URL: `https://mcp.open-mcp.org/api/server/open-weather@latest/mcp`
  - Auth: No auth
  - Tool count: 2
  - Probe status: OK (direct streamable HTTP probe)
  - Tools: `expandSchema`, `getweatherdata`
  - Details: Lightweight global weather retrieval endpoint suitable for adding environmental context into planning workflows. Useful for quick forecast/current-condition checks in travel, field-ops, or event-risk tasks where weather is a secondary but important variable.
  - Source: `https://mcp.open-mcp.org/server/open-weather`

- `pubmed`
  - URL: `http://127.0.0.1:9807/mcp`
  - Auth: No auth
  - Tool count: 16
  - Probe status: OK (via `node build/index.js` + stdio_proxy)
  - Tools: `search_articles`, `advanced_search`, `search_by_author`, `search_by_journal`, `search_by_mesh_terms`, `get_trending_articles`, `get_article_details`, `get_abstract`, `get_full_text`, `batch_article_lookup`, `get_cited_by`, `get_references`, `get_similar_articles`, `export_citation`, `validate_pmid`, `convert_identifiers`
  - Details: Biomedical literature retrieval over PubMed for evidence-backed medical/scientific information tasks.
  - Source: `https://github.com/Augmented-Nature/PubMed-MCP-Server`

- `tavily`
  - URL: `https://mcp.tavily.com/mcp/`
  - Auth: API key (header: `Authorization: Bearer <key>`)
  - Tool count: 5
  - Probe status: OK (direct streamable HTTP probe with Bearer token)
  - Tools: `tavily_search`, `tavily_extract`, `tavily_crawl`, `tavily_map`, `tavily_research`
  - Details: Authenticated web-search/research endpoint aimed at retrieval-heavy agent workflows. Useful for evidence collection, source triangulation, and answer grounding where agents need up-to-date web context and structured search outputs.
  - Source: `https://docs.tavily.com/documentation/mcp`

- `time-mcp`
  - URL: `http://127.0.0.1:9809/mcp`
  - Auth: No auth
  - Tool count: 6
  - Probe status: OK (via `node dist/index.js` from cloned repo + stdio_proxy; npm package `@cbuk100011/mcp-time` is unpublished/404)
  - Tools: `current_time`, `relative_time`, `days_in_month`, `get_timestamp`, `convert_time`, `get_week_year`
  - Details: Time/timezone utility endpoint for date arithmetic, conversion, scheduling normalization, and temporal consistency checks.
  - Source: `https://github.com/Cam10001110101/mcp-server-http-time`

- `weather-mcp`
  - URL: `http://127.0.0.1:9810/mcp`
  - Auth: No auth
  - Tool count: 8
  - Probe status: OK (via `python -m mcp_weather_server` + stdio_proxy)
  - Tools: `get_current_weather`, `get_weather_byDateTimeRange`, `get_weather_details`, `get_current_datetime`, `get_timezone_info`, `convert_time`, `get_air_quality`, `get_air_quality_details`
  - Details: Weather retrieval endpoint for current/forecast context in travel, field operations, and risk-aware planning tasks.
  - Source: `https://github.com/isdaniel/mcp_weather_server`

- `wikipedia`
  - URL: `http://127.0.0.1:9811/mcp`
  - Auth: No auth
  - Tool count: 10
  - Probe status: OK (via `node dist/index.js` from cloned repo + stdio_proxy)
  - Tools: `search`, `getPage`, `getPageSummary`, `getPageById`, `random`, `pageLanguages`, `batchSearch`, `batchGetPages`, `searchNearby`, `getPagesInCategory`
  - Details: Public encyclopedia retrieval endpoint for neutral background research, entity disambiguation, and citation-oriented knowledge tasks.
  - Source: `https://github.com/1999AZZAR/wikipedia-mcp-server`
