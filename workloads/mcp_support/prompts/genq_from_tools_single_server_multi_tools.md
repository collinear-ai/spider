## Task
Generate a **Tool Use Question** based on the provided MCP Server and its tool descriptions.

## Objective
Analyze the provided MCP Server and its available tools, then create a realistic user question that would naturally require the use of **{NUM_TOOLS} tools** from this MCP Server to solve completely.

## Guidelines

### Question Realism
- Create questions that represent real-world scenarios where users would need to interact with the MCP Server's tools
- The question should sound natural and authentic, as if asked by someone genuinely needing to accomplish a task
- Consider common use cases, problems, or workflows that would require the functionality provided by the MCP Server's tools
- Do not ask for information that is likely nonexistent or unverifiable (for example, fabricated paper IDs, fake record IDs, or imaginary entities). If a direct lookup target may not exist, reformulate into criteria-based search (for example, "find papers about X published after Y with method Z").
- Do not ask for information tied to private user accounts, private databases, internal dashboards, personal files, or authenticated tenant-specific records. All requested information should be publicly accessible via read-only public sources.

### Tool Selection
- Focus on **{NUM_TOOLS} tools** from the MCP Server that would work together to answer the question
- The question should require a sequence or combination of tool calls to solve completely
- Choose tools based on how they complement each other and create a logical workflow
- Consider each tool's description and purpose when crafting the question that requires multiple steps

### Question Complexity
- Create questions that are complex enough to warrant using {NUM_TOOLS} tools
- The question should have multiple components or require several steps to solve
- Include relevant context or constraints that make the multi-tool usage necessary
- Do not contain the exact tool names in the question
- Ensure the question cannot be reasonably answered with just a single tool
- You must generate exactly {NUM_TURNS} user questions (follow-ups) that use the SAME {NUM_TOOLS} tools and have similar complexity
- Follow-ups should be distinct, realistic refinements or continuations of the original scenario
- Do not introduce new tools or shift domains in follow-ups

### Output Format
Your response should include:
1. **Tool Analysis**: Briefly analyze the MCP Server's available tools and their main functionalities.
2. **Target Tools**: The specific tool names from the MCP Server that should be used together to answer this question, in the order they would likely be called.
3. **Questions**: A list of exactly {NUM_TURNS} user questions that all require the SAME {NUM_TOOLS} tools.

## MCP Server Description
{MCP_SERVER_NAME}: {MCP_SERVER_DESCRIPTION}

Available Tools:
{TOOL_LIST}

## Output
Ensure your question requires exactly {NUM_TOOLS} tools to solve completely. Provide your response in the following XML format:

<response>
  <server_analysis>
    <!-- Briefly analyze the MCP Server's available tools and their main functionalities. -->
  </server_analysis>
  <target_tools>
    <!-- The specific tool names from the MCP Server that should be used together to answer this question, listed in order. e.g., <tool>create_twitter_post</tool> <tool>get_last_tweet</tool> -->
  </target_tools>
  <questions>
    <!-- Provide exactly {NUM_TURNS} questions. All require the SAME tools and similar complexity. -->
    <question>...</question>
    <question>...</question>
  </questions>
</response>
