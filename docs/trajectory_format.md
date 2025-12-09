# OpenHands Trajectory Format

This document describes the trajectory format used by Spider's OpenHands wrapper, which matches the format used by [SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories).

## Dataset Structure

Each dataset row contains:

### Core Fields

- **`instance_id`** (string): Unique task identifier (e.g., `"conan-io__conan-123"`)
- **`resolved`** (bool): Whether the task was successfully solved
- **`messages`** (list): Chat-format conversation for training (see below)

### Metadata Fields

- **`repo`** (string): Repository name
- **`image_name`** (string): Docker image used for the task
- **`instruction`** (string): Task instruction given to the agent
- **`problem_statement`** (string): Original problem description
- **`git_patch`** (string): Generated code changes
- **`agent_class`** (string): Agent type used (e.g., "CodeActAgent")
- **`model`** (string): LLM model used (e.g., "gpt-4o")
- **`max_iterations`** (int): Maximum iterations allowed
- **`metrics`** (dict): Evaluation metrics
- **`trajectory`** (list): Raw OpenHands events for analysis/debugging

## Messages Format

The \`messages\` field contains a list of dicts in standard chat format:

\`\`\`python
[
  {
    "role": "user",
    "content": "Fix the bug in the authentication system..."
  },
  {
    "role": "assistant",
    "content": "Reading file: \`auth.py\`\\n\\nOutput:\\n\`\`\`\\ndef authenticate(user):\\n    # buggy code\\n\`\`\`"
  },
  {
    "role": "assistant", 
    "content": "Running command: \`pytest tests/test_auth.py\`\\n\\nOutput:\\n\`\`\`\\nFAILED tests/test_auth.py::test_login\\n\`\`\`"
  },
  {
    "role": "assistant",
    "content": "Writing to file: \`auth.py\`\\n\`\`\`\\ndef authenticate(user):\\n    # fixed code\\n\`\`\`"
  }
]
\`\`\`

### Assistant Message Types

1. **Tool calls with observations**: Action + Output combined
2. **File operations**: Read/write with file contents
3. **Code edits**: File modifications with results

## Compatibility

This format matches [SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) and is compatible with:
- ✅ SWE-Gym training pipeline
- ✅ OpenAI fine-tuning API
- ✅ HuggingFace TRL/transformers trainers
- ✅ Any chat-format training framework

## References

- [SWE-Gym Paper](https://arxiv.org/abs/2412.21139)
- [SWE-Gym Dataset](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories)
- [SWE-Gym GitHub](https://github.com/SWE-Gym)
