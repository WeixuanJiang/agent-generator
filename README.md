# Auto AI Agents Generator

Autonomous multi-agent system generator powered by Claude Code CLI. Provide business context in natural language, and the system will autonomously generate, test, debug, and perfect a multi-agent system using your preferred AI framework.

## Features

- **Fully Autonomous**: Zero human intervention required - runs until the generated system works perfectly
- **Free-form Context**: Describe your requirements in natural language, no strict schemas
- **Multi-Framework Support**: LangChain, LangGraph, Strands Agents, CrewAI
- **Multi-Provider Support**: AWS Bedrock, OpenAI, Google Gemini, Anthropic
- **Auto MCP Setup**: Automatically detects and configures required MCP servers
- **Iterative Refinement**: Generate → Test → Debug → Fix → Repeat until perfect
- **Progress Tracking**: All thinking, planning, and execution saved for resumption
- **Comprehensive Testing**: Auto-generates and runs test cases

## Prerequisites

1. **Claude Code CLI**: Install from [https://code.claude.com](https://code.claude.com)
2. **Python 3.9+**: Required for running the generator
3. **Node.js & npm**: Optional, only required if using MCP servers
4. **Model Provider Credentials**: API keys for your chosen provider (OpenAI, Anthropic, AWS, or Google)

## Installation

```bash
# Clone or download this repository
cd auto_agent_generator

# Install in editable mode (optional - for running as console command)
pip install -e .

# Verify Claude CLI is installed
claude --version
```

## Quick Start

### 1. Create Context File

Create `data/context.md` describing your use case in natural language:

```markdown
# My Multi-Agent System

## Overview
Build a system that processes customer feedback and generates insights.

## Business Flow
1. Collect customer feedback from multiple sources
2. Analyze sentiment and categorize feedback
3. Generate actionable insights
4. Create summary reports

## Inputs
- Customer feedback text
- Feedback metadata (source, timestamp, customer ID)

## Outputs
- Categorized feedback
- Sentiment analysis results
- Insight reports

## Data Sources
- PostgreSQL database with feedback table
- REST API for customer information

## Required Tools
- Database access (postgres)
- Web search for competitive analysis
```

Check the `data/context.md` file in this repository for an example.

### 2. Create Configuration File

Create `data/config.json` with your settings:

```json
{
  "framework": "langchain",
  "model_provider": {
    "name": "openai",
    "model": "gpt-4-turbo-preview",
    "credentials": {
      "api_key": "env:OPENAI_API_KEY"
    },
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 4096
    }
  },
  "generation_settings": {
    "max_iterations": 10,
    "enable_testing": true,
    "output_directory": "outputs/",
    "verbose": true,
    "save_progress": true
  }
}
```

**Optional Configuration Fields:**
- `llm_runner`: Set to `"claude"` (default) or `"codex"` to specify which CLI interface to use
- `mcp_servers`: List of MCP server names to use (e.g., `["filesystem", "github", "postgres"]`)

Check the `data/config.json` file in this repository for an example.

### 3. Set Environment Variables

```bash
# For Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# For OpenAI
export OPENAI_API_KEY="your-api-key"

# For AWS Bedrock
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

### 4. Run the Generator

```bash
python -m auto_agent_generator.cli

# Or if you installed with pip install -e .
auto-agent-generator
```

The system will:
1. ✓ Parse your context
2. ✓ Analyze and plan architecture
3. ✓ Setup required MCPs
4. ✓ Generate initial code
5. ✓ Validate and test
6. ✓ Debug and fix issues
7. ✓ Iterate until perfect

## Project Structure

```
auto_agent_generator/
├── data/                         # User-provided inputs
│   ├── context.md                # Your business context
│   └── config.json               # Your configuration
├── src/auto_agent_generator/     # Package source
│   ├── cli.py                    # CLI entry point
│   ├── pipeline/generator.py     # Main orchestrator
│   ├── context/parser.py         # Parse context.md
│   ├── config/parser.py          # Parse config.json
│   ├── progress/manager.py       # Track progress
│   ├── llm/claude.py             # Claude CLI wrapper
│   ├── llm/codex.py              # Codex CLI wrapper
│   └── mcp/manager.py            # MCP auto-setup
├── progress/                     # Execution progress artifacts
├── outputs/                      # Generated systems
├── pyproject.toml                # Packaging + console script
└── README.md                     # This file
```

## Supported Frameworks

### LangChain
Traditional chain-based workflow framework
```json
{"framework": "langchain"}
```

### LangGraph
Graph-based workflow with state management
```json
{"framework": "langgraph"}
```

### Strands Agents
Composio's agent framework
```json
{"framework": "strands"}
```

### CrewAI
Role-based multi-agent collaboration
```json
{"framework": "crewai"}
```

## Supported Model Providers

### Anthropic Claude
```json
{
  "model_provider": {
    "name": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "credentials": {"api_key": "env:ANTHROPIC_API_KEY"}
  }
}
```

### OpenAI
```json
{
  "model_provider": {
    "name": "openai",
    "model": "gpt-4-turbo-preview",
    "credentials": {"api_key": "env:OPENAI_API_KEY"}
  }
}
```

### AWS Bedrock
```json
{
  "model_provider": {
    "name": "aws",
    "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "region": "us-east-1",
    "credentials": {
      "access_key": "env:AWS_ACCESS_KEY_ID",
      "secret_key": "env:AWS_SECRET_ACCESS_KEY"
    }
  }
}
```

### Google Gemini
```json
{
  "model_provider": {
    "name": "gemini",
    "model": "gemini-pro",
    "credentials": {"api_key": "env:GOOGLE_API_KEY"}
  }
}
```

## MCP Server Support

MCP (Model Context Protocol) servers can be configured to provide additional capabilities to the generated agents. The system can work with standard MCP servers when specified in your configuration.

To use MCP servers, add them to your `config.json`:

```json
{
  "mcp_servers": ["filesystem", "github", "postgres"]
}
```

Common MCP servers include:
- **filesystem**: File system operations
- **github**: GitHub API integration
- **postgres**: PostgreSQL database
- **sqlite**: SQLite database
- **brave-search**: Web search
- And others available through the MCP ecosystem

**Note**: MCP servers are optional. The generator will analyze your context and suggest appropriate MCPs, or you can explicitly specify them in your configuration.

## Command Line Interface

```bash
# Run with data/context.md and data/config.json
python -m auto_agent_generator.cli

# Clear progress and start fresh
python -m auto_agent_generator.cli --restart

# Override LLM runner from config
python -m auto_agent_generator.cli --llm claude
python -m auto_agent_generator.cli --llm codex

# Show current status
python -m auto_agent_generator.cli status

# Clear progress files
python -m auto_agent_generator.cli clear

# Show help
python -m auto_agent_generator.cli help
```

## Configuration Options

### Generation Settings

```json
{
  "generation_settings": {
    "max_iterations": 10,
    "enable_testing": true,
    "output_directory": "outputs/",
    "verbose": true,
    "save_progress": true
  }
}
```

- `max_iterations`: Maximum debug iterations (default: 10)
- `enable_testing`: Run tests after generation (default: true)
- `output_directory`: Where to save generated code (default: "outputs/")
- `verbose`: Detailed output (default: true)
- `save_progress`: Save execution progress (default: true)

### Model Parameters

```json
{
  "model_provider": {
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 4096,
      "top_p": 1.0
    }
  }
}
```

- `temperature`: Creativity level, 0.0-1.0 (default: 0.7)
- `max_tokens`: Maximum response length (default: 4096)
- `top_p`: Nucleus sampling parameter (default: 1.0)

## Progress Tracking

All execution progress is saved to the `progress/` folder:

- **thinking.md**: Claude's analysis and reasoning
- **planning.md**: Architecture and design decisions
- **generation.md**: Code generation steps
- **testing.md**: Test execution results
- **debugging.md**: Error fixes and iterations
- **summary.md**: Overall progress summary
- **errors.md**: Error log

These files accumulate context for iterative refinement and enable resumption if interrupted.

## Output Structure

Generated systems are saved to `outputs/` (configurable) with:

```
outputs/
├── main.py                    # Main agent system
├── agents/                    # Agent definitions
├── tools/                     # Custom tools
├── config/                    # Configuration files
├── tests/                     # Test cases
├── README.md                  # Usage instructions
├── requirements.txt           # Dependencies
├── mcp_config.json           # MCP server config
├── .env.mcp.template         # Environment variables template
└── GENERATION_REPORT.md      # Generation report
```

## Example Use Cases

### Customer Support Automation

```markdown
# Customer Support Automation

Build a multi-agent system that processes customer support tickets.

## Workflow
1. Classify incoming tickets by priority and category
2. Search knowledge base for relevant solutions
3. Generate draft responses
4. Quality review before sending

## Agents
- Classifier: Categorizes and prioritizes tickets
- Searcher: Finds relevant knowledge base articles
- Responder: Drafts customer responses
- Reviewer: Quality checks responses
```

### Data Pipeline

```markdown
# Data Pipeline System

Build a multi-agent system that:
1. Extracts data from multiple APIs
2. Transforms and validates data
3. Loads into data warehouse
4. Generates quality reports

## Agents
Extractor, Transformer, Validator, Loader, Reporter
```

### Content Creation

```markdown
# Content Creation Pipeline

Multi-agent system for:
1. Research topics and gather information
2. Generate article drafts
3. Fact-check content
4. Optimize for SEO
5. Publish to CMS

## Agents
Researcher, Writer, FactChecker, SEOOptimizer, Publisher
```

## Troubleshooting

### Issue: Claude Code CLI not found
```bash
# Install Claude Code CLI
# Visit: https://code.claude.com

# Verify installation
claude --version
```

### Issue: MCP server not available
```bash
# Check Node.js and npm are installed
node --version
npm --version

# Test MCP package
npx -y @modelcontextprotocol/server-filesystem --help
```

### Issue: API key not found
```bash
# Set environment variable
export ANTHROPIC_API_KEY="your-key"

# Or add to .env file
echo "ANTHROPIC_API_KEY=your-key" >> .env
```

### Issue: Generation failed

1. Check `progress/errors.md` for error details
2. Review `auto_agent_generator.log` for full logs
3. Verify data/context.md syntax
4. Check config.json is valid JSON
5. Ensure all dependencies are installed

## Advanced Usage

### Custom MCP Configuration

Add custom MCPs programmatically:

```python
from auto_agent_generator.mcp.manager import MCPManager

mcp_manager = MCPManager()
mcp_manager.add_custom_mcp("my-custom-mcp", {
    "command": "node",
    "args": ["/path/to/my-mcp-server.js"],
    "env": {"API_KEY": "${MY_API_KEY}"}
})
```

### Resume Interrupted Generation

Progress is automatically saved. Simply run again:

```bash
python -m auto_agent_generator.cli
```

The system will load previous progress and continue.

### Adjust Iteration Limit

```json
{
  "generation_settings": {
    "max_iterations": 20
  }
}
```

Increase `max_iterations` for more complex systems that may require additional debugging cycles.

## Architecture

The system consists of several key components:

- **CLI** (`cli.py`): Entry point and command-line interface
- **Context Parser** (`context/parser.py`): Parses business context from markdown
- **Config Parser** (`config/parser.py`): Loads and validates configuration
- **LLM Interface** (`llm/claude.py`, `llm/codex.py`): Wrappers for AI CLI tools
- **MCP Manager** (`mcp/manager.py`): Manages MCP server configuration
- **Agent Generator** (`pipeline/generator.py`): Main orchestration and generation logic
- **Progress Manager** (`progress/manager.py`): Tracks progress and enables resumption

