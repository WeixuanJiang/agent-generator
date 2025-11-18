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

1. **Claude Code CLI** or **Codex CLI**: Install one runtime (Claude: [https://code.claude.com](https://code.claude.com), Codex: `npm/pip` per Codex docs)
2. **Python 3.8+**: Required for running the generator
3. **Node.js & npm**: Required for MCP servers
4. **Model Provider Credentials**: API keys for your chosen provider

## Installation

```bash
# Clone or download this repository
cd auto_agent_generator

# Install in editable mode
pip install -e .

# Verify desired CLI is installed (choose one)
claude --version   # or: codex --version
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

See `examples/example_context.md` for a comprehensive example.

### 2. Create Configuration File

Create `data/config.json` with your settings:

```json
{
  "framework": "langgraph",
  "llm_runner": "claude", // or "codex"
  "model_provider": {
    "name": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "credentials": {
      "api_key": "env:ANTHROPIC_API_KEY"
    },
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 4096
    }
  },
  "mcp_servers": ["filesystem", "github", "postgres"]
  "generation_settings": {
    "max_iterations": 10,
    "enable_testing": true,
    "output_directory": "outputs/",
    "verbose": true
  }
}
```

See `examples/` folder for more configuration examples.

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
python -m auto_agent_generator.cli              # defaults to Claude
# or
python -m auto_agent_generator.cli --llm codex
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
├── examples/                     # Example templates
├── pyproject.toml                # Packaging + console script
├── CODE_REFERENCE.md             # Architecture docs
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

## Available MCPs

The system automatically detects and configures these MCP servers:

- **filesystem**: File system operations
- **github**: GitHub API integration
- **gitlab**: GitLab API integration
- **postgres**: PostgreSQL database
- **sqlite**: SQLite database
- **google-drive**: Google Drive files
- **slack**: Slack messaging
- **brave-search**: Web search
- **puppeteer**: Web scraping
- **memory**: Vector database for knowledge

## Command Line Interface

```bash
# Run with data/context.md and data/config.json
python -m auto_agent_generator.cli                 # default Claude runtime
python -m auto_agent_generator.cli --llm codex     # use Codex runtime
python -m auto_agent_generator.cli --restart       # clear progress and start fresh

# Run with example configuration
python -m auto_agent_generator.cli example

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
    "max_iterations": 10,        // Maximum debug iterations
    "enable_testing": true,      // Run tests after generation
    "output_directory": "outputs/",  // Where to save generated code
    "verbose": true,             // Detailed output
    "save_progress": true        // Save execution progress
  }
}
```

### Model Parameters

```json
{
  "model_provider": {
    "parameters": {
      "temperature": 0.7,        // Creativity level (0.0-1.0)
      "max_tokens": 4096,        // Maximum response length
      "top_p": 1.0              // Nucleus sampling parameter
    }
  }
}
```

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

## Examples

### Example 1: Customer Support Automation

See `examples/example_context.md` - A complete customer support ticket processing system with:
- Ticket classification
- Knowledge base search
- Response generation
- Quality review

### Example 2: Data Pipeline

```markdown
# Data Pipeline System

Build a multi-agent system that:
1. Extracts data from multiple APIs
2. Transforms and validates data
3. Loads into data warehouse
4. Generates quality reports

Agents: Extractor, Transformer, Validator, Loader, Reporter
```

### Example 3: Content Creation

```markdown
# Content Creation Pipeline

Multi-agent system for:
1. Research topics and gather information
2. Generate article drafts
3. Fact-check content
4. Optimize for SEO
5. Publish to CMS

Agents: Researcher, Writer, FactChecker, SEOOptimizer, Publisher
```

## Troubleshooting

### Issue: LLM CLI not found
```bash
# Install Claude Code CLI (if using Claude runtime)
# Visit: https://code.claude.com

# OR install Codex CLI (if using Codex runtime)
# codex --version   # verify installation per Codex instructions
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
from src.mcp_manager import MCPManager

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
    "max_iterations": 20  // Increase for complex systems
  }
}
```

## Architecture

See `CODE_REFERENCE.md` for detailed architecture documentation including:
- Module descriptions
- Data flow diagrams
- Extension points
- API documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- Documentation: See `CODE_REFERENCE.md`
- Examples: See `examples/` folder
- Issues: Check `progress/errors.md` and logs
- Claude Code CLI: https://code.claude.com

## Acknowledgments

- Powered by Claude Code CLI
- Built with Anthropic Claude
- Supports LangChain, LangGraph, CrewAI, and Strands frameworks
- MCP protocol by Anthropic

---

**Built with ❤️ using Claude Code**
