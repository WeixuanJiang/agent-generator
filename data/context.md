# Business Context - Customer Support Automation

## Overview
Build an intelligent multi-agent system to automate customer support ticket processing and response generation.

## Business Flow

1. **Ticket Reception**: Receive customer support tickets from multiple channels (email, chat, web form)
2. **Classification**: Automatically classify tickets by category (technical, billing, general inquiry)
3. **Priority Assessment**: Determine urgency and priority level
4. **Information Gathering**: Search knowledge base and past tickets for relevant information
5. **Response Generation**: Generate personalized, context-aware responses
6. **Quality Review**: Review and validate responses before sending
7. **Ticket Resolution**: Send responses and mark tickets as resolved or escalate if needed

## Inputs

- **Customer Tickets**: Raw ticket data including:
  - Customer name and email
  - Subject and description
  - Timestamp
  - Channel (email/chat/form)
  - Customer history (optional)

- **Knowledge Base**: Collection of:
  - Product documentation
  - FAQ articles
  - Past resolved tickets
  - Company policies

## Expected Outputs

- **Classified Tickets**: Tickets with assigned categories and priority levels
- **Generated Responses**: Personalized response drafts with:
  - Greeting using customer name
  - Relevant information addressing the issue
  - Step-by-step solutions when applicable
  - Links to relevant documentation
  - Professional closing
- **Resolution Reports**: Summary of resolved tickets with statistics

## Data Sources

### Primary Data Sources
- **Ticket Database**: PostgreSQL database with customer tickets
  - Table: tickets
  - Columns: id, customer_email, subject, description, category, status, created_at

- **Knowledge Base**: Vector database (ChromaDB or Pinecone) with:
  - Product documentation embeddings
  - FAQ embeddings
  - Historical ticket resolutions

### Secondary Data Sources
- **Customer Profile API**: REST API providing customer history and preferences
- **Product Information API**: Real-time product status and updates

## Knowledge Base Requirements

- **Type**: Vector database for semantic search
- **Content Types**:
  - Product documentation (PDF, Markdown)
  - FAQ articles (HTML, Text)
  - Historical ticket resolutions
  - Company policies and guidelines

- **Update Frequency**: Daily batch updates for documentation, real-time for ticket resolutions

- **Search Capabilities**:
  - Semantic similarity search
  - Keyword filtering by category
  - Relevance ranking

## Required Tools and MCPs

### Essential MCPs
- **Database MCP** (postgres): For ticket database access
- **Vector Store MCP** (memory): For knowledge base search
- **Web Search MCP** (brave-search): For finding external resources when needed
- **File System MCP** (filesystem): For reading local documentation files

### Optional MCPs
- **Slack MCP**: For sending notifications to support team
- **Email MCP**: For direct email integration

### Custom Tools Needed
- **Ticket Classifier**: ML model or rule-based classifier for categorization
- **Priority Scorer**: Algorithm to determine ticket urgency
- **Response Validator**: Check response quality and completeness

## Constraints

1. **Response Time**: Generate responses within 30 seconds
2. **Quality**: Responses must be professional, grammatically correct, and empathetic
3. **Accuracy**: Must provide accurate information from knowledge base
4. **Privacy**: Do not share customer information across tickets
5. **Tone**: Maintain friendly but professional tone
6. **Length**: Responses should be concise (200-500 words typically)

## Preferences

### Coding Style
- Use type hints throughout
- Write comprehensive docstrings
- Follow PEP 8 standards
- Modular design with separate agents for each major function

### Error Handling
- Graceful degradation (if knowledge base unavailable, use general responses)
- Log all errors with context
- Retry failed operations with exponential backoff

### Logging
- INFO level for normal operations
- DEBUG level for development/troubleshooting
- Track ticket processing times and success rates

## Agent Architecture Suggestions

### Recommended Type: Hierarchical

A supervisor agent coordinating specialized sub-agents:

1. **Supervisor Agent**: Orchestrates the workflow
2. **Classifier Agent**: Categorizes and prioritizes tickets
3. **Research Agent**: Searches knowledge base and gathers information
4. **Writer Agent**: Generates response drafts
5. **Reviewer Agent**: Validates and improves responses

### Communication Pattern
- Sequential processing with supervisor coordination
- Research agent may run parallel searches
- Reviewer agent has veto power to request rewrites

## Additional Requirements

- Generate comprehensive unit tests
- Include example tickets for testing
- Add monitoring and metrics collection
- Support for batch processing multiple tickets
- Configuration file for customizing behavior
- README with setup and usage instructions

## Success Criteria

1. Successfully process at least 100 tickets per hour
2. Response quality score > 4.0/5.0 (human evaluation)
3. 90%+ of responses require no human edits
4. Average response time < 20 seconds
5. Properly escalate complex issues (< 10% false escalations)
