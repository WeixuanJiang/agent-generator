"""
Context Parser Module

Parses free-form context.md and extracts structured information using Claude Code CLI.
"""

import os
import json
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path


PROJECT_ROOT = Path.cwd()

class ContextParser:
    """
    Parse free-form business context and extract structured information.

    Uses Claude Code CLI to intelligently analyze unstructured text and extract:
    - Business flow and logic
    - Input/output requirements
    - Data sources and integrations
    - Knowledge base needs
    - Required MCPs/tools
    """

    def __init__(self, context_file: str = str(PROJECT_ROOT / "data" / "context.md")):
        """
        Initialize ContextParser.

        Args:
            context_file: Path to context.md file
        """
        self.context_file = context_file
        self.context_content: Optional[str] = None
        self.parsed_context: Optional[Dict[str, Any]] = None

    def read_context(self) -> str:
        """
        Read context.md file.

        Returns:
            Content of context.md

        Raises:
            FileNotFoundError: If context.md doesn't exist
        """
        if not os.path.exists(self.context_file):
            raise FileNotFoundError(
                f"Context file not found: {self.context_file}\n"
                f"Please create the file with your business requirements."
            )

        with open(self.context_file, 'r', encoding='utf-8') as f:
            self.context_content = f.read()

        return self.context_content

    def parse_context(self) -> Dict[str, Any]:
        """
        Parse context using Claude Code CLI to extract structured information.

        Returns:
            Dictionary with extracted context information:
            {
                "business_flow": str,
                "inputs": List[Dict],
                "outputs": List[Dict],
                "data_sources": List[Dict],
                "knowledge_base": Dict,
                "required_mcps": List[str],
                "constraints": List[str],
                "preferences": Dict
            }
        """
        if self.context_content is None:
            self.read_context()

        # Use Claude Code CLI to analyze and extract structured information
        prompt = self._build_extraction_prompt()

        try:
            result = self._execute_claude(prompt)
            self.parsed_context = self._parse_claude_response(result)
        except Exception:
            self.parsed_context = self._get_default_context()

        return self.parsed_context

    def _build_extraction_prompt(self) -> str:
        """
        Build prompt for Claude Code CLI to extract structured information.

        Returns:
            Extraction prompt
        """
        prompt = f"""
Analyze the following business context and extract structured information in JSON format.

CONTEXT:
{self.context_content}

Extract the following information and return ONLY valid JSON (no markdown, no explanations):
{{
    "business_flow": "Detailed description of the business process and workflow",
    "inputs": [
        {{
            "name": "input_name",
            "type": "data_type",
            "description": "description",
            "required": true/false,
            "source": "where it comes from"
        }}
    ],
    "outputs": [
        {{
            "name": "output_name",
            "type": "data_type",
            "description": "description",
            "format": "output format"
        }}
    ],
    "data_sources": [
        {{
            "name": "source_name",
            "type": "database|api|file|stream",
            "connection": "connection details or requirements",
            "operations": ["read", "write", "update"]
        }}
    ],
    "knowledge_base": {{
        "required": true/false,
        "type": "vector_db|graph_db|document_store",
        "content_types": ["documents", "embeddings", "etc"],
        "update_frequency": "real-time|batch|manual"
    }},
    "required_mcps": [
        "mcp_name_1",
        "mcp_name_2"
    ],
    "required_tools": [
        "tool_name_1",
        "tool_name_2"
    ],
    "constraints": [
        "constraint 1",
        "constraint 2"
    ],
    "preferences": {{
        "coding_style": "preference",
        "error_handling": "approach",
        "logging_level": "level"
    }},
    "agent_architecture": {{
        "suggested_type": "sequential|hierarchical|collaborative",
        "number_of_agents": 0,
        "agent_roles": ["role1", "role2"]
    }}
}}

Return ONLY the JSON object, no other text.
"""
        return prompt

    def _execute_claude(self, prompt: str) -> str:
        """
        Execute Claude Code CLI command in non-interactive mode.

        Args:
            prompt: Prompt to execute

        Returns:
            Claude Code CLI output
        """
        import platform
        import tempfile

        # On Windows, use shell=True; on Unix-like systems, don't use shell
        is_windows = platform.system() == 'Windows'
        claude_cmd = 'claude.cmd' if is_windows else 'claude'

        try:
            # Write prompt to temp file and pipe it
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(prompt)
                temp_file = f.name

            # Build command
            cmd_parts = [claude_cmd, '-p', '--max-turns', '5', '--output-format', 'text']

            # Pipe the temp file to claude
            if is_windows:
                cmd = f'type "{temp_file}" | {" ".join(cmd_parts)}'
            else:
                cmd = f'cat "{temp_file}" | {" ".join(cmd_parts)}'

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                shell=True,
                encoding="utf-8",
                errors="replace"
            )

            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass

            if result.returncode != 0:
                raise RuntimeError(
                    f"Claude Code execution failed: {result.stderr}"
                )
            out = result.stdout or ""
            print(out)
            return out

        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude Code execution timed out after 5 minutes")

        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI not found. Please ensure Claude Code is installed.\n"
                "Install from: https://code.claude.com"
            )

    def _parse_claude_response(self, response: str) -> Dict[str, Any]:
        """
        Parse Claude Code CLI response to extract JSON.

        Args:
            response: Raw Claude Code output

        Returns:
            Parsed context dictionary
        """
        # Try to find JSON in the response
        try:
            # First, try direct JSON parsing
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re

            # Look for JSON in code blocks
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            matches = re.findall(json_pattern, response, re.DOTALL)

            if matches:
                try:
                    return json.loads(matches[0])
                except json.JSONDecodeError:
                    pass

            # Try to find any JSON object in the response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)

            for match in matches:
                try:
                    parsed = json.loads(match)
                    if 'business_flow' in parsed:  # Validate it's our expected structure
                        return parsed
                except json.JSONDecodeError:
                    continue

            # If all parsing fails, return a default structure
            print(f"Warning: Could not parse JSON from response. Using default structure.")
            return self._get_default_context()

    def _get_default_context(self) -> Dict[str, Any]:
        """
        Get default context structure when parsing fails.

        Returns:
            Default context dictionary
        """
        return {
            "business_flow": "Unable to parse context automatically. Please review context.md",
            "inputs": [],
            "outputs": [],
            "data_sources": [],
            "knowledge_base": {
                "required": False,
                "type": "vector_db",
                "content_types": [],
                "update_frequency": "manual"
            },
            "required_mcps": [],
            "required_tools": [],
            "constraints": [],
            "preferences": {},
            "agent_architecture": {
                "suggested_type": "sequential",
                "number_of_agents": 1,
                "agent_roles": ["default_agent"]
            }
        }

    def extract_business_flow(self) -> str:
        """
        Extract business flow description.

        Returns:
            Business flow description
        """
        if self.parsed_context is None:
            self.parse_context()

        return self.parsed_context.get("business_flow", "")

    def extract_io_requirements(self) -> Dict[str, List[Dict]]:
        """
        Extract input and output requirements.

        Returns:
            Dictionary with 'inputs' and 'outputs' keys
        """
        if self.parsed_context is None:
            self.parse_context()

        return {
            "inputs": self.parsed_context.get("inputs", []),
            "outputs": self.parsed_context.get("outputs", [])
        }

    def extract_data_sources(self) -> List[Dict]:
        """
        Extract data source requirements.

        Returns:
            List of data source dictionaries
        """
        if self.parsed_context is None:
            self.parse_context()

        return self.parsed_context.get("data_sources", [])

    def extract_knowledge_requirements(self) -> Dict:
        """
        Extract knowledge base requirements.

        Returns:
            Knowledge base configuration dictionary
        """
        if self.parsed_context is None:
            self.parse_context()

        return self.parsed_context.get("knowledge_base", {})

    def extract_mcp_requirements(self) -> List[str]:
        """
        Extract required MCP servers.

        Returns:
            List of required MCP names
        """
        if self.parsed_context is None:
            self.parse_context()

        mcps = self.parsed_context.get("required_mcps", [])
        tools = self.parsed_context.get("required_tools", [])

        # Combine MCPs and tools
        return list(set(mcps + tools))

    def get_agent_architecture_suggestions(self) -> Dict:
        """
        Get suggested agent architecture from context analysis.

        Returns:
            Agent architecture suggestions
        """
        if self.parsed_context is None:
            self.parse_context()

        return self.parsed_context.get("agent_architecture", {})

    def save_parsed_context(self, output_file: str = str(PROJECT_ROOT / "progress" / "parsed_context.json")):
        """
        Save parsed context to JSON file.

        Args:
            output_file: Output file path
        """
        if self.parsed_context is None:
            self.parse_context()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.parsed_context, f, indent=2)

    def get_full_context(self) -> Dict[str, Any]:
        """
        Get the complete parsed context.

        Returns:
            Full parsed context dictionary
        """
        if self.parsed_context is None:
            self.parse_context()

        return self.parsed_context

    def extract_project_name(self) -> str:
        """
        Extract project name from context title for use as folder name.
        Uses Claude Code CLI to intelligently extract and sanitize the name.

        Returns:
            Sanitized project name suitable for filesystem use
        """
        if self.context_content is None:
            self.read_context()

        prompt = f"""
Extract a filesystem-safe project name from the following context.

CONTEXT:
{self.context_content[:500]}

TASK:
Find the main project title (usually in the first H1 heading), remove common prefixes like "Business Context -", "Context -", "Project -", then convert to a filesystem-safe slug (lowercase, spaces/dashes to underscores, only alphanumerics and underscores).

Return ONLY the sanitized project name, nothing else.
If no clear title found, return: generated_agent

Example: "# Business Context - Customer Support Automation" -> customer_support_automation
"""

        try:
            result = self._execute_claude(prompt)
            project_name = result.strip().lower().strip('"\'` \n\r')
            # Validate filesystem-safe
            if project_name and all(c.isalnum() or c == '_' for c in project_name):
                return project_name
        except Exception:
            pass

        return 'generated_agent'


# Example usage
if __name__ == "__main__":
    parser = ContextParser("data/context.md")

    try:
        context = parser.parse_context()
        print("Parsed Context:")
        print(json.dumps(context, indent=2))

        # Save to file
        parser.save_parsed_context()
        print("\nParsed context saved to progress/parsed_context.json")

    except Exception as e:
        print(f"Error parsing context: {e}")
