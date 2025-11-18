"""
MCP Manager Module

Auto-detect, configure, and set up Model Context Protocol (MCP) servers.
"""

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional


class MCPManager:
    """
    Manage MCP server configuration and setup.

    Handles:
    - Gathering context to propose required MCPs via LLM
    - Generating MCP server configurations
    - Setting up and validating MCP servers
    - Managing custom MCPs
    """

    def __init__(self, llm_interface=None, verbose: bool = True):
        """
        Initialize MCPManager.

        Args:
            llm_interface: LLM interface for intelligent analysis (optional)
            verbose: Enable verbose output
        """
        self.llm_interface = llm_interface
        self.verbose = verbose
        self.required_mcps: List[str] = []
        self.custom_mcps: Dict[str, Dict] = {}
        self.mcp_plan: Dict[str, Any] = {"mcps": []}

    def analyze_requirements(self, context: Dict[str, Any], fixed_mcps: Optional[List[str]] = None) -> List[str]:
        """
        Analyze context to determine required MCPs using the LLM (or fallback heuristic).

        Args:
            context: Parsed context from ContextParser
            fixed_mcps: Optional list of MCP names to force/include

        Returns:
            List of required MCP names
        """
        plan = self._generate_mcp_plan(context, fixed_mcps=fixed_mcps)
        self.mcp_plan = plan

        self.required_mcps = [m.get("name") for m in plan.get("mcps", []) if m.get("name")]

        if self.verbose:
            detected = ", ".join(self.required_mcps) if self.required_mcps else "none"
            print(f"Detected required MCPs: {detected}")

        return self.required_mcps

    def set_required_mcps(self, mcps: List[str]):
        """Force a fixed list of MCPs and ensure they exist in the plan."""
        filtered = [name for name in mcps if name]
        self.required_mcps = filtered

        existing = {m.get("name") for m in self.mcp_plan.get("mcps", []) if m.get("name")}
        for name in filtered:
            if name not in existing:
                self.mcp_plan.setdefault("mcps", []).append(
                    {
                        "name": name,
                        "purpose": "User-provided MCP",
                        "command": "",
                        "args": [],
                        "env": {},
                    }
                )

        if self.verbose:
            print(f"Using fixed MCP list: {', '.join(self.required_mcps) if self.required_mcps else 'none'}")

    def generate_mcp_config(self, mcp_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate MCP server configuration.

        Args:
            mcp_list: List of MCP names (uses required_mcps if None)

        Returns:
            MCP configuration dictionary
        """
        mcps = mcp_list or self.required_mcps or [m.get("name") for m in self.mcp_plan.get("mcps", []) if m.get("name")]

        config = {"mcpServers": {}}
        plan_by_name = {m.get("name"): m for m in self.mcp_plan.get("mcps", []) if m.get("name")}

        for mcp_name in mcps:
            if mcp_name in self.custom_mcps:
                config["mcpServers"][mcp_name] = deepcopy(self.custom_mcps[mcp_name])
                continue

            mcp_info = plan_by_name.get(mcp_name)
            if not mcp_info:
                if self.verbose:
                    print(f"Warning: Unknown MCP '{mcp_name}', skipping...")
                continue

            server_config = {
                "command": mcp_info.get("command"),
                "args": mcp_info.get("args", []),
                "env": mcp_info.get("env", {}),
            }

            config["mcpServers"][mcp_name] = server_config

        return config

    def save_mcp_config(self, config: Optional[Dict] = None, output_file: str = "mcp_config.json"):
        """
        Save MCP configuration to file.

        Args:
            config: MCP configuration (generates if None)
            output_file: Output file path
        """
        if config is None:
            config = self.generate_mcp_config()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        if self.verbose:
            print(f"MCP configuration saved to: {output_file}")

    def setup_mcp_servers(self) -> Dict[str, bool]:
        """
        Prepare MCP servers based on the generated configuration.

        Returns:
            Dictionary mapping MCP name to setup success status
        """
        results: Dict[str, bool] = {}

        if self.verbose:
            print("\nSetting up MCP servers...")

        config = self.generate_mcp_config()
        for mcp_name, server in config.get("mcpServers", {}).items():
            ready = bool(server.get("command"))
            results[mcp_name] = ready
            if self.verbose:
                status = "OK" if ready else "MISSING COMMAND"
                print(f"  {status} {mcp_name}")

        return results

    def validate_mcps(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        Validate that required MCPs have runnable configuration.

        Args:
            config: Optional pre-generated MCP config

        Returns:
            Dictionary mapping MCP name to availability status
        """
        results: Dict[str, bool] = {}

        if self.verbose:
            print("\nValidating MCP servers...")

        config = config or self.generate_mcp_config()
        for mcp_name, server in config.get("mcpServers", {}).items():
            available = bool(server.get("command"))
            results[mcp_name] = available

            if self.verbose:
                status = "OK" if available else "MISSING COMMAND"
                print(f"  {status} {mcp_name}")

        return results

    def list_available_mcps(self) -> List[str]:
        """List all MCPs currently in the plan or custom set."""
        plan_names = [m.get("name") for m in self.mcp_plan.get("mcps", []) if m.get("name")]
        return sorted(set(plan_names + list(self.custom_mcps.keys())))

    def get_mcp_info(self, mcp_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an MCP from the current plan/custom set.

        Args:
            mcp_name: Name of the MCP

        Returns:
            MCP information or None
        """
        for mcp in self.mcp_plan.get("mcps", []):
            if mcp.get("name") == mcp_name:
                return mcp
        if mcp_name in self.custom_mcps:
            return self.custom_mcps[mcp_name]
        return None

    def add_custom_mcp(self, name: str, config: Dict[str, Any]):
        """
        Add a custom MCP configuration.

        Args:
            name: Custom MCP name
            config: MCP server configuration
        """
        self.custom_mcps[name] = config

        if name not in self.required_mcps:
            self.required_mcps.append(name)

        self.mcp_plan.setdefault("mcps", []).append({"name": name, **config})

        if self.verbose:
            print(f"Added custom MCP: {name}")

    def generate_env_template(self) -> str:
        """
        Generate .env template for required MCPs based on the LLM plan/custom configs.

        Returns:
            .env template content
        """
        env_entries = set()

        for key, value in self._iter_env_entries():
            placeholder = value if isinstance(value, str) else ""
            env_entries.add(f"{key}={placeholder}")

        template = "# MCP Server Environment Variables\n\n"
        template += "\n".join(sorted(env_entries))

        return template

    def save_env_template(self, output_file: str = ".env.mcp.template"):
        """
        Save .env template to file.

        Args:
            output_file: Output file path
        """
        template = self.generate_env_template()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(template)

        if self.verbose:
            print(f"Environment template saved to: {output_file}")

    def get_setup_summary(self) -> str:
        """
        Get MCP setup summary based on the current plan.

        Returns:
            Formatted setup summary
        """
        summary = "# MCP Setup Summary\n\n"
        summary += f"## Required MCPs ({len(self.required_mcps)})\n\n"

        for mcp_name in self.required_mcps:
            info = self.get_mcp_info(mcp_name)
            if info:
                summary += f"- **{mcp_name}**: {info.get('purpose', 'No purpose provided')}\n"
                command = info.get("command")
                if command:
                    summary += f"  - Command: `{command}`\n"
                args = info.get("args")
                if args:
                    summary += f"  - Args: {args}\n"
                env = info.get("env")
                if env:
                    summary += f"  - Env: {env}\n"
            else:
                summary += f"- **{mcp_name}**: Custom MCP\n"

        return summary

    def _iter_env_entries(self):
        plan_mcps = self.mcp_plan.get("mcps", [])
        for mcp in plan_mcps:
            env = mcp.get("env") or {}
            for key, value in env.items():
                yield key, value

        for _, config in self.custom_mcps.items():
            env = config.get("env") or {}
            for key, value in env.items():
                yield key, value

    def _generate_mcp_plan(self, context: Dict[str, Any], fixed_mcps: Optional[List[str]] = None) -> Dict[str, Any]:
        """Use the LLM to propose MCP servers; fallback to simple heuristics."""
        fixed_mcps = fixed_mcps or []
        plan: Dict[str, Any] = {"mcps": []}

        if self.llm_interface:
            prompt = self._build_mcp_prompt(context, fixed_mcps)
            raw_response = self.llm_interface.execute_with_progress_retry(prompt, max_turns=20)
            parsed = self.llm_interface.extract_json(raw_response)

            if isinstance(parsed, dict) and parsed.get("mcps"):
                return parsed

            if self.verbose:
                print("Warning: LLM did not return structured MCP plan; falling back to heuristics.")

        # Heuristic fallback: derive MCP names from data sources and fixed list
        seen = set()
        for source in context.get("data_sources", []):
            name = (source.get("type") or "").lower()
            if name and name not in seen:
                plan["mcps"].append(
                    {
                        "name": name,
                        "purpose": f"MCP for {name}",
                        "command": "",
                        "args": [],
                        "env": {},
                    }
                )
                seen.add(name)

        for name in fixed_mcps:
            if name and name not in seen:
                plan["mcps"].append(
                    {
                        "name": name,
                        "purpose": "User-specified MCP",
                        "command": "",
                        "args": [],
                        "env": {},
                    }
                )
                seen.add(name)

        if not plan["mcps"]:
            plan["mcps"].append(
                {
                    "name": "filesystem",
                    "purpose": "Basic file access",
                    "command": "",
                    "args": [],
                    "env": {},
                }
            )

        return plan

    def _build_mcp_prompt(self, context: Dict[str, Any], fixed_mcps: List[str]) -> str:
        """Build the LLM prompt for generating MCP definitions."""
        try:
            context_json = json.dumps(context, indent=2, default=str)
        except TypeError:
            context_json = str(context)

        context_snippet = context_json[:8000]
        fixed_label = ", ".join(fixed_mcps) if fixed_mcps else "(none)"

        prompt = f"""
You are designing Model Context Protocol (MCP) servers for an auto agent generator.
Use the provided project context to propose the minimal set of MCP servers needed.
Return ONLY valid JSON matching this schema:
{{
  "mcps": [
    {{
      "name": "<mcp_name>",
      "purpose": "<what it does>",
      "command": "<binary or entrypoint>",
      "args": ["<arg1>", "<arg2>"],
      "env": {{"ENV_NAME": "${{ENV_NAME}}"}},
      "notes": "<optional notes>",
      "sources": ["<optional data source references>"]
    }}
  ]
}}
- Include only MCPs required for the business flow and data sources.
- If fixed MCP names must be included, ensure they appear: {fixed_label}.
- Commands and args must be directly runnable (no placeholders beyond env vars in "env").
- Prefer standard MCP servers when appropriate, but choose based on the context not on a fixed catalog.
- Do not add extra fields and do not wrap the JSON in code fences.

Context:
{context_snippet}
"""
        return prompt


# Example usage
if __name__ == "__main__":
    manager = MCPManager(verbose=True)

    # Example context
    context = {
        "business_flow": "Fetch data from GitHub and store in PostgreSQL database",
        "data_sources": [
            {"type": "github"},
            {"type": "postgres"}
        ],
        "required_mcps": []
    }

    # Analyze requirements
    required_mcps = manager.analyze_requirements(context)

    # Generate configuration
    config = manager.generate_mcp_config()
    print("\nGenerated MCP Config:")
    print(json.dumps(config, indent=2))

    # Save configuration
    manager.save_mcp_config()

    # Generate environment template
    manager.save_env_template()

    # Print summary
    print("\n" + manager.get_setup_summary())
