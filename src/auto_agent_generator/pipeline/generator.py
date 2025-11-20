"""
Agent Generator Module

Main orchestrator for autonomous multi-agent system generation.
Implements full autonomous iteration: generate → test → debug → fix → repeat
"""

import os
import json
import subprocess
import sys
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


class AgentGenerator:
    """
    Generate multi-agent systems autonomously with iterative refinement.

    Workflow:
    1. Analyze context and plan architecture
    2. Generate initial agent system code
    3. Validate syntax and imports
    4. Generate and run test cases
    5. Debug and fix errors
    6. Iterate until perfect
    """

    def __init__(
        self,
        context_parser,
        config_parser,
        progress_manager,
        llm_interface,
        mcp_manager,
        verbose: bool = True
    ):
        """
        Initialize AgentGenerator.

        Args:
            context_parser: ContextParser instance
            config_parser: ConfigParser instance
            progress_manager: ProgressManager instance
            llm_interface: ClaudeInterface or CodexInterface instance
            mcp_manager: MCPManager instance
            verbose: Enable verbose output
        """
        self.context_parser = context_parser
        self.config_parser = config_parser
        self.progress_manager = progress_manager
        self.llm = llm_interface
        self.mcp_manager = mcp_manager
        self.verbose = verbose

        self.context: Optional[Dict] = None
        self.framework: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.max_iterations: int = 10
        self.current_iteration: int = 0
        self.current_stage: Optional[str] = None
        self.completed_stages: List[str] = []
        self._pipeline_valid: bool = False
        self.generation_plan: Optional[Dict] = None

    def generate(self, resume: bool = False) -> Dict[str, Any]:
        """
        Main generation entry point - fully autonomous execution.

        Returns:
            Generation results dictionary
        """
        if self.verbose:
            print("\n" + "="*80)
            print("AUTO AGENT GENERATOR - AUTONOMOUS EXECUTION")
            print("="*80 + "\n")

        try:
            # Ensure run state is initialized regardless of resume stage
            self.context = self.context or self.context_parser.get_full_context()

            # Determine resume stage and restore state
            start_stage = None
            checkpoint_data = None
            if resume:
                checkpoint_data = self.progress_manager.load_checkpoint("pipeline")
                if checkpoint_data:
                    start_stage = checkpoint_data.get("next_stage")
                    # Restore state from checkpoint
                    self.completed_stages = checkpoint_data.get("completed_stages", [])
                    self.current_iteration = checkpoint_data.get("iterations_completed", 0)
                    self.output_dir = checkpoint_data.get("output_dir")
                    self.framework = checkpoint_data.get("framework")
                    self.generation_plan = checkpoint_data.get("generation_plan")

                    if self.verbose:
                        print(f"\n[RESUMING] from stage: {start_stage}")
                        if self.completed_stages:
                            print(f"[COMPLETED] stages: {', '.join(self.completed_stages)}")
                        print(f"[OUTPUT] directory: {self.output_dir}\n")

            # Initialize if not resuming or no checkpoint found
            if not self.framework:
                self.framework = self.config_parser.get_framework()

            if not self.output_dir:
                # Create project-specific output directory
                base_output_dir = self.config_parser.get_output_directory()
                project_name = self.context_parser.extract_project_name()
                self.output_dir = os.path.join(base_output_dir, project_name)

            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            if not self.max_iterations:
                self.max_iterations = self.config_parser.get_max_iterations()

            if self.verbose and not resume:
                print(f"Project: {os.path.basename(self.output_dir)}")
                print(f"Output directory: {self.output_dir}")

            # Phase 1: Analyze and Plan
            if "analysis_and_planning" not in self.completed_stages:
                if not start_stage or start_stage == "analysis_and_planning":
                    self._enter_stage("analysis_and_planning")
                    self._phase_analyze_and_plan()
                    self._complete_stage("analysis_and_planning", next_stage="mcp_setup")
            else:
                if self.verbose:
                    print("[SKIPPING] analysis_and_planning (already completed)")

            # Phase 2: Setup MCPs
            if "mcp_setup" not in self.completed_stages:
                if not start_stage or start_stage in ("mcp_setup", "analysis_and_planning"):
                    self._enter_stage("mcp_setup")
                    self._phase_setup_mcps()
                    self._complete_stage("mcp_setup", next_stage="generation_core_models")
            else:
                if self.verbose:
                    print("[SKIPPING] mcp_setup (already completed)")

            # Phase 3: Generate Initial Code (staged)
            # Define all generation stages in order
            
            # Use dynamic plan if available, otherwise fallback to default
            if self.generation_plan and "files" in self.generation_plan:
                files_plan = self.generation_plan["files"]
                generation_stages = [
                    ("generation_core_models", "Core Config and Models", files_plan.get("core", [])),
                    ("generation_agents", "Agents", files_plan.get("agents", [])),
                    ("generation_supervisor_workflow", "Supervisor and Workflow", files_plan.get("workflows", [])),
                    ("generation_tools_mcps", "Tools and MCP Clients", files_plan.get("tools", [])),
                    ("generation_tests_docs", "Tests and Docs", files_plan.get("tests", []))
                ]
                # Filter out empty stages
                generation_stages = [s for s in generation_stages if s[2]]
            else:
                # Fallback to hardcoded list if no plan available
                generation_stages = [
                    ("generation_core_models", "Core Config and Models", ["config/settings.py", "models/ticket.py"]),
                    ("generation_agents", "Agents", ["agents/classifier.py", "agents/researcher.py", "agents/writer.py", "agents/reviewer.py"]),
                    ("generation_supervisor_workflow", "Supervisor and Workflow", ["agents/supervisor.py", "workflows/ticket_workflow.py"]),
                    ("generation_tools_mcps", "Tools and MCP Clients", ["mcps/postgres_client.py", "mcps/vector_db_client.py", "tools/external_apis.py"]),
                    ("generation_tests_docs", "Tests and Docs", ["tests/test_system.py", "requirements.txt", "README.md", ".env.example"])
                ]

            # Determine which generation stage to start from
            for i, (stage_name, stage_desc, files) in enumerate(generation_stages):
                # Determine next stage
                next_stage = generation_stages[i + 1][0] if i + 1 < len(generation_stages) else "iterative_refinement"

                # Skip if this stage is already completed
                if stage_name in self.completed_stages:
                    if self.verbose:
                        print(f"[SKIPPING] {stage_name} (already completed)")
                    continue

                # Skip if we're resuming and haven't reached the resume stage yet
                if start_stage and start_stage != stage_name:
                    # Check if start_stage is a later generation stage
                    start_stage_names = [s[0] for s in generation_stages]
                    if start_stage in start_stage_names:
                        start_idx = start_stage_names.index(start_stage)
                        current_idx = i
                        if current_idx < start_idx:
                            if self.verbose:
                                print(f"[SKIPPING] {stage_name} (before resume point)")
                            continue

                # Run this stage
                self._enter_stage(stage_name)
                self._phase_generate_initial_code_stage(stage_desc, files)
                self._complete_stage(stage_name, next_stage=next_stage)

            # Phase 4: Iterative Refinement
            if "iterative_refinement" not in self.completed_stages:
                if not start_stage or start_stage in ("iterative_refinement", "generation_tests_docs"):
                    self._enter_stage("iterative_refinement")
                    self._phase_iterative_refinement()
                    self._complete_stage("iterative_refinement", next_stage="finalize")
            else:
                if self.verbose:
                    print("[SKIPPING] iterative_refinement (already completed)")

            # Gate finalization on validation and tests passing
            gate_validation = self._validate_code()
            if not gate_validation.get("valid"):
                raise RuntimeError("Validation failed; not finalizing")
            if self.config_parser.is_testing_enabled():
                gate_tests = self._run_tests()
                if not gate_tests.get("success"):
                    raise RuntimeError("Tests failed; not finalizing")
            self._pipeline_valid = True

            # Phase 5: Finalize
            self._enter_stage("finalize")
            results = self._phase_finalize()
            self._complete_stage("finalize", next_stage="done")

            return results

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            self.progress_manager.save_error(error_msg)
            # Persist failure checkpoint
            self.progress_manager.save_checkpoint("pipeline", {
                "current_stage": self.current_stage,
                "next_stage": self.current_stage,
                "completed_stages": self.completed_stages,
                "iterations_completed": self.current_iteration,
                "output_dir": self.output_dir,
                "framework": self.framework,
                "generation_plan": self.generation_plan,
                "last_error": error_msg
            })
            self.progress_manager.save_stage_status(self.current_stage or "unknown", "failed", error_msg)

            if self.verbose:
                print(f"\nERROR: {error_msg}\n")

            return {
                "success": False,
                "error": error_msg,
                "iteration": self.current_iteration
            }

    def _enter_stage(self, stage: str):
        self.current_stage = stage
        self.progress_manager.save_stage_status(stage, "started")
        self.progress_manager.save_checkpoint("pipeline", {
            "current_stage": stage,
            "next_stage": stage,
            "completed_stages": self.completed_stages,
            "iterations_completed": self.current_iteration,
            "output_dir": self.output_dir,
            "framework": self.framework,
            "generation_plan": self.generation_plan
        })

    def _complete_stage(self, stage: str, next_stage: str):
        self.completed_stages.append(stage)
        self.progress_manager.save_stage_status(stage, "success")
        self.progress_manager.save_checkpoint("pipeline", {
            "current_stage": stage,
            "next_stage": next_stage,
            "completed_stages": self.completed_stages,
            "iterations_completed": self.current_iteration,
            "output_dir": self.output_dir,
            "framework": self.framework,
            "generation_plan": self.generation_plan
        })

    def _phase_generate_initial_code_stage(self, stage_name: str, target_files: List[str]):
        prompt = self._build_stage_generation_prompt(stage_name, target_files)
        response = self.llm.execute_with_progress_retry(prompt, retries=2, backoff=1.0, max_turns=25)
        python_code = self.llm.extract_python_code(response)
        if not python_code:
            blocks = self.llm.extract_code_blocks(response)
            candidates = []
            for b in blocks:
                code = b.get("code", "")
                if "filename:" in code.lower():
                    candidates.append(code)
            if not candidates:
                candidates = self.llm.extract_code_by_filename_comments(response)
            python_code = candidates
        if python_code:
            complete_files = []
            for code in python_code:
                if self._is_code_complete(code):
                    complete_files.append(code)
                else:
                    fixed = self._complete_file_via_claude(code)
                    if fixed:
                        complete_files.append(fixed)
            if complete_files:
                self._save_generated_code(complete_files)
                self.progress_manager.save_generation(
                    f"Generated {len(complete_files)} files in stage: {stage_name}",
                    f"Initial Code Generation - {stage_name}"
                )
                return
        raise RuntimeError(f"No code blocks detected for stage: {stage_name}")


    def _phase_analyze_and_plan(self):
        """Phase 1: Analyze context and plan architecture."""
        if self.verbose:
            print("Phase 1: Analyzing context and planning architecture...")

        # Parse context (output_dir already set in generate())
        if not self.context:
            self.context = self.context_parser.get_full_context()
        if not self.framework:
            self.framework = self.config_parser.get_framework()
        if not self.max_iterations:
            self.max_iterations = self.config_parser.get_max_iterations()

        # Build analysis prompt
        prompt = self._build_analysis_prompt()

        response = self.llm.execute_with_retry(prompt, retries=2, backoff=1.0)
        thinking = self.llm.extract_thinking(response) or response
        self.progress_manager.save_thinking(thinking, "Context Analysis")
        
        # Parse the JSON plan
        try:
            self.generation_plan = self.llm.extract_json(response)
            if not self.generation_plan:
                # Try to find JSON in code blocks if extract_json failed
                code_blocks = self.llm.extract_code_blocks(response)
                for block in code_blocks:
                    if block["language"] == "json":
                        self.generation_plan = json.loads(block["code"])
                        break
            
            if self.generation_plan:
                self.progress_manager.save_planning(json.dumps(self.generation_plan, indent=2), "Architecture Planning")
            else:
                # Fallback if JSON parsing fails completely
                planning = self._extract_planning_from_response(response)
                self.progress_manager.save_planning(planning, "Architecture Planning")
                if self.verbose:
                    print("  Warning: Could not parse structured plan, using fallback.")
        except Exception as e:
            if self.verbose:
                print(f"  Warning: Error parsing plan: {e}")
            planning = self._extract_planning_from_response(response)
            self.progress_manager.save_planning(planning, "Architecture Planning")

        if self.verbose:
            print("  Analysis and planning complete\n")

    def _build_analysis_prompt(self) -> str:
        """Build prompt for context analysis and planning."""
        context_json = json.dumps(self.context, indent=2)

        prompt = f"""
Analyze the following business context and create a detailed plan for a multi-agent system.

CONTEXT:
{context_json}

FRAMEWORK: {self.framework}
MODEL PROVIDER: {self.config_parser.get_provider_name()}

TASK:
1. Analyze the business requirements
2. Determine the optimal agent architecture (sequential/hierarchical/collaborative)
3. Define each agent's role and responsibilities
4. Plan the data flow between agents
5. Identify required tools and integrations
6. Design the testing strategy
7. List ALL files that need to be generated for this system

Return your analysis and plan in the following JSON format (no markdown, no explanations):

{{
    "architecture": {{
        "type": "sequential|hierarchical|collaborative",
        "description": "..."
    }},
    "agents": [
        {{
            "name": "agent_name",
            "role": "...",
            "responsibilities": "..."
        }}
    ],
    "files": {{
        "core": ["config/settings.py", "models/ticket.py"],
        "agents": ["agents/classifier.py", "agents/researcher.py"],
        "workflows": ["workflows/ticket_workflow.py"],
        "tools": ["mcps/postgres_client.py"],
        "tests": ["tests/test_system.py", "requirements.txt", "README.md"]
    }},
    "data_flow": "...",
    "testing_strategy": "..."
}}
"""
        return prompt

    def _extract_planning_from_response(self, response: str) -> str:
        """Extract planning content from response."""
        # Look for architecture plan section
        if "# ARCHITECTURE PLAN" in response:
            return response.split("# ARCHITECTURE PLAN")[1].strip()
        return response

    def _phase_setup_mcps(self):
        """Phase 2: Setup required MCP servers."""
        if self.verbose:
            print("Phase 2: Setting up MCP servers...")

        # Analyze MCP requirements via LLM (or fixed list if provided)
        fixed_mcps = self.config_parser.get_fixed_mcps()
        required_mcps = self.mcp_manager.analyze_requirements(self.context, fixed_mcps=fixed_mcps)

        # Generate and save MCP configuration
        mcp_config = self.mcp_manager.generate_mcp_config()
        os.makedirs(self.output_dir, exist_ok=True)
        self.mcp_manager.save_mcp_config(mcp_config, os.path.join(self.output_dir, "mcp_config.json"))

        # Generate environment template
        self.mcp_manager.save_env_template(os.path.join(self.output_dir, ".env.mcp.template"))

        # Validate MCPs
        validation_results = self.mcp_manager.validate_mcps(mcp_config)

        # Save planning
        mcp_summary = self.mcp_manager.get_setup_summary()
        self.progress_manager.save_planning(mcp_summary, "MCP Setup")
        
        # Inject custom MCP files into generation plan if needed
        if self.generation_plan and "files" in self.generation_plan:
            tools_files = self.generation_plan["files"].get("tools", [])
            
            # Check for custom MCPs in the plan
            for mcp in self.mcp_manager.mcp_plan.get("mcps", []):
                impl_file = mcp.get("implementation_file")
                if impl_file and impl_file not in tools_files:
                    if self.verbose:
                        print(f"  Injecting custom MCP file into generation plan: {impl_file}")
                    tools_files.append(impl_file)
            
            # Update the plan
            self.generation_plan["files"]["tools"] = tools_files
            
            # Update checkpoint with modified plan
            self.progress_manager.save_checkpoint("pipeline", {
                "current_stage": self.current_stage,
                "next_stage": "generation_core_models",
                "completed_stages": self.completed_stages,
                "iterations_completed": self.current_iteration,
                "output_dir": self.output_dir,
                "framework": self.framework,
                "generation_plan": self.generation_plan
            })

        if self.verbose:
            print("  MCP setup complete\n")

    def _phase_generate_initial_code(self):
        """Phase 3: Generate initial agent system code (staged)."""
        if self.verbose:
            print("Phase 3: Generating initial code (staged)...")

        stages = [
            ("Core Config and Models", [
                "config/settings.py",
                "models/ticket.py"
            ]),
            ("Agents", [
                "agents/classifier.py",
                "agents/researcher.py",
                "agents/writer.py",
                "agents/reviewer.py"
            ]),
            ("Supervisor and Workflow", [
                "agents/supervisor.py",
                "workflows/ticket_workflow.py"
            ]),
            ("Tools and MCP Clients", [
                "mcps/postgres_client.py",
                "mcps/vector_db_client.py",
                "tools/external_apis.py"
            ]),
            ("Tests and Docs", [
                "tests/test_system.py",
                "requirements.txt",
                "README.md",
                ".env.example"
            ])
        ]

        total_files = 0
        for stage_name, target_files in stages:
            if self.verbose:
                print(f"  Stage: {stage_name}")

            if self.verbose:
                print(f"    Generating code... (this may take a minute)")

            prompt = self._build_stage_generation_prompt(stage_name, target_files)
            response = self.llm.execute_with_progress_retry(prompt, retries=2, backoff=1.0, max_turns=25)

            python_code = self.llm.extract_python_code(response)
            if not python_code:
                blocks = self.llm.extract_code_blocks(response)
                candidates = []
                for b in blocks:
                    code = b.get("code", "")
                    if "filename:" in code.lower():
                        candidates.append(code)
                if not candidates:
                    candidates = self.llm.extract_code_by_filename_comments(response)
                python_code = candidates

            if python_code:
                complete_files = []
                for code in python_code:
                    if self._is_code_complete(code):
                        complete_files.append(code)
                    else:
                        fixed = self._complete_file_via_claude(code)
                        if fixed:
                            complete_files.append(fixed)
                if complete_files:
                    self._save_generated_code(complete_files)
                    count = len(complete_files)
                    total_files += count
                    self.progress_manager.save_generation(
                        f"Generated {count} files in stage: {stage_name}",
                        f"Initial Code Generation - {stage_name}"
                    )
                    if self.verbose:
                        print(f"    Generated {count} files")
                else:
                    if self.verbose:
                        print("    No complete files generated")
            else:
                if self.verbose:
                    print("    No code blocks detected")

        if self.verbose:
            print(f"  Total generated files: {total_files}\n")

    def _is_code_complete(self, code: str) -> bool:
        # Basic checks for completeness
        text = code
        # Check balanced triple quotes
        triple_count = text.count('"""') + text.count("'''")
        if triple_count % 2 != 0:
            return False
        # Check for dangling if/open quotes patterns
        if re.search(r"if\s+\"[^\"]*$", text):
            return False
        # Bracket balance
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        for ch in text:
            if ch in '([{':
                stack.append(ch)
            elif ch in ')]}':
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
        if stack:
            return False
        # In-memory syntax check
        filename = self._extract_filename_from_code(text) or "generated.py"
        try:
            compile(text, filename, 'exec')
        except Exception:
            return False
        return True

    def _complete_file_via_claude(self, partial_code: str) -> Optional[str]:
        # Try to extract filename
        filename = self._extract_filename_from_code(partial_code) or "file.py"
        prompt = f"""
The following Python file is incomplete. Complete it into a valid, self-contained file.

FILENAME: {filename}

PARTIAL CONTENT:
```python
{partial_code}
```

Return ONLY one python code block starting with:
```python
# filename: {filename}
...
```
"""
        resp = self.llm.execute_with_progress_retry(prompt, retries=2, backoff=1.0, max_turns=25)
        code_blocks = self.llm.extract_python_code(resp)
        if not code_blocks:
            code_blocks = self.llm.extract_code_by_filename_comments(resp)
        return code_blocks[0] if code_blocks else None

    def _build_generation_prompt(self) -> str:
        """Build prompt for code generation."""
        context_json = json.dumps(self.context, indent=2)
        llm_config = json.dumps(self.config_parser.create_llm_config(), indent=2)

        prompt = f"""
Generate a complete multi-agent system based on the context and planning.

CONTEXT:
{context_json}

FRAMEWORK: {self.framework}
LLM CONFIGURATION:
{llm_config}

REQUIREMENTS:
1. Generate complete, production-ready code
2. Use {self.framework} framework properly
3. Include proper error handling
4. Add comprehensive docstrings
5. Follow Python best practices
6. Make code modular and maintainable
7. Include requirements.txt if needed
8. Add a README.md with usage instructions

Generate all necessary files:
- Main agent system file(s)
- Configuration/setup files
- Helper/utility modules if needed
- Test file(s)
- README.md

Return each file in a separate Python code block with a comment indicating the filename:
```python
# filename: main.py
[code here]
```
 
CRITICAL OUTPUT RULES:
1. Return ONLY code blocks. Do NOT include any explanations or markdown outside code fences.
2. Each code block MUST start with ```python and include the first line as "# filename: <path>".
3. If you cannot generate code right now, return exactly: NO_CODE
"""
        return prompt

    def _build_stage_generation_prompt(self, stage_name: str, target_files: List[str]) -> str:
        context_json = json.dumps(self.context, indent=2)
        llm_config = json.dumps(self.config_parser.create_llm_config(), indent=2)
        files_list = "\n".join([f"- {p}" for p in target_files])
        return (
            f"You are now in the IMPLEMENTATION phase. You must generate Python code for the stage '{stage_name}'.\n"
            f"DO NOT generate a plan. DO NOT generate YAML. DO NOT use placeholders.\n\n"
            f"CONTEXT:\n{context_json}\n\n"
            f"FRAMEWORK: {self.framework}\n"
            f"LLM CONFIGURATION:\n{llm_config}\n\n"
            f"FILES TO GENERATE:\n{files_list}\n\n"
            "CRITICAL OUTPUT RULES:\n"
            "1. Return ONLY valid Python code blocks.\n"
            "2. Each block MUST start with ```python and the first line as '# filename: <path>'.\n"
            "3. Generate ONLY the files listed above for this stage.\n"
            "4. Ensure the code uses the specified FRAMEWORK ({self.framework}).\n"
            "5. Do not output any text before or after the code blocks.\n"
        )

    def _save_generated_code(self, code_blocks: List[str]):
        """Save generated code to files."""
        os.makedirs(self.output_dir, exist_ok=True)

        for i, code in enumerate(code_blocks):
            # Try to extract filename from code
            filename = self._extract_filename_from_code(code)

            # Skip blocks without filename to avoid ambiguous saves
            if not filename:
                continue

            file_path = os.path.join(self.output_dir, filename)

            # Create subdirectories if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Sanitize code before saving (fix common LLM artifacts)
            sanitized = self._sanitize_code(code)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(sanitized)

            if self.verbose:
                print(f"  Saved: {file_path}")

    def _extract_filename_from_code(self, code: str) -> Optional[str]:
        """Extract filename from code comment."""
        lines = code.split('\n')

        for line in lines[:5]:  # Check first 5 lines
            if 'filename:' in line.lower():
                # Extract filename
                parts = line.split('filename:', 1)
                if len(parts) == 2:
                    filename = parts[1].strip().strip('"\'')
                    return filename

        return None

    def _sanitize_code(self, code: str) -> str:
        """Apply simple sanitizations to reduce obvious syntax issues."""
        text = code
        # Remove stray Markdown fences if present
        if text.strip().startswith('```') and text.strip().endswith('```'):
            text = text.strip().strip('`')

        # Wrap top-level awaits if code uses await outside functions
        uses_await = ' await ' in f" {text} " or text.strip().startswith('await ')
        has_async_def = 'async def ' in text
        if uses_await and not has_async_def:
            indented = "\n".join([f"    {line}" for line in text.splitlines()])
            text = (
                "async def __auto_main__():\n"
                f"{indented}\n\n"
                "import asyncio\n"
                "asyncio.run(__auto_main__())\n"
            )

        return text

    def _phase_iterative_refinement(self):
        """Phase 4: Iterative testing, debugging, and refinement."""
        if self.verbose:
            print("Phase 4: Iterative refinement (test → debug → fix)...")

        self.current_iteration = 0

        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1

            if self.verbose:
                print(f"\n  Iteration {self.current_iteration}/{self.max_iterations}")

            # Step 1: Validate code
            validation_result = self._validate_code()

            if not validation_result["valid"]:
                if self.verbose:
                    print(f"    Validation failed: {validation_result['error']}")

                # Fix validation errors
                self._fix_errors(validation_result["error"], "validation")
                continue

            if self.verbose:
                print("    Validation passed")

            # Step 2: Run tests (if enabled)
            if self.config_parser.is_testing_enabled():
                test_result = self._run_tests()

                if not test_result["success"]:
                    if self.verbose:
                        print(f"    Tests failed: {test_result['error']}")

                    # Fix test failures
                    self._fix_errors(test_result["error"], "testing")
                    continue

                if self.verbose:
                    print("    All tests passed")

            # If we reach here, everything passed!
            if self.verbose:
                print("\n  All iterations completed successfully!")

            break

        if self.current_iteration >= self.max_iterations:
            if self.verbose:
                print(f"\n  WARNING: Reached maximum iterations ({self.max_iterations})")

    def _validate_code(self) -> Dict[str, Any]:
        """Validate generated code syntax and imports."""
        validation_errors = []

        # Ensure output directory is set
        if not self.output_dir:
            self.output_dir = self.config_parser.get_output_directory()
        os.makedirs(self.output_dir, exist_ok=True)

        # Find all Python files
        py_files = list(Path(self.output_dir).rglob("*.py"))

        for py_file in py_files:
            try:
                # Check syntax
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()

                compile(code, str(py_file), 'exec')

                # Check imports (basic check)
                result = subprocess.run(
                    ['python', '-m', 'py_compile', str(py_file)],
                    capture_output=True,
                    timeout=30
                )

                if result.returncode != 0:
                    validation_errors.append(f"{py_file.name}: {result.stderr.decode()}")

            except SyntaxError as e:
                validation_errors.append(f"{py_file.name}: Syntax error at line {e.lineno}: {e.msg}")

            except Exception as e:
                validation_errors.append(f"{py_file.name}: {str(e)}")

        if validation_errors:
            error_msg = "\n".join(validation_errors)
            self.progress_manager.save_testing(f"Validation failed:\n{error_msg}", f"Iteration {self.current_iteration}")

            return {
                "valid": False,
                "error": error_msg
            }

        self.progress_manager.save_testing("Validation passed", f"Iteration {self.current_iteration}")

        return {"valid": True}

    def _run_tests(self) -> Dict[str, Any]:
        """Run generated test cases."""
        # Look for test files
        test_files = list(Path(self.output_dir).rglob("test_*.py"))

        if not test_files:
            # No test files, consider it a pass
            return {"success": True}

        test_errors = []

        for test_file in test_files:
            try:
                rel_path = os.path.relpath(str(test_file), self.output_dir)
                result = subprocess.run(
                    [sys.executable, rel_path],
                    capture_output=True,
                    timeout=60,
                    cwd=self.output_dir
                )

                if result.returncode != 0:
                    stderr = result.stderr.decode(errors='replace')
                    stdout = result.stdout.decode(errors='replace')
                    test_errors.append(f"{test_file.name}:\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

            except subprocess.TimeoutExpired:
                test_errors.append(f"{test_file.name}: Test timed out")

            except Exception as e:
                test_errors.append(f"{test_file.name}: {str(e)}")

        if test_errors:
            error_msg = "\n\n".join(test_errors)
            self.progress_manager.save_testing(f"Tests failed:\n{error_msg}", f"Iteration {self.current_iteration}")

            return {
                "success": False,
                "error": error_msg
            }

        self.progress_manager.save_testing("All tests passed", f"Iteration {self.current_iteration}")

        return {"success": True}

    def _fix_errors(self, error: str, error_type: str):
        """Use the configured LLM to fix errors."""
        if self.verbose:
            print(f"    Fixing {error_type} errors...")

        # Build fix prompt
        prompt = self._build_fix_prompt(error, error_type)

        # Execute fix
        response = self.llm.execute_with_progress_retry(prompt, retries=2, backoff=1.0)

        # Extract and save fixed code
        python_code = self.llm.extract_python_code(response)

        if python_code:
            self._save_generated_code(python_code)

            debug_msg = f"Fixed {error_type} errors in iteration {self.current_iteration}"
            self.progress_manager.save_debugging(debug_msg, f"Iteration {self.current_iteration}")

            if self.verbose:
                print(f"    Applied fixes")
        else:
            if self.verbose:
                print(f"    WARNING: No fixes generated")

    def _build_fix_prompt(self, error: str, error_type: str) -> str:
        """Build prompt for error fixing."""
        prompt = f"""
The generated code has {error_type} errors. Please fix them.

ERROR:
{error}

TASK:
1. Analyze the error
2. Fix the code
3. Ensure all syntax is correct
4. Return the complete fixed files

Return each fixed file in a separate Python code block with filename:
```python
# filename: [filename]
[fixed code]
```
"""
        return prompt

    def _phase_finalize(self) -> Dict[str, Any]:
        """Phase 5: Finalize and create summary."""
        if self.verbose:
            print("\nPhase 5: Finalizing...")

        # Create summary
        summary = self._create_summary()
        self.progress_manager.update_summary(summary)

        # Save final report
        report = self.progress_manager.create_progress_report()
        # Ensure output directory is set
        if not self.output_dir:
            self.output_dir = self.config_parser.get_output_directory()
        os.makedirs(self.output_dir, exist_ok=True)
        report_file = os.path.join(self.output_dir, "GENERATION_REPORT.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        if self.verbose:
            print(f"  Generation complete!")
            print(f"  Output directory: {self.output_dir}")
            print(f"  Total iterations: {self.current_iteration}")
            print("\n" + "="*80 + "\n")

        return {
            "success": True,
            "output_directory": self.output_dir,
            "iterations": self.current_iteration,
            "framework": self.framework
        }

    def _create_summary(self) -> str:
        """Create generation summary."""
        summary = f"""
# Generation Summary

**Framework**: {self.framework}
**Output Directory**: {self.output_dir}
**Iterations**: {self.current_iteration}/{self.max_iterations}
**Status**: {'Success' if self.current_iteration < self.max_iterations else 'Max iterations reached'}

## Context
{json.dumps(self.context, indent=2)}

## Generated Files
"""
        py_files = list(Path(self.output_dir).rglob("*.py"))
        for py_file in py_files:
            rel_path = py_file.relative_to(self.output_dir)
            summary += f"- {rel_path}\n"

        return summary


# Example usage
if __name__ == "__main__":
    print("AgentGenerator requires initialized dependencies to run.")
    print("Please use main.py to run the full generation pipeline.")
