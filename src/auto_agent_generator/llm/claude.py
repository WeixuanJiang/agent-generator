"""
Claude Code Interface Module

Wrapper for Claude Code CLI commands with context injection and progress tracking.
"""

import subprocess
import os
import re
import json
import time
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path


class ClaudeInterface:
    """
    Interface to execute Claude Code CLI commands with accumulated context.

    Handles:
    - Executing Claude Code commands
    - Context injection from progress files
    - Output streaming and capture
    - Response parsing
    - Code extraction
    """

    def __init__(self, progress_manager=None, verbose: bool = True):
        """
        Initialize ClaudeInterface.

        Args:
            progress_manager: ProgressManager instance for context loading
            verbose: Enable verbose output
        """
        self.progress_manager = progress_manager
        self.verbose = verbose
        self.last_response: Optional[str] = None
        self.last_error: Optional[str] = None

    def execute(
        self,
        prompt: str,
        context: Optional[str] = None,
        timeout: int = 600,
        max_turns: int = 10,
        output_format: str = "text"
    ) -> str:
        """
        Execute Claude Code CLI command in non-interactive (print) mode.

        Args:
            prompt: Prompt to execute
            context: Optional additional context
            timeout: Timeout in seconds
            max_turns: Maximum number of agentic turns (default: 10)
            output_format: Output format - "text" or "json" (default: "text")

        Returns:
            Claude Code output

        Raises:
            RuntimeError: If Claude Code execution fails
        """
        full_prompt = self._build_prompt(prompt, context)

        import platform

        is_windows = platform.system() == 'Windows'
        claude_cmd = 'claude.cmd' if is_windows else 'claude'

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Executing Claude Code CLI...")
            print(f"{'='*80}\n")

        try:
            cmd_parts = [claude_cmd, '-p']
            if max_turns:
                cmd_parts.extend(['--max-turns', str(max_turns)])
            if output_format:
                cmd_parts.extend(['--output-format', output_format])
            # if self.verbose:
            #     cmd_parts.append('--verbose')

            if is_windows:
                # Try to find full path to claude.cmd to avoid shell=True
                import shutil
                claude_path = shutil.which("claude.cmd") or "claude.cmd"
                
                # Use shell=False if we found the path, otherwise fallback to shell=True
                use_shell = False if shutil.which("claude.cmd") else True
                cmd_to_run = [claude_path] + cmd_parts[1:] if not use_shell else " ".join(cmd_parts)

                if self.verbose:
                    print(f"DEBUG: Running command: {cmd_to_run} (shell={use_shell})")

                result = subprocess.run(
                    cmd_to_run,
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=use_shell,
                    encoding="utf-8",
                    errors="replace"
                )
            else:
                result = subprocess.run(
                    cmd_parts,
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=False,
                    encoding="utf-8",
                    errors="replace"
                )

            self.last_response = result.stdout or ""
            self.last_error = result.stderr if result.returncode != 0 else None

            if result.returncode != 0:
                stderr_output = result.stderr.strip() if result.stderr else ""
                stdout_output = result.stdout.strip() if result.stdout else ""
                
                # Check for specific error patterns
                if "Session limit reached" in stderr_output or "Session limit reached" in stdout_output:
                    error_msg = "Claude Code session limit reached. Please wait until 1am for the session to reset."
                elif stderr_output:
                    error_msg = f"Claude Code execution failed (exit {result.returncode}): {stderr_output}"
                elif stdout_output:
                    error_msg = f"Claude Code execution failed (exit {result.returncode}): {stdout_output}"
                else:
                    error_msg = f"Claude Code execution failed (exit {result.returncode}): Unknown error"
                
                if self.progress_manager:
                    self.progress_manager.save_error(error_msg, "Claude CLI")
                if self.verbose:
                    print(f"ERROR: {error_msg}\n")
                raise RuntimeError(error_msg)

            if self.verbose:
                resp = result.stdout or ""
                print(f"Response received ({len(resp)} characters)\n")
                print(resp)
                print()

            return result.stdout

        except subprocess.TimeoutExpired:
            error_msg = f"Claude Code execution timed out after {timeout} seconds"
            if self.verbose:
                print(f"ERROR: {error_msg}\n")
            raise RuntimeError(error_msg)

        except KeyboardInterrupt:
            if self.verbose:
                print("DEBUG: Caught KeyboardInterrupt in ClaudeInterface.execute")
            raise

        except FileNotFoundError:
            error_msg = (
                "Claude Code CLI not found. Please ensure Claude Code is installed.\n"
                "Install from: https://code.claude.com"
            )
            if self.verbose:
                print(f"ERROR: {error_msg}\n")
            raise RuntimeError(error_msg)

    def execute_with_progress(
        self,
        prompt: str,
        include_thinking: bool = True,
        include_planning: bool = True,
        include_generation: bool = True,
        include_testing: bool = True,
        include_debugging: bool = True,
        max_turns: int = 25
    ) -> str:
        """
        Execute Claude Code with progress history as context.

        Args:
            prompt: Prompt to execute
            include_thinking: Include thinking history
            include_planning: Include planning history
            include_generation: Include generation history
            include_testing: Include testing history
            include_debugging: Include debugging history

        Returns:
            Claude Code output
        """
        if self.progress_manager is None:
            return self.execute(prompt)

        # Build context from progress files
        context_parts = []

        if include_thinking:
            thinking = self.progress_manager.load_thinking()
            if thinking:
                context_parts.append(f"# PREVIOUS THINKING\n{thinking}")

        if include_planning:
            planning = self.progress_manager.load_planning()
            if planning:
                context_parts.append(f"# PREVIOUS PLANNING\n{planning}")

        if include_generation:
            generation = self.progress_manager.load_generation()
            if generation:
                context_parts.append(f"# PREVIOUS GENERATION\n{generation}")

        if include_testing:
            testing = self.progress_manager.load_testing()
            if testing:
                context_parts.append(f"# PREVIOUS TESTING\n{testing}")

        if include_debugging:
            debugging = self.progress_manager.load_debugging()
            if debugging:
                context_parts.append(f"# PREVIOUS DEBUGGING\n{debugging}")

        context = "\n\n".join(context_parts) if context_parts else None

        return self.execute(prompt, context=context, max_turns=max_turns)

    def execute_with_retry(
        self,
        prompt: str,
        context: Optional[str] = None,
        retries: int = 2,
        backoff: float = 1.0,
        timeout: int = 600,
        max_turns: int = 10,
        output_format: str = "text"
    ) -> str:
        attempt = 0
        while True:
            try:
                return self.execute(
                    prompt,
                    context=context,
                    timeout=timeout,
                    max_turns=max_turns,
                    output_format=output_format
                )
            except RuntimeError:
                if attempt >= retries:
                    raise
                time.sleep(backoff)
                attempt += 1
                backoff *= 2

    def execute_with_progress_retry(
        self,
        prompt: str,
        retries: int = 2,
        backoff: float = 1.0,
        include_thinking: bool = True,
        include_planning: bool = True,
        include_generation: bool = True,
        include_testing: bool = True,
        include_debugging: bool = True,
        max_turns: int = 10
    ) -> str:
        context_parts = []
        if self.progress_manager is not None:
            if include_thinking:
                thinking = self.progress_manager.load_thinking()
                if thinking:
                    context_parts.append(f"# PREVIOUS THINKING\n{thinking}")
            if include_planning:
                planning = self.progress_manager.load_planning()
                if planning:
                    context_parts.append(f"# PREVIOUS PLANNING\n{planning}")
            if include_generation:
                generation = self.progress_manager.load_generation()
                if generation:
                    context_parts.append(f"# PREVIOUS GENERATION\n{generation}")
            if include_testing:
                testing = self.progress_manager.load_testing()
                if testing:
                    context_parts.append(f"# PREVIOUS TESTING\n{testing}")
            if include_debugging:
                debugging = self.progress_manager.load_debugging()
                if debugging:
                    context_parts.append(f"# PREVIOUS DEBUGGING\n{debugging}")
        context = "\n\n".join(context_parts) if context_parts else None
        return self.execute_with_retry(prompt, context=context, retries=retries, backoff=backoff, max_turns=max_turns)

    def _build_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Build full prompt with context.

        Args:
            prompt: Main prompt
            context: Additional context

        Returns:
            Full prompt with context
        """
        if context:
            return f"{context}\n\n{'='*80}\n\n# CURRENT TASK\n\n{prompt}"
        return prompt

    def extract_code_blocks(self, response: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Extract code blocks from response.

        Args:
            response: Response to parse (uses last_response if None)

        Returns:
            List of code blocks with language and content
        """
        response = response or self.last_response

        if not response:
            return []

        # Pattern to match markdown code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        code_blocks = []
        for language, code in matches:
            code_blocks.append({
                "language": language or "text",
                "code": code.strip()
            })

        return code_blocks

    def extract_python_code(self, response: Optional[str] = None) -> List[str]:
        """
        Extract Python code blocks from response.

        Args:
            response: Response to parse (uses last_response if None)

        Returns:
            List of Python code strings
        """
        code_blocks = self.extract_code_blocks(response)

        python_code = []
        for block in code_blocks:
            if block["language"].lower() in ["python", "py"]:
                python_code.append(block["code"])

        return python_code

    def extract_code_by_filename_comments(self, response: Optional[str] = None) -> List[str]:
        response = response or self.last_response
        if not response:
            return []
        lines = response.splitlines()
        files: List[str] = []
        current: List[str] = []
        started = False
        for line in lines:
            if '# filename:' in line.lower():
                if current:
                    files.append("\n".join(current).strip())
                    current = []
                started = True
                current.append(line)
            else:
                if started:
                    current.append(line)
        if current:
            files.append("\n".join(current).strip())
        return files

    def extract_json(self, response: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from response.

        Args:
            response: Response to parse (uses last_response if None)

        Returns:
            Parsed JSON or None
        """
        response = response or self.last_response

        if not response:
            return None

        # Try to find JSON in code blocks first
        code_blocks = self.extract_code_blocks(response)
        for block in code_blocks:
            if block["language"].lower() == "json":
                try:
                    return json.loads(block["code"])
                except json.JSONDecodeError:
                    continue

        # Try to find JSON anywhere in the response
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    def extract_thinking(self, response: Optional[str] = None) -> str:
        """
        Extract thinking/reasoning from response.

        Args:
            response: Response to parse (uses last_response if None)

        Returns:
            Thinking content
        """
        response = response or self.last_response

        if not response:
            return ""

        # Look for thinking sections
        patterns = [
            r'(?:## )?(?:Thinking|Analysis|Reasoning):?\s*\n(.*?)(?:\n##|\n\n```|$)',
            r'(?:Let me think|I will|I need to|First,)(.*?)(?:\n\n|$)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()

        return ""

    def extract_file_operations(self, response: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract file operations (create, update, delete) from response.

        Args:
            response: Response to parse (uses last_response if None)

        Returns:
            List of file operations
        """
        response = response or self.last_response

        if not response:
            return []

        operations = []

        # Look for file creation/update patterns
        create_patterns = [
            r'(?:Create|Creating|Created)\s+(?:file|`)([\w/.-]+)`?',
            r'(?:Write|Writing|Written)\s+(?:to\s+)?(?:file|`)([\w/.-]+)`?',
        ]

        for pattern in create_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                operations.append({
                    "operation": "create",
                    "file": match.strip()
                })

        return operations

    def parse_response(self, response: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse response and extract all relevant information.

        Args:
            response: Response to parse (uses last_response if None)

        Returns:
            Dictionary with parsed information
        """
        response = response or self.last_response

        return {
            "raw_response": response,
            "code_blocks": self.extract_code_blocks(response),
            "python_code": self.extract_python_code(response),
            "json_data": self.extract_json(response),
            "thinking": self.extract_thinking(response),
            "file_operations": self.extract_file_operations(response)
        }

    def save_code_to_file(self, code: str, file_path: str):
        """
        Save extracted code to file.

        Args:
            code: Code content
            file_path: Output file path
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        if self.verbose:
            print(f"Saved code to: {file_path}")

    def execute_and_extract_code(
        self,
        prompt: str,
        output_dir: str = "../outputs",
        filename_prefix: str = "generated"
    ) -> List[str]:
        """
        Execute prompt and save all extracted Python code to files.

        Args:
            prompt: Prompt to execute
            output_dir: Output directory
            filename_prefix: Prefix for generated files

        Returns:
            List of created file paths
        """
        response = self.execute(prompt)
        python_code = self.extract_python_code(response)

        created_files = []

        for i, code in enumerate(python_code):
            filename = f"{filename_prefix}_{i}.py" if len(python_code) > 1 else f"{filename_prefix}.py"
            file_path = os.path.join(output_dir, filename)

            self.save_code_to_file(code, file_path)
            created_files.append(file_path)

        return created_files

    def test_connection(self) -> bool:
        """
        Test if Claude Code CLI is available.

        Returns:
            True if Claude Code is available, False otherwise
        """
        import platform

        # Try different approaches based on platform
        commands_to_try = []

        if platform.system() == 'Windows':
            # On Windows, try both claude.cmd and claude with shell
            commands_to_try = [
                (['claude', '--version'], False),  # Try without shell first
                (['claude.cmd', '--version'], True),  # Try with .cmd and shell
                (['claude', '--version'], True),  # Try with shell
            ]
        else:
            # On macOS/Linux/WSL, just use claude
            commands_to_try = [
                (['claude', '--version'], False),
            ]

        for cmd, use_shell in commands_to_try:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=10,
                    shell=use_shell
                )
                if result.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue

        return False


# Example usage
if __name__ == "__main__":
    # Test connection
    claude = ClaudeInterface(verbose=True)

    if not claude.test_connection():
        print("ERROR: Claude Code CLI not available. Please install Claude Code.")
        exit(1)

    print("Claude Code CLI is available\n")

    # Example execution
    try:
        response = claude.execute(
            "Explain the benefits of using multi-agent systems for complex workflows."
        )

        print("Response:")
        print(response[:500] + "..." if len(response) > 500 else response)

        # Parse response
        parsed = claude.parse_response()
        print(f"\nExtracted {len(parsed['code_blocks'])} code blocks")
        print(f"Thinking: {parsed['thinking'][:100]}..." if parsed['thinking'] else "No thinking found")

    except Exception as e:
        print(f"Error: {e}")
