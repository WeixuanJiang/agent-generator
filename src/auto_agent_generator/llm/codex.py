"""
Codex CLI Interface Module

Wrapper for Codex CLI commands with context injection and progress tracking.
Matches the interface of ClaudeInterface so it can be swapped at runtime.
"""

import subprocess
import os
import re
import json
import time
from typing import Dict, Optional, Any, List


class CodexInterface:
    """
    Interface to execute Codex CLI commands with accumulated context.
    """

    def __init__(self, progress_manager=None, verbose: bool = True):
        self.progress_manager = progress_manager
        self.verbose = verbose
        self.last_response: Optional[str] = None
        self.last_error: Optional[str] = None

    def execute(
        self,
        prompt: str,
        context: Optional[str] = None,
        timeout: int = 600,
        output_format: str = "text",
    ) -> str:
        """Execute Codex in non-interactive mode via `codex exec -`."""
        full_prompt = self._build_prompt(prompt, context)

        if self.verbose:
            print(f"\n{'='*80}")
            print("Executing Codex CLI...")
            print(f"{'='*80}\n")

        # Use --skip-git-repo-check so Codex can run in non-git or untrusted dirs
        cmd_parts = ["codex", "exec", "--skip-git-repo-check", "-"]

        try:
            result = subprocess.run(
                cmd_parts,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                encoding="utf-8",
                errors="replace",
                env=self._env_with_output_format(output_format),
            )

            self.last_response = result.stdout or ""
            self.last_error = result.stderr if result.returncode != 0 else None

            if result.returncode != 0:
                stderr_output = result.stderr.strip() if result.stderr else ""
                stdout_output = result.stdout.strip() if result.stdout else ""
                error_msg = (
                    f"Codex execution failed (exit {result.returncode}): {stderr_output or stdout_output or 'Unknown error'}"
                )
                if self.progress_manager:
                    self.progress_manager.save_error(error_msg, "Codex CLI")
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
            error_msg = f"Codex execution timed out after {timeout} seconds"
            if self.verbose:
                print(f"ERROR: {error_msg}\n")
            raise RuntimeError(error_msg)

        except FileNotFoundError:
            error_msg = (
                "Codex CLI not found. Please ensure Codex is installed.\n"
                "Install from: https://github.com"
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
        max_turns: int = 10,
    ) -> str:
        if self.progress_manager is None:
            return self.execute(prompt)

        context_parts: List[str] = []
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
        # Codex does not expose max_turns; included to keep signature compatible
        return self.execute(prompt, context=context)

    def execute_with_retry(
        self,
        prompt: str,
        context: Optional[str] = None,
        retries: int = 2,
        backoff: float = 1.0,
        timeout: int = 600,
        max_turns: int = 10,
        output_format: str = "text",
    ) -> str:
        attempt = 0
        while True:
            try:
                return self.execute(
                    prompt,
                    context=context,
                    timeout=timeout,
                    output_format=output_format,
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
        max_turns: int = 10,
        timeout: int = 600,
    ) -> str:
        context_parts: List[str] = []
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
        return self.execute_with_retry(
            prompt,
            context=context,
            retries=retries,
            backoff=backoff,
            timeout=timeout,
            output_format="text",
        )

    def _build_prompt(self, prompt: str, context: Optional[str] = None) -> str:
        if context:
            return f"{context}\n\n{'='*80}\n\n# CURRENT TASK\n\n{prompt}"
        return prompt

    def _env_with_output_format(self, output_format: str) -> Dict[str, str]:
        env = os.environ.copy()
        if output_format == "json":
            env["CODEX_OUTPUT_FORMAT"] = "json"
        return env

    # Parsing helpers mirror ClaudeInterface to keep compatibility
    def extract_code_blocks(self, response: Optional[str] = None) -> List[Dict[str, str]]:
        response = response or self.last_response
        if not response:
            return []
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        return [
            {"language": language or "text", "code": code.strip()}
            for language, code in matches
        ]

    def extract_python_code(self, response: Optional[str] = None) -> List[str]:
        code_blocks = self.extract_code_blocks(response)
        return [block["code"] for block in code_blocks if block["language"].lower() in ["python", "py"]]

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

    def extract_thinking(self, response: Optional[str] = None) -> str:
        response = response or self.last_response or ""
        thinking = []
        in_thinking = False
        for line in response.splitlines():
            if "# Thinking" in line or "<thinking>" in line.lower():
                in_thinking = True
                continue
            if in_thinking and line.strip() == "":
                break
            if in_thinking:
                thinking.append(line)
        return "\n".join(thinking).strip()

    def parse_response(self, response: Optional[str] = None) -> Dict[str, Any]:
        response = response or self.last_response or ""
        return {
            "code_blocks": self.extract_code_blocks(response),
            "python_code": self.extract_python_code(response),
            "thinking": self.extract_thinking(response)
        }

    def save_code_to_file(self, code: str, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)

    def save_code_blocks_to_files(self, python_code: List[str], output_dir: str, filename_prefix: str = "generated") -> List[str]:
        created_files = []
        os.makedirs(output_dir, exist_ok=True)
        for i, code in enumerate(python_code):
            filename = f"{filename_prefix}_{i}.py" if len(python_code) > 1 else f"{filename_prefix}.py"
            file_path = os.path.join(output_dir, filename)
            self.save_code_to_file(code, file_path)
            created_files.append(file_path)
        return created_files

    def test_connection(self) -> bool:
        commands_to_try = [(["codex", "--version"], False)]
        for cmd, use_shell in commands_to_try:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=10,
                    shell=use_shell,
                )
                if result.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        return False


if __name__ == "__main__":
    client = CodexInterface(verbose=True)
    if not client.test_connection():
        print("ERROR: Codex CLI not available. Please install Codex.")
        exit(1)
    print("Codex CLI is available\n")
    try:
        resp = client.execute("Reply with OK")
        print(resp)
    except Exception as e:
        print(f"Error: {e}")
