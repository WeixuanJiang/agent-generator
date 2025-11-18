"""
Progress Manager Module

Tracks and saves all execution progress as context for continued generation.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


PROJECT_ROOT = Path.cwd()

class ProgressManager:
    """
    Manage progress tracking and resumption.

    Saves all Claude Code execution steps as markdown files:
    - thinking.md: Analysis and reasoning
    - planning.md: Architecture and design decisions
    - generation.md: Code generation steps
    - testing.md: Test execution results
    - debugging.md: Error fixes and iterations
    """

    def __init__(self, progress_dir: str = str(PROJECT_ROOT / "progress")):
        """
        Initialize ProgressManager.

        Args:
            progress_dir: Directory to store progress files
        """
        self.progress_dir = progress_dir
        self.progress_files = {
            "thinking": os.path.join(progress_dir, "thinking.md"),
            "planning": os.path.join(progress_dir, "planning.md"),
            "generation": os.path.join(progress_dir, "generation.md"),
            "testing": os.path.join(progress_dir, "testing.md"),
            "debugging": os.path.join(progress_dir, "debugging.md"),
            "summary": os.path.join(progress_dir, "summary.md"),
            "errors": os.path.join(progress_dir, "errors.md"),
            "stages": os.path.join(progress_dir, "stages.md")
        }

        self._ensure_progress_dir()
        self._ensure_baseline_files()

    def _ensure_progress_dir(self):
        """Create progress directory if it doesn't exist."""
        os.makedirs(self.progress_dir, exist_ok=True)

    def _ensure_baseline_files(self):
        """Ensure core progress files exist even if empty."""
        for key in ("thinking", "planning"):
            path = self.progress_files.get(key)
            if path and not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("")

    def _append_to_file(self, file_path: str, content: str, heading: Optional[str] = None):
        """
        Append content to a progress file.

        Args:
            file_path: Path to progress file
            content: Content to append
            heading: Optional heading for the entry
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(file_path, 'a', encoding='utf-8') as f:
            if heading:
                f.write(f"\n## {heading}\n")
            f.write(f"**Timestamp**: {timestamp}\n\n")
            f.write(content)
            f.write("\n\n---\n\n")

    def save_thinking(self, content: str, heading: Optional[str] = None):
        """
        Save thinking/reasoning process.

        Args:
            content: Thinking content
            heading: Optional heading
        """
        self._append_to_file(
            self.progress_files["thinking"],
            content,
            heading or "Thinking Process"
        )

    def save_planning(self, content: str, heading: Optional[str] = None):
        """
        Save planning and architecture decisions.

        Args:
            content: Planning content
            heading: Optional heading
        """
        self._append_to_file(
            self.progress_files["planning"],
            content,
            heading or "Planning Phase"
        )

    def save_generation(self, content: str, heading: Optional[str] = None):
        """
        Save code generation log.

        Args:
            content: Generation content
            heading: Optional heading
        """
        self._append_to_file(
            self.progress_files["generation"],
            content,
            heading or "Code Generation"
        )

    def save_testing(self, content: str, heading: Optional[str] = None):
        """
        Save testing results.

        Args:
            content: Testing content
            heading: Optional heading
        """
        self._append_to_file(
            self.progress_files["testing"],
            content,
            heading or "Testing Phase"
        )

    def save_debugging(self, content: str, heading: Optional[str] = None):
        """
        Save debugging iterations.

        Args:
            content: Debugging content
            heading: Optional heading
        """
        self._append_to_file(
            self.progress_files["debugging"],
            content,
            heading or "Debugging Iteration"
        )

    def save_error(self, error: str, context: Optional[str] = None):
        """
        Save error information.

        Args:
            error: Error message
            context: Optional context about when the error occurred
        """
        content = f"**Error**: {error}\n\n"
        if context:
            content += f"**Context**: {context}\n\n"

        self._append_to_file(
            self.progress_files["errors"],
            content,
            "Error Encountered"
        )

    def save_stage_status(self, stage: str, status: str, note: Optional[str] = None):
        """Append stage status to stages.md."""
        content = f"**Stage**: {stage}\n**Status**: {status}\n"
        if note:
            content += f"**Note**: {note}\n"
        self._append_to_file(self.progress_files["stages"], content, "Stage Status")

    def update_summary(self, summary: str):
        """
        Update progress summary.

        Args:
            summary: Current progress summary
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.progress_files["summary"], 'w', encoding='utf-8') as f:
            f.write(f"# Progress Summary\n\n")
            f.write(f"**Last Updated**: {timestamp}\n\n")
            f.write(summary)

    def load_thinking(self) -> str:
        """Load thinking progress."""
        return self._load_file(self.progress_files["thinking"])

    def load_planning(self) -> str:
        """Load planning progress."""
        return self._load_file(self.progress_files["planning"])

    def load_generation(self) -> str:
        """Load generation progress."""
        return self._load_file(self.progress_files["generation"])

    def load_testing(self) -> str:
        """Load testing progress."""
        return self._load_file(self.progress_files["testing"])

    def load_debugging(self) -> str:
        """Load debugging progress."""
        return self._load_file(self.progress_files["debugging"])

    def load_errors(self) -> str:
        """Load error log."""
        return self._load_file(self.progress_files["errors"])

    def load_summary(self) -> str:
        """Load progress summary."""
        return self._load_file(self.progress_files["summary"])

    def _load_file(self, file_path: str) -> str:
        """
        Load content from a progress file.

        Args:
            file_path: Path to progress file

        Returns:
            File content or empty string if file doesn't exist
        """
        if not os.path.exists(file_path):
            return ""

        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_all_progress(self) -> Dict[str, str]:
        """
        Load all progress files.

        Returns:
            Dictionary mapping progress type to content
        """
        return {
            "thinking": self.load_thinking(),
            "planning": self.load_planning(),
            "generation": self.load_generation(),
            "testing": self.load_testing(),
            "debugging": self.load_debugging(),
            "errors": self.load_errors(),
            "summary": self.load_summary()
        }

    def get_progress_context(self) -> str:
        """
        Get all progress as a formatted context string for Claude Code.

        Returns:
            Formatted context string with all progress
        """
        progress = self.load_all_progress()

        context_parts = []

        if progress["summary"]:
            context_parts.append("# PROGRESS SUMMARY\n" + progress["summary"])

        if progress["thinking"]:
            context_parts.append("# THINKING HISTORY\n" + progress["thinking"])

        if progress["planning"]:
            context_parts.append("# PLANNING HISTORY\n" + progress["planning"])

        if progress["generation"]:
            context_parts.append("# GENERATION HISTORY\n" + progress["generation"])

        if progress["testing"]:
            context_parts.append("# TESTING HISTORY\n" + progress["testing"])

        if progress["debugging"]:
            context_parts.append("# DEBUGGING HISTORY\n" + progress["debugging"])

        if progress["errors"]:
            context_parts.append("# ERROR LOG\n" + progress["errors"])

        return "\n\n".join(context_parts)

    def clear_progress(self):
        """Clear all progress files."""
        for file_path in self.progress_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
        self._ensure_baseline_files()

    def get_iteration_count(self) -> int:
        """
        Get number of debugging iterations.

        Returns:
            Number of iterations
        """
        debugging_content = self.load_debugging()
        return debugging_content.count("## Debugging Iteration")

    def has_progress(self) -> bool:
        """
        Check if any progress exists.

        Returns:
            True if progress files exist, False otherwise
        """
        for file_path in self.progress_files.values():
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return True
        return False

    def save_checkpoint(self, checkpoint_name: str, data: Dict[str, Any]):
        """
        Save a checkpoint for resumption.

        Args:
            checkpoint_name: Name of the checkpoint
            data: Checkpoint data
        """
        checkpoint_file = os.path.join(
            self.progress_dir,
            f"checkpoint_{checkpoint_name}.json"
        )

        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_name": checkpoint_name,
            "data": data
        }

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)

    def get_last_checkpoint_stage(self, checkpoint_name: str = "pipeline") -> Optional[str]:
        """Return next_stage from the pipeline checkpoint if available."""
        data = self.load_checkpoint(checkpoint_name)
        if not data:
            return None
        return data.get("next_stage")

    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint

        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_file = os.path.join(
            self.progress_dir,
            f"checkpoint_{checkpoint_name}.json"
        )

        if not os.path.exists(checkpoint_file):
            return None

        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        return checkpoint_data.get("data")

    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint names
        """
        checkpoints = []

        if not os.path.exists(self.progress_dir):
            return checkpoints

        for filename in os.listdir(self.progress_dir):
            if filename.startswith("checkpoint_") and filename.endswith(".json"):
                checkpoint_name = filename[11:-5]  # Remove "checkpoint_" and ".json"
                checkpoints.append(checkpoint_name)

        return checkpoints

    def create_progress_report(self) -> str:
        """
        Create a comprehensive progress report.

        Returns:
            Formatted progress report
        """
        progress = self.load_all_progress()
        iteration_count = self.get_iteration_count()
        checkpoints = self.list_checkpoints()

        report = "# Auto Agent Generator - Progress Report\n\n"
        report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report += f"## Statistics\n\n"
        report += f"- **Debugging Iterations**: {iteration_count}\n"
        report += f"- **Checkpoints Created**: {len(checkpoints)}\n\n"

        if checkpoints:
            report += f"## Available Checkpoints\n\n"
            for checkpoint in checkpoints:
                report += f"- {checkpoint}\n"
            report += "\n"

        report += f"## Phase Status\n\n"
        report += f"- **Thinking**: {'✓ Complete' if progress['thinking'] else '○ Pending'}\n"
        report += f"- **Planning**: {'✓ Complete' if progress['planning'] else '○ Pending'}\n"
        report += f"- **Generation**: {'✓ Complete' if progress['generation'] else '○ Pending'}\n"
        report += f"- **Testing**: {'✓ Complete' if progress['testing'] else '○ Pending'}\n"
        report += f"- **Debugging**: {'✓ Complete' if progress['debugging'] else '○ Pending'}\n\n"

        if progress["errors"]:
            error_count = progress["errors"].count("## Error Encountered")
            report += f"## Errors\n\n"
            report += f"**Total Errors**: {error_count}\n\n"

        return report


# Example usage
if __name__ == "__main__":
    manager = ProgressManager()

    # Example progress tracking
    manager.save_thinking("Analyzing the business context to determine agent architecture...")
    manager.save_planning("Planning a hierarchical multi-agent system with 3 agents")
    manager.save_generation("Generated initial agent code using LangGraph framework")

    # Load and display progress
    print("Progress Context:")
    print(manager.get_progress_context())

    print("\n" + "="*80 + "\n")
    print("Progress Report:")
    print(manager.create_progress_report())
