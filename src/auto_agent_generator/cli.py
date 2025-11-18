"""
Main Entry Point

Autonomous multi-agent system generator using Claude Code CLI.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from auto_agent_generator.context.parser import ContextParser
from auto_agent_generator.config.parser import ConfigParser
from auto_agent_generator.progress.manager import ProgressManager
from auto_agent_generator.llm.claude import ClaudeInterface
from auto_agent_generator.llm.codex import CodexInterface
from auto_agent_generator.mcp.manager import MCPManager
from auto_agent_generator.pipeline.generator import AgentGenerator


REPO_ROOT = Path.cwd()
DATA_DIR = REPO_ROOT / "data"
PROGRESS_DIR = REPO_ROOT / "progress"


def create_llm_interface(llm_runner: str, progress_manager=None, verbose: bool = True):
    if llm_runner == "codex":
        return CodexInterface(progress_manager=progress_manager, verbose=verbose)
    return ClaudeInterface(progress_manager=progress_manager, verbose=verbose)


def parse_cli_flags(args):
    """Parse CLI flags for llm override and restart."""
    runner = None
    restart = False
    cleaned = []
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in ("--llm", "--runner"):
            if i + 1 < len(args):
                runner = args[i + 1].lower()
                skip_next = True
            continue
        if arg == "--restart":
            restart = True
            continue
        cleaned.append(arg)
    return runner, restart, cleaned


def setup_logging(verbose: bool = True):
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.INFO if verbose else logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('auto_agent_generator.log')
        ]
    )


def print_banner():
    """Print application banner."""
    banner = """
================================================================

            AUTO AI AGENTS GENERATOR

    Autonomous multi-agent system generation using
    Claude Code CLI with full iteration until perfect

================================================================
"""
    print(banner)


def check_prerequisites(llm_runner: str):
    """
    Check if all prerequisites are met.

    Returns:
        True if all prerequisites met, False otherwise
    """
    print("Checking prerequisites...")

    context_path = DATA_DIR / "context.md"
    config_path = DATA_DIR / "config.json"

    # Check for context.md
    if not context_path.exists():
        print("  [x] data/context.md not found")
        print("    Please create data/context.md with your business requirements")
        return False
    print("  [+] data/context.md found")

    # Check for config.json
    if not config_path.exists():
        print("  [x] data/config.json not found")
        print("    Please create data/config.json with your configuration")
        return False
    print("  [+] data/config.json found")

    # Check for selected LLM CLI
    if llm_runner == "codex":
        llm_client = CodexInterface(verbose=False)
        cli_name = "Codex"
        install_hint = "https://github.com/openai"
    else:
        llm_client = ClaudeInterface(verbose=False)
        cli_name = "Claude Code"
        install_hint = "https://code.claude.com"

    if not llm_client.test_connection():
        print(f"  [x] {cli_name} CLI not available")
        print(f"    Please install {cli_name} from: {install_hint}")
        return False
    print(f"  [+] {cli_name} CLI available")

    print()
    return True


def main(llm_override: Optional[str] = None, restart: bool = False):
    """Main execution function."""
    # Print banner
    print_banner()

    # Ensure data files exist before loading config
    context_path = DATA_DIR / "context.md"
    config_path = DATA_DIR / "config.json"
    if not context_path.exists():
        print("Context file data/context.md not found. Please create it.")
        return 1
    if not config_path.exists():
        print("Configuration file data/config.json not found. Please create it.")
        return 1

    # Load configuration early to determine runner
    config_parser = ConfigParser(str(config_path))
    try:
        config_parser.load_config()
    except Exception as e:
        print(f"Failed to load config: {e}")
        return 1

    llm_runner = config_parser.get_llm_runner(override=llm_override)
    if llm_runner not in ConfigParser.SUPPORTED_LLMS:
        print(f"Unsupported llm_runner: {llm_runner}. Supported: {', '.join(ConfigParser.SUPPORTED_LLMS)}")
        return 1

    # Check prerequisites for selected runtime
    if not check_prerequisites(llm_runner):
        print("Prerequisites not met. Exiting.")
        return 1

    try:
        # Setup logging
        setup_logging(verbose=True)
        logger = logging.getLogger(__name__)

        logger.info("Starting Auto Agent Generator...")

        # Initialize components
        print("Initializing components...")

        base_dir = REPO_ROOT
        context_parser = ContextParser(str(DATA_DIR / "context.md"))
        print("  [+] Context Parser initialized")

        progress_manager = ProgressManager(str(PROGRESS_DIR))
        if restart:
            progress_manager.clear_progress()
            if (PROGRESS_DIR / "checkpoint_pipeline.json").exists():
                try:
                    (PROGRESS_DIR / "checkpoint_pipeline.json").unlink()
                except Exception:
                    pass
            print("  [+] Progress cleared (restart requested)")
        else:
            print("  [+] Progress Manager initialized")

        llm_interface = create_llm_interface(
            llm_runner,
            progress_manager=progress_manager,
            verbose=config_parser.is_verbose(),
        )
        print(f"  [+] {llm_runner.title()} Interface initialized")

        mcp_manager = MCPManager(
            llm_interface=llm_interface,
            verbose=config_parser.is_verbose()
        )
        print("  [+] MCP Manager initialized")

        agent_generator = AgentGenerator(
            context_parser=context_parser,
            config_parser=config_parser,
            progress_manager=progress_manager,
            llm_interface=llm_interface,
            mcp_manager=mcp_manager,
            verbose=config_parser.is_verbose()
        )
        print("  [+] Agent Generator initialized\n")

        # Preflight LLM CLI
        print(f"Running {llm_runner.title()} CLI preflight...")
        try:
            llm_interface.execute_with_retry(
                prompt="Reply with OK",
                timeout=60,
                max_turns=1
            )
            print(f"  [+] {llm_runner.title()} CLI preflight passed\n")
        except Exception as e:
            progress_manager.save_error(str(e), f"{llm_runner.title()} Preflight")
            print(f"  [x] {llm_runner.title()} CLI preflight failed")
            print("    See progress/errors.md and auto_agent_generator.log")
            return 1

        # Parse context
        print("Parsing context...")
        try:
            context = context_parser.parse_context()
            context_parser.save_parsed_context()
            print("  [+] Context parsed and saved\n")
            progress_manager.save_stage_status("analysis_and_planning", "success")
        except Exception as e:
            progress_manager.save_error(str(e), "Parsing context")
            progress_manager.save_stage_status("analysis_and_planning", "failed", str(e))
            logger = logging.getLogger(__name__)
            logger.exception("Context parsing failed")
            print("Parsing failed. Exiting.")
            return 1

        # Display configuration
        print("Configuration:")
        print(f"  Framework: {config_parser.get_framework()}")
        print(f"  LLM Runner: {llm_runner}")
        print(f"  Model Provider: {config_parser.get_provider_name()}")
        print(f"  Model: {config_parser.get_model_name()}")
        print(f"  Max Iterations: {config_parser.get_max_iterations()}")
        print(f"  Output Directory: {config_parser.get_output_directory()}")
        print(f"  Testing Enabled: {config_parser.is_testing_enabled()}\n")

        # Start generation
        start_time = datetime.now()

        logger.info("Starting autonomous generation...")

        # Resume if checkpoint exists
        resume_stage = None if restart else progress_manager.get_last_checkpoint_stage("pipeline")
        if resume_stage:
            print(f"Resuming from stage: {resume_stage}")
        results = agent_generator.generate(resume=bool(resume_stage))

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Display results
        print("\n" + "="*80)
        print("GENERATION RESULTS")
        print("="*80 + "\n")

        if results["success"]:
            print("[SUCCESS]")
            print(f"\n  Output Directory: {results['output_directory']}")
            print(f"  Framework: {results['framework']}")
            print(f"  Total Iterations: {results['iterations']}")
            print(f"  Duration: {duration:.2f} seconds")

            print("\nGenerated files:")
            output_dir = Path(results['output_directory'])
            for py_file in output_dir.rglob("*.py"):
                rel_path = py_file.relative_to(output_dir)
                print(f"  - {rel_path}")

            print(f"\nSee {results['output_directory']}/GENERATION_REPORT.md for details.")

            logger.info(f"Generation completed successfully in {duration:.2f} seconds")

            return 0

        else:
            print("[FAILED] GENERATION FAILED")
            print(f"\n  Error: {results.get('error', 'Unknown error')}")
            print(f"  Iteration: {results.get('iteration', 0)}")
            print(f"  Duration: {duration:.2f} seconds")

            print("\nCheck progress/ folder for detailed logs.")

            logger.error(f"Generation failed: {results.get('error')}")

            return 1

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        print("Progress has been saved. You can resume later.")
        return 130

    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"\n[ERROR]: {str(e)}")
        print("\nCheck auto_agent_generator.log for details.")
        return 1


def run_example(llm_override: Optional[str] = None, restart: bool = False):
    """Run with example context and config."""
    print("Running with example configuration...")

    import shutil

    # Determine example config to use
    example_config = None
    if os.path.exists("examples/example_config_langchain_openai.json"):
        example_config = "examples/example_config_langchain_openai.json"
    elif os.path.exists("examples/example_config_crewai_aws.json"):
        example_config = "examples/example_config_crewai_aws.json"

    if not example_config:
        print("No example config found in examples/.")
        return 1

    # Copy config to expected location
    os.makedirs(DATA_DIR, exist_ok=True)
    shutil.copy(example_config, str(DATA_DIR / "config.json"))

    # Ensure context exists at expected location
    if not (DATA_DIR / "context.md").exists():
        print("Context file data/context.md not found. Please create it.")
        return 1

    print("Example config copied. Running generation...")

    return main(llm_override=llm_override, restart=restart)


def clear_progress():
    """Clear all progress files."""
    print("Clearing progress...")

    progress_manager = ProgressManager(str(PROGRESS_DIR))
    progress_manager.clear_progress()

    print("Progress cleared.")


def show_status():
    """Show current generation status."""
    progress_manager = ProgressManager(str(PROGRESS_DIR))

    if not progress_manager.has_progress():
        print("No progress found.")
        return

    print("\nCurrent Status:")
    print(progress_manager.create_progress_report())


if __name__ == "__main__":
    import sys

    runner_override, restart_requested, remaining_args = parse_cli_flags(sys.argv[1:])

    # Simple CLI argument handling
    if remaining_args:
        command = remaining_args[0].lower()

        if command == "example":
            sys.exit(run_example(llm_override=runner_override, restart=restart_requested))

        elif command == "clear":
            clear_progress()
            sys.exit(0)

        elif command == "status":
            show_status()
            sys.exit(0)

        elif command == "help":
            print("""
Auto Agent Generator - Command Line Interface

USAGE:
    python -m auto_agent_generator.cli [command] [--llm claude|codex] [--restart]
    # or after editable install: auto-agent-generator [command] [--llm claude|codex] [--restart]

COMMANDS:
    (no command)    Run the generator with data/context.md and data/config.json
    example         Run with example configuration
    clear           Clear all progress files
    status          Show current generation status
    help            Show this help message

FLAGS:
    --llm / --runner    Select runtime (claude|codex)
    --restart           Clear progress and start from scratch

SETUP:
    1. Create data/context.md with your business requirements
    2. Create data/config.json with your configuration
    3. Run: python -m auto_agent_generator.cli

For more information, see README.md
""")
            sys.exit(0)

        else:
            print(f"Unknown command: {command}")
            print("Run 'python -m auto_agent_generator.cli help' for usage information")
            sys.exit(1)

    else:
        # Run main generation
        sys.exit(main(llm_override=runner_override, restart=restart_requested))
