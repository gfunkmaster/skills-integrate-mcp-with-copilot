"""
Command-line interface for the Agentic Chain package.
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path

from .orchestrator import AgenticChain


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Agentic Chain - AI-powered issue solving framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a project
  agentic-chain analyze /path/to/project

  # Solve an issue from a JSON file
  agentic-chain solve /path/to/project --issue-file issue.json

  # Solve an issue with inline data
  agentic-chain solve /path/to/project --title "Bug in login" --body "Description..."

  # Use LLM for AI-powered solutions
  agentic-chain solve /path/to/project --issue-file issue.json --llm openai

  # Use specific model
  agentic-chain solve /path/to/project --issue-file issue.json --llm anthropic --model claude-3-sonnet-20240229

  # Export results to JSON
  agentic-chain solve /path/to/project --issue-file issue.json --output result.json
        """
    )
    
    # Get available providers dynamically
    try:
        from .llm import LLMFactory
        available_providers = LLMFactory.list_providers()
    except ImportError:
        available_providers = ["openai", "anthropic"]  # Fallback
    
    # Global LLM options
    parser.add_argument(
        "--llm",
        choices=available_providers,
        help="LLM provider to use for AI-powered solutions"
    )
    parser.add_argument(
        "--model",
        help="Specific model to use (e.g., gpt-4, claude-3-sonnet-20240229)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for the LLM provider (defaults to env var)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a project structure and patterns"
    )
    analyze_parser.add_argument(
        "project_path",
        help="Path to the project to analyze"
    )
    analyze_parser.add_argument(
        "-o", "--output",
        help="Output file for JSON results"
    )
    analyze_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Solve command
    solve_parser = subparsers.add_parser(
        "solve",
        help="Solve an issue in a project"
    )
    solve_parser.add_argument(
        "project_path",
        help="Path to the project"
    )
    solve_parser.add_argument(
        "--issue-file",
        help="JSON file containing issue data"
    )
    solve_parser.add_argument(
        "--title",
        help="Issue title (alternative to --issue-file)"
    )
    solve_parser.add_argument(
        "--body",
        help="Issue body/description"
    )
    solve_parser.add_argument(
        "--labels",
        nargs="+",
        help="Issue labels"
    )
    solve_parser.add_argument(
        "-o", "--output",
        help="Output file for JSON results"
    )
    solve_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    solve_parser.add_argument(
        "--summary",
        action="store_true",
        help="Print human-readable summary"
    )
    solve_parser.add_argument(
        "--show-usage",
        action="store_true",
        help="Show LLM token usage and cost"
    )
    
    # Providers command - list available LLM providers
    providers_parser = subparsers.add_parser(
        "providers",
        help="List available LLM providers"
    )
    
    # Memory command - manage memory system
    memory_parser = subparsers.add_parser(
        "memory",
        help="Manage the memory system for context awareness"
    )
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command")
    
    # Memory stats
    memory_stats_parser = memory_subparsers.add_parser(
        "stats",
        help="Show memory system statistics"
    )
    memory_stats_parser.add_argument(
        "--db-path",
        help="Path to memory database"
    )
    
    # Memory search
    memory_search_parser = memory_subparsers.add_parser(
        "search",
        help="Search memories"
    )
    memory_search_parser.add_argument(
        "query",
        help="Search query"
    )
    memory_search_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Maximum results"
    )
    memory_search_parser.add_argument(
        "--project",
        help="Filter by project path"
    )
    memory_search_parser.add_argument(
        "--db-path",
        help="Path to memory database"
    )
    
    # Memory export
    memory_export_parser = memory_subparsers.add_parser(
        "export",
        help="Export memories to JSON file"
    )
    memory_export_parser.add_argument(
        "output",
        help="Output file path"
    )
    memory_export_parser.add_argument(
        "--db-path",
        help="Path to memory database"
    )
    
    # Memory import
    memory_import_parser = memory_subparsers.add_parser(
        "import",
        help="Import memories from JSON file"
    )
    memory_import_parser.add_argument(
        "input",
        help="Input file path"
    )
    memory_import_parser.add_argument(
        "--db-path",
        help="Path to memory database"
    )
    
    # Memory prune
    memory_prune_parser = memory_subparsers.add_parser(
        "prune",
        help="Clean up old or low-importance memories"
    )
    memory_prune_parser.add_argument(
        "--max-age",
        type=int,
        help="Remove entries older than N days"
    )
    memory_prune_parser.add_argument(
        "--max-entries",
        type=int,
        help="Keep only N entries"
    )
    memory_prune_parser.add_argument(
        "--min-importance",
        type=float,
        help="Remove entries below this importance"
    )
    memory_prune_parser.add_argument(
        "--db-path",
        help="Path to memory database"
    )
    
    # Memory clear
    memory_clear_parser = memory_subparsers.add_parser(
        "clear",
        help="Clear all memories (use with caution!)"
    )
    memory_clear_parser.add_argument(
        "--db-path",
        help="Path to memory database"
    )
    memory_clear_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    setup_logging(getattr(args, 'verbose', False))
    
    if args.command == "analyze":
        handle_analyze(args)
    elif args.command == "solve":
        handle_solve(args)
    elif args.command == "providers":
        handle_providers(args)
    elif args.command == "memory":
        handle_memory(args)


def get_llm_config(args) -> dict:
    """Build LLM config from command line arguments."""
    if not args.llm:
        return None
    
    config = {"provider": args.llm}
    
    if args.model:
        config["model"] = args.model
    
    if args.api_key:
        config["api_key"] = args.api_key
    
    return config


def handle_analyze(args):
    """Handle the analyze command."""
    project_path = Path(args.project_path)
    
    if not project_path.exists():
        print(f"Error: Project path '{project_path}' does not exist")
        sys.exit(1)
        
    chain = AgenticChain(project_path=str(project_path))
    result = chain.analyze_project()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Analysis saved to {args.output}")
    else:
        print(json.dumps(result, indent=2, default=str))


def handle_solve(args):
    """Handle the solve command."""
    project_path = Path(args.project_path)
    
    if not project_path.exists():
        print(f"Error: Project path '{project_path}' does not exist")
        sys.exit(1)
        
    # Get issue data
    issue_data = {}
    
    if args.issue_file:
        try:
            with open(args.issue_file, 'r') as f:
                issue_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading issue file: {e}")
            sys.exit(1)
    elif args.title:
        issue_data = {
            "title": args.title,
            "body": args.body or "",
            "labels": [{"name": label} for label in (args.labels or [])],
        }
    else:
        print("Error: Either --issue-file or --title is required")
        sys.exit(1)
    
    # Build LLM config
    llm_config = get_llm_config(args)
    
    # Create chain
    chain = AgenticChain(
        project_path=str(project_path),
        llm_config=llm_config,
    )
    
    try:
        result = chain.solve_issue(issue_data)
    except Exception as e:
        print(f"Error solving issue: {e}")
        sys.exit(1)
        
    if args.summary:
        print(chain.get_solution_summary())
    
    if getattr(args, 'show_usage', False):
        usage = chain.get_llm_usage()
        if usage.get('total_tokens', 0) > 0:
            print("\n🤖 LLM Usage:")
            print(f"  Provider: {usage.get('provider', 'N/A')}")
            print(f"  Model: {usage.get('model', 'N/A')}")
            print(f"  Total tokens: {usage.get('total_tokens', 0)}")
            print(f"  Estimated cost: ${usage.get('estimated_cost', 0):.4f}")
        
    if args.output:
        chain.export_result(args.output)
        print(f"Full results saved to {args.output}")
    elif not args.summary:
        print(json.dumps(result, indent=2, default=str))


def handle_providers(args):
    """Handle the providers command."""
    try:
        from .llm import LLMFactory
        providers = LLMFactory.list_providers()
        print("Available LLM providers:")
        for provider in providers:
            env_var = f"{provider.upper()}_API_KEY"
            has_key = "✓" if os.environ.get(env_var) else "✗"
            print(f"  • {provider} (env: {env_var}) [{has_key}]")
    except ImportError as e:
        print(f"Error loading LLM module: {e}")
        sys.exit(1)


def handle_memory(args):
    """Handle memory commands."""
    from .memory import MemoryManager
    
    if not args.memory_command:
        print("Error: Please specify a memory command (stats, search, export, import, prune, clear)")
        sys.exit(1)
    
    db_path = getattr(args, 'db_path', None)
    manager = MemoryManager(db_path=db_path)
    
    try:
        if args.memory_command == "stats":
            handle_memory_stats(manager)
        elif args.memory_command == "search":
            handle_memory_search(manager, args)
        elif args.memory_command == "export":
            handle_memory_export(manager, args)
        elif args.memory_command == "import":
            handle_memory_import(manager, args)
        elif args.memory_command == "prune":
            handle_memory_prune(manager, args)
        elif args.memory_command == "clear":
            handle_memory_clear(manager, args)
    finally:
        manager.close()


def handle_memory_stats(manager):
    """Show memory statistics."""
    print(manager.get_stats_summary())


def handle_memory_search(manager, args):
    """Search memories."""
    results = manager.search(
        query=args.query,
        project_path=getattr(args, 'project', None),
        limit=args.limit,
    )
    
    if not results:
        print("No memories found matching your query.")
        return
    
    print(f"Found {len(results)} matching memories:\n")
    for i, result in enumerate(results, 1):
        entry = result.entry
        print(f"{i}. [{entry.memory_type.value}] Score: {result.score:.3f}")
        print(f"   ID: {entry.id}")
        content_preview = entry.content[:100].replace('\n', ' ')
        if len(entry.content) > 100:
            content_preview += "..."
        print(f"   Content: {content_preview}")
        if entry.tags:
            print(f"   Tags: {', '.join(entry.tags)}")
        print()


def handle_memory_export(manager, args):
    """Export memories to file."""
    count = manager.export_memories(args.output)
    print(f"Exported {count} memories to {args.output}")


def handle_memory_import(manager, args):
    """Import memories from file."""
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    count = manager.import_memories(args.input)
    print(f"Imported {count} memories from {args.input}")


def handle_memory_prune(manager, args):
    """Prune old or low-importance memories."""
    max_age = getattr(args, 'max_age', None)
    max_entries = getattr(args, 'max_entries', None)
    min_importance = getattr(args, 'min_importance', None)
    
    if not any([max_age, max_entries, min_importance]):
        print("Performing automatic pruning...")
        manager.auto_prune()
        print("Done.")
    else:
        deleted = manager.prune(
            max_age_days=max_age,
            max_entries=max_entries,
            min_importance=min_importance,
        )
        print(f"Pruned {deleted} memory entries")


def handle_memory_clear(manager, args):
    """Clear all memories."""
    if not args.yes:
        confirm = input("Are you sure you want to clear ALL memories? This cannot be undone. [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    
    manager.clear_all()
    print("All memories have been cleared.")


if __name__ == "__main__":
    main()
