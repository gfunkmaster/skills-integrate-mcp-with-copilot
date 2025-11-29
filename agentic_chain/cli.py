"""
Command-line interface for the Agentic Chain package.
"""

import argparse
import json
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

  # Export results to JSON
  agentic-chain solve /path/to/project --issue-file issue.json --output result.json
        """
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
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    setup_logging(getattr(args, 'verbose', False))
    
    if args.command == "analyze":
        handle_analyze(args)
    elif args.command == "solve":
        handle_solve(args)


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
        
    chain = AgenticChain(project_path=str(project_path))
    
    try:
        result = chain.solve_issue(issue_data)
    except Exception as e:
        print(f"Error solving issue: {e}")
        sys.exit(1)
        
    if args.summary:
        print(chain.get_solution_summary())
        
    if args.output:
        chain.export_result(args.output)
        print(f"Full results saved to {args.output}")
    elif not args.summary:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
