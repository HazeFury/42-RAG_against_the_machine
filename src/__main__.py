"""Entry point for uv run python -m src <command> [options]."""

import sys

try:
    import fire
    from src.cli import RAGCLI

    def main() -> None:
        """Run Python Fire CLI interface."""
        fire.Fire(RAGCLI)

except ImportError:

    def main() -> None:
        """Print missing dependency error and exit."""
        print(
            "Missing dependency. Please run 'uv sync'.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
