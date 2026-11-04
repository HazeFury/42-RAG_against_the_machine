"""Entry point for uv run python -m src <command> [options]."""

import sys

try:
    import fire

    from src.cli import RAGCLI

    def main() -> None:
        """Run Python Fire CLI interface."""
        try:
            fire.Fire(RAGCLI)
        except KeyboardInterrupt:
            print(
                "\n\033[93m[INFO] User deliberately interrupted"
                " the program\033[0m",
                file=sys.stderr,
            )

            sys.exit(0)

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
