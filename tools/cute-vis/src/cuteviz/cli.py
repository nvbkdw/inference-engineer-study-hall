import argparse
import sys

from .model import Capture


def main():
    parser = argparse.ArgumentParser(
        prog="cuteviz", description="Inspect compile-time CuTeDSL captures locally"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Open the local editor and capture viewer")
    serve.add_argument("capture", nargs="?", help="Optional capture to inspect immediately")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--open", action="store_true", help="Open the browser")
    serve.add_argument("--read-only", action="store_true", help="Disable Python execution")
    serve.add_argument("--workdir", help="Project directory for Python imports (default: cwd)")
    serve.add_argument("--output-dir", help="Run files directory (default: WORKDIR/.cuteviz)")
    serve.add_argument("--timeout", type=float, default=120, help="Compilation limit in seconds")
    validate = sub.add_parser("validate", help="Validate a portable capture")
    validate.add_argument("capture")
    args = parser.parse_args()
    try:
        data = Capture.load(args.capture) if args.capture else None
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Invalid capture: {exc}\n")
    if args.command == "validate":
        print(f"Valid schema {data.schema_version}: {len(data.objects)} objects, {data.outcome}")
        return
    import uvicorn

    from .server import create_app

    if not 1 <= args.port <= 65535:
        parser.exit(2, "Port must be between 1 and 65535\n")
    if not 0 < args.timeout <= 3600:
        parser.exit(2, "Timeout must be greater than zero and at most 3600 seconds\n")
    from pathlib import Path

    if args.workdir and not Path(args.workdir).is_dir():
        parser.exit(2, "Workdir must be an existing directory\n")
    url = f"http://127.0.0.1:{args.port}"
    print(f"CuteViz: {url} — compile-time observations", flush=True)
    if args.open:
        import threading
        import webbrowser

        timer = threading.Timer(1, lambda: webbrowser.open(url))
        timer.daemon = True
        timer.start()
    uvicorn.run(
        create_app(
            data,
            enabled=not args.read_only,
            workdir=args.workdir,
            output_dir=args.output_dir,
            timeout=args.timeout,
        ),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    sys.exit(main())
