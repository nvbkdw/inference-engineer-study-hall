"""Fresh-process compiler entry point used by the local workbench."""

import importlib.util
import json
import sys
import traceback
from pathlib import Path

from .capture import capture


def resolve(module, name):
    value = module
    for part in name.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"{name} must be callable")
    return value


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    source = Path(config["source_path"])
    # Local project imports behave as they do when running a driver from the project.
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(1, str(source.parent))
    try:
        with capture(
            config["capture_path"],
            target=config["target"] or None,
            parameters={"entry": config["entry"], "args_factory": config["args_factory"]},
        ):
            import cutlass.cute as cute

            spec = importlib.util.spec_from_file_location("cuteviz_user_kernel", source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            entry = resolve(module, config["entry"])
            factory = config["args_factory"]
            # The conventional make_args is optional for a zero-argument entry.
            values = ()
            if factory and not (factory == "make_args" and not hasattr(module, factory)):
                values = resolve(module, factory)()
            kwargs = {}
            if isinstance(values, dict):
                if set(values) - {"args", "kwargs"}:
                    raise TypeError("Argument factory must return {'args': (...), 'kwargs': {...}}")
                kwargs = values.get("kwargs", {})
                values = values.get("args", ())
            if not isinstance(values, (tuple, list)) or not isinstance(kwargs, dict):
                raise TypeError(
                    "Argument factory must return a tuple/list or args/kwargs dictionary"
                )
            if any(not isinstance(key, str) for key in kwargs):
                raise TypeError("Compile keyword argument names must be strings")
            print(f"Compiling {config['entry']}…", flush=True)
            cute.compile(entry, *values, **kwargs)
        print("Capture saved.", flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
