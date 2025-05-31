import inspect
import importlib
import argparse
from typing import get_type_hints, Any
import sys
import os

sys.path.insert(0, os.getcwd())


def extract_doc(method, method_name: str) -> str:
    sig = inspect.signature(method)
    hints = get_type_hints(method)
    doc = f"## `{method_name}`\n\n"
    doc += "The method defines configuration or behavior.\n\n"
    doc += "---\n### Parameters\n"

    for name, param in sig.parameters.items():
        if name in {"self", "*args", "**kwargs"}:
            continue

        param_type = hints.get(name, "Any")
        param_type_str = param_type.__name__ if hasattr(param_type, "__name__") else str(param_type).replace("typing.", "")
        if param.default is inspect.Parameter.empty:
            default_str = ""
        else:
            default_str = f", *optional*, defaults to `{param.default}`" if param.default is not None else ", *optional*"

        doc += f"#### `{name}`\n"
        doc += f"`{param_type_str}`{default_str}\n\n"
        doc += f"Description for `{name}`.\n\n---\n"

    return doc


def get_all_methods(cls, only_defined=False):
    if only_defined:
        return {
            name: obj
            for name, obj in cls.__dict__.items()
            if inspect.isfunction(obj)
        }
    else:
        return {
            name: method
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
            if not name.startswith("_") or name == "__init__"
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown doc from class method signature.")
    parser.add_argument("--module", required=True, help="Python module path, e.g., gliclass.model")
    parser.add_argument("--class", dest="class_name", required=True, help="Class name to extract methods from")
    parser.add_argument("--method", default="__init__", help="Method name, or use 'all' / 'all_new'")
    parser.add_argument("--output", help="Output markdown file")

    args = parser.parse_args()

    mod = importlib.import_module(args.module)
    cls = getattr(mod, args.class_name)

    if args.method == "all":
        methods = get_all_methods(cls, only_defined=False)
        markdown_doc = "\n\n".join(extract_doc(m, name) for name, m in methods.items())
        output_path = args.output or f"{args.class_name}_all_methods.md"
    elif args.method == "all_new":
        methods = get_all_methods(cls, only_defined=True)
        markdown_doc = "\n\n".join(extract_doc(m, name) for name, m in methods.items())
        output_path = args.output or f"{args.class_name}_all_new_methods.md"
    else:
        method = getattr(cls, args.method)
        markdown_doc = extract_doc(method, args.method)
        output_path = args.output or f"{args.class_name}_{args.method}.md"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_doc)


if __name__ == "__main__":
    main()
