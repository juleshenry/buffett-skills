import ast
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
REFERENCES_DIR = ROOT_DIR / "skills" / "buffett" / "references"
SRC_DIR = ROOT_DIR / "src"


def get_markdown_headers(directory):
    headers = {}
    for filepath in sorted(directory.glob("*.md")):
        headers[filepath.name] = []
        for line in filepath.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                headers[filepath.name].append(line[3:].strip())
    return headers


def is_implemented(class_node):
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            for expr in node.body:
                if isinstance(expr, ast.Raise):
                    if isinstance(expr.exc, ast.Call) and getattr(expr.exc.func, "id", "") == "NotImplementedError":
                        return False
            return True
    return False


def get_documented_classes(directory):
    mappings = {}
    for filepath in directory.glob("*.py"):
        try:
            tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if not docstring:
                    continue
                match = re.search(r"Heuristic:\s*(.+)", docstring)
                if match:
                    mappings[match.group(1).strip()] = (node.name, is_implemented(node))
    return mappings


def main():
    headers_by_file = get_markdown_headers(REFERENCES_DIR)
    class_mappings = get_documented_classes(SRC_DIR)
    normalized_mappings = {key.lower(): value for key, value in class_mappings.items()}

    print("# Buffett Skills True Class Coverage Report\n")

    total_headers = 0
    total_covered = 0
    for filename, headers in headers_by_file.items():
        print(f"### {filename}")
        for header in headers:
            total_headers += 1
            matched = normalized_mappings.get(header.lower())
            if matched:
                class_name, implemented = matched
                if implemented:
                    print(f"OK {header} -> class {class_name}")
                    total_covered += 1
                else:
                    print(f"WARN {header} -> class {class_name} (Stub only)")
            else:
                print(f"MISS {header} (Missing class implementation)")
        print()

    print(f"True Coverage: {total_covered} / {total_headers} ({total_covered / total_headers * 100:.2f}%)")


if __name__ == "__main__":
    main()
