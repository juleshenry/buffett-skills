import ast
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
REFERENCES_DIR = ROOT_DIR / "skills" / "buffett" / "references"
SRC_DIR = ROOT_DIR / "src"

END_TO_END_CALL_HINTS = (
    "fetch_",
    "query_",
    "analyze_",
    "process_",
)

END_TO_END_EVALUATOR_CALLS = {
    "Compounding.evaluate",
    "IntrinsicValue.evaluate",
    "IntrinsicValueEstimation.evaluate",
    "MarginOfSafety.evaluate",
    "ShareBuybackAnalysis.evaluate",
}

PUBLIC_MARKET_CALLS = {
    "yf.download",
}


def get_evaluate_method(class_node):
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            return node
    return None


def raises_not_implemented(function_node):
    for expr in function_node.body:
        if isinstance(expr, ast.Raise):
            if isinstance(expr.exc, ast.Call) and getattr(expr.exc.func, "id", "") == "NotImplementedError":
                return True
    return False


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""


def uses_end_to_end_evidence(function_node):
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue

        call_name = _call_name(node.func)
        if any(hint in call_name for hint in END_TO_END_CALL_HINTS):
            return True
        if call_name in END_TO_END_EVALUATOR_CALLS:
            return True
        if call_name in {
            "yf.Ticker",
            *PUBLIC_MARKET_CALLS,
            "requests.get",
            "requests.post",
            "urllib.request.urlopen",
            "pipeline",
        }:
            return True
    return False


def classify_evaluator(class_node):
    evaluate_node = get_evaluate_method(class_node)
    if evaluate_node is None:
        return "missing"
    if raises_not_implemented(evaluate_node):
        return "stub"
    if uses_end_to_end_evidence(evaluate_node):
        return "end_to_end"
    return "input_wrapper"


def get_markdown_headers(directory):
    headers = {}
    for filepath in sorted(directory.glob("*.md")):
        headers[filepath.name] = []
        for line in filepath.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                headers[filepath.name].append(line[3:].strip())
    return headers


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
                    mappings[match.group(1).strip()] = (node.name, classify_evaluator(node))
    return mappings


def main():
    headers_by_file = get_markdown_headers(REFERENCES_DIR)
    class_mappings = get_documented_classes(SRC_DIR)
    normalized_mappings = {key.lower(): value for key, value in class_mappings.items()}

    print("# Buffett Skills True Class Coverage Report\n")

    total_headers = 0
    total_covered = 0
    total_wrappers = 0
    total_stubs = 0
    for filename, headers in headers_by_file.items():
        print(f"### {filename}")
        for header in headers:
            total_headers += 1
            matched = normalized_mappings.get(header.lower())
            if matched:
                class_name, classification = matched
                if classification == "end_to_end":
                    print(f"OK {header} -> class {class_name} (end-to-end)")
                    total_covered += 1
                elif classification == "input_wrapper":
                    print(f"WARN {header} -> class {class_name} (Input wrapper only)")
                    total_wrappers += 1
                elif classification == "stub":
                    print(f"WARN {header} -> class {class_name} (Stub only)")
                    total_stubs += 1
                else:
                    print(f"MISS {header} -> class {class_name} (Missing evaluate method)")
            else:
                print(f"MISS {header} (Missing class implementation)")
        print()

    print(f"True Coverage: {total_covered} / {total_headers} ({total_covered / total_headers * 100:.2f}%)")
    print(f"Input Wrappers: {total_wrappers}")
    print(f"Stubs: {total_stubs}")


if __name__ == "__main__":
    main()
