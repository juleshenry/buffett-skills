import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import coverage_check


def parse_first_class(source: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node
    raise AssertionError("No class found")


class TestCoverageCheck(unittest.TestCase):
    def test_classify_stub_evaluator(self):
        class_node = parse_first_class(
            '''
class Example:
    """Heuristic: Example"""
    def evaluate(self, ticker):
        raise NotImplementedError("todo")
'''
        )
        self.assertEqual(coverage_check.classify_evaluator(class_node), "stub")

    def test_classify_input_wrapper(self):
        class_node = parse_first_class(
            '''
class Example:
    """Heuristic: Example"""
    def evaluate(self, positions):
        return {"count": len(positions)}
'''
        )
        self.assertEqual(coverage_check.classify_evaluator(class_node), "input_wrapper")

    def test_classify_end_to_end_via_fetcher(self):
        class_node = parse_first_class(
            '''
class Example:
    """Heuristic: Example"""
    def evaluate(self, ticker):
        data = self._fetch_real_data(ticker)
        return data
'''
        )
        self.assertEqual(coverage_check.classify_evaluator(class_node), "end_to_end")

    def test_classify_end_to_end_via_requests(self):
        class_node = parse_first_class(
            '''
class Example:
    """Heuristic: Example"""
    def evaluate(self, ticker):
        response = requests.get("https://example.com")
        return {"ok": bool(response)}
'''
        )
        self.assertEqual(coverage_check.classify_evaluator(class_node), "end_to_end")

    def test_classify_missing_evaluate(self):
        class_node = parse_first_class(
            '''
class Example:
    """Heuristic: Example"""
    pass
'''
        )
        self.assertEqual(coverage_check.classify_evaluator(class_node), "missing")


if __name__ == "__main__":
    unittest.main()
