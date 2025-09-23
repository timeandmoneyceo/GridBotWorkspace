import os
import time
import tempfile
from typing import Dict, Any

from .automated_file_editor import SafeFileEditor


def write_file(path: str, content: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def read_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def scenario_function() -> Dict[str, Any]:
    original = (
        '''
# test module

def compute(a, b):
    """Add two numbers"""
    res = a + b
    return res
'''
    ).strip()

    optimized = (
        '''
def compute(a, b):
    """Multiply two numbers (optimized)"""
    res = a * b
    return res
'''
    ).strip()

    return {
        'name': 'module_function',
        'original': original,
        'optimized': optimized,
        'target_name': 'compute',
        'expect': 'Multiply two numbers (optimized)'
    }


def scenario_class_method() -> Dict[str, Any]:
    original = (
        '''
class Greeter:
    @staticmethod
    def greet(name: str) -> str:
        """Return a greeting"""
        return f"Hello {name}"
'''
    ).strip()

    # Provide full class with optimized method for replacement
    optimized = (
        '''
class Greeter:
    @staticmethod
    def greet(name: str) -> str:
        """Return a more enthusiastic greeting"""
        return f"Hello, {name}! Welcome!"
'''
    ).strip()

    return {
        'name': 'class_method',
        'original': original,
        'optimized': optimized,
        'target_name': 'greet',
        'expect': 'Welcome!'
    }


def run_case(editor: SafeFileEditor, case: Dict[str, Any]) -> Dict[str, Any]:
    tmpdir = tempfile.mkdtemp(prefix='editor_bench_')
    path = os.path.join(tmpdir, f"{case['name']}.py")
    write_file(path, case['original'])

    # Peek ranking before applying
    ranking = editor._evaluate_edit_strategies(path, case['optimized'], {'target_name': case['target_name'], 'original_code': case['original']})
    rank_list = [(s['name'], round(s['score'], 2)) for s in ranking]

    t0 = time.perf_counter()
    result = editor.edit_file_content_serena(
        file_path=path,
        optimized_code=case['optimized'],
        context={'target_name': case['target_name'], 'original_code': case['original']}
    )
    dt = (time.perf_counter() - t0) * 1000.0

    final = read_file(path)
    # verify by expected substring presence
    expected = case.get('expect')
    success = result.get('success', False) and (expected in final if expected else True)
    return {
        'path': path,
        'ranking': rank_list,
        'result': result,
        'success': success,
        'elapsed_ms': round(dt, 2),
    }


def main():
    editor = SafeFileEditor(validate_syntax=True, use_serena=False)  # force non-Serena for local benchmark

    cases = [scenario_function(), scenario_class_method()]
    summary = []
    for case in cases:
        out = run_case(editor, case)
        print(f"\n=== Case: {case['name']} ===")
        print("Ranking (name, score):", out['ranking'])
        print("Result:", {k: out['result'].get(k) for k in ['success', 'method', 'error']})
        print("Elapsed (ms):", out['elapsed_ms'])
        print("Success Verified:", out['success'])
        summary.append(out)

    # Overall
    ok = all(o['success'] for o in summary)
    print("\n=== Overall ===")
    print("All cases succeeded:", ok)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
