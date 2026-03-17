#!/usr/bin/env python3
"""
Script to find problematic JSON lines in a JSONL file.
Tests each line individually with both Python's json module and PyArrow.
"""

import io
import json
import sys

try:
    import pyarrow.json as paj
except ImportError:
    paj = None
    print("Warning: PyArrow not available, will only test with Python json module")


def test_line_python_json(line, line_num):
    """Test if a line can be parsed by Python's json module."""
    try:
        json.loads(line)
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Line {line_num}: Python JSON error: {e}"


def test_line_pyarrow(line, line_num):
    """Test if a line can be parsed by PyArrow's JSON reader."""
    if paj is None:
        return True, None
    try:
        # PyArrow expects bytes
        paj.read_json(io.BytesIO(line.encode("utf-8")))
        return True, None
    except Exception as e:
        return False, f"Line {line_num}: PyArrow error: {e}"


def analyze_line(line, line_num):
    """Analyze a line for problematic characters."""
    issues = []

    # Check for non-ASCII characters
    for i, c in enumerate(line):
        code = ord(c)
        if code > 127:
            context_start = max(0, i - 10)
            context_end = min(len(line), i + 10)
            issues.append(
                f"  Position {i}: Non-ASCII char {repr(c)} (U+{code:04X}) context: {repr(line[context_start:context_end])}"
            )
        # Check for control characters (except newline, tab)
        elif code < 32 and c not in ["\n", "\t", "\r"]:
            context_start = max(0, i - 10)
            context_end = min(len(line), i + 10)
            issues.append(
                f"  Position {i}: Control char {repr(c)} (0x{code:02X}) context: {repr(line[context_start:context_end])}"
            )
        # Check for backslashes that might be escape issues
        elif c == "\\":
            context_start = max(0, i - 5)
            context_end = min(len(line), i + 10)
            issues.append(
                f"  Position {i}: Backslash context: {repr(line[context_start:context_end])}"
            )

    return issues


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_bad_json.py <path_to_jsonl_file> [--verbose]")
        print(
            "       python find_bad_json.py <path_to_jsonl_file> --line <line_number>"
        )
        sys.exit(1)

    json_file = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Check for specific line analysis
    if "--line" in sys.argv:
        line_idx = sys.argv.index("--line")
        if line_idx + 1 < len(sys.argv):
            specific_line = int(sys.argv[line_idx + 1])
        else:
            print("Error: --line requires a line number")
            sys.exit(1)
    else:
        specific_line = None

    print(f"Scanning {json_file}...")
    print()

    bad_lines = []
    total_lines = 0

    with open(json_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total_lines = line_num
            line = line.rstrip("\n\r")

            if not line:
                continue

            # If specific line requested, only analyze that line
            if specific_line is not None:
                if line_num != specific_line:
                    continue

                print(f"=== Detailed analysis of line {line_num} ===")
                print(f"Line length: {len(line)} characters")
                print(f"First 100 chars: {repr(line[:100])}")
                print(f"Last 100 chars: {repr(line[-100:])}")
                print()

                # Test with Python json
                ok, error = test_line_python_json(line, line_num)
                if ok:
                    print("✓ Python json.loads: OK")
                else:
                    print(f"✗ {error}")

                # Test with PyArrow
                ok, error = test_line_pyarrow(line, line_num)
                if ok:
                    print("✓ PyArrow JSON reader: OK")
                else:
                    print(f"✗ {error}")

                print()
                print("Character analysis:")
                issues = analyze_line(line, line_num)
                if issues:
                    for issue in issues[:50]:  # Limit output
                        print(issue)
                    if len(issues) > 50:
                        print(f"  ... and {len(issues) - 50} more issues")
                else:
                    print("  No suspicious characters found")

                sys.exit(0)

            # Normal scanning mode
            # Test with Python json first (faster)
            ok, error = test_line_python_json(line, line_num)
            if not ok:
                bad_lines.append((line_num, "python_json", error, line))
                print(f"✗ Line {line_num}: Python JSON parse error")
                if verbose:
                    issues = analyze_line(line, line_num)
                    for issue in issues[:10]:
                        print(issue)
                continue

            # Test with PyArrow
            ok, error = test_line_pyarrow(line, line_num)
            if not ok:
                bad_lines.append((line_num, "pyarrow", error, line))
                print(f"✗ Line {line_num}: PyArrow parse error")
                if verbose:
                    issues = analyze_line(line, line_num)
                    for issue in issues[:10]:
                        print(issue)
                continue

            if verbose and line_num % 10000 == 0:
                print(f"  Processed {line_num} lines...")

    print()
    print(f"=== Summary ===")
    print(f"Total lines scanned: {total_lines}")
    print(f"Bad lines found: {len(bad_lines)}")

    if bad_lines:
        print()
        print("Bad lines:")
        for line_num, parser, error, line in bad_lines[:20]:
            print(f"  Line {line_num} ({parser}): {error[:100]}")
            print(f"    First 80 chars: {repr(line[:80])}")

        if len(bad_lines) > 20:
            print(f"  ... and {len(bad_lines) - 20} more bad lines")

        print()
        print("To analyze a specific line in detail, run:")
        print(f"  python find_bad_json.py {json_file} --line <line_number>")
    else:
        print()
        print("All lines parse OK with both Python json and PyArrow!")
        print()
        print("The issue might be with how PyArrow reads the file in batches.")
        print("Try testing with a batch simulation:")
        print()
        print("You could also try converting to a clean JSONL format:")
        print('  python -c "')
        print("import json")
        print(
            f"with open('{json_file}', 'r') as f_in, open('clean.jsonl', 'w') as f_out:"
        )
        print("    for line in f_in:")
        print("        obj = json.loads(line)")
        print("        f_out.write(json.dumps(obj, ensure_ascii=False) + '\\n')\"")


if __name__ == "__main__":
    main()
