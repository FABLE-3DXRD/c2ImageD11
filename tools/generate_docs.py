#!/usr/bin/env python3
"""generate_docs.py -- Generate API documentation from live module introspection.

Reads every exported C function from c2ImageD11, produces `docs/api/*.md`
pages with help() output, C2PY_BEGIN doc/params, and (when present)
function-specific example and benchmark output.

Usage:
    python tools/generate_docs.py              # generate docs/api/*.md
    python tools/generate_docs.py --dry-run    # validate only, no file writes
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import ast
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNC_DIR = os.path.join(HERE, "lib", "functions")
API_DIR = os.path.join(HERE, "docs", "api")
BENCH_DIR = os.path.join(HERE, "docs", "bench")

# Order in which categories appear in the API index / mkdocs nav.
BROWSE_CATEGORIES = [
    "Image corrections",
    "Peak labelling",
    "Sparse",
    "Scattering geometry",
    "Indexing",
    "Crystallography",
    "Array operations",
    "Runtime",
]

# Guide page that explains the AoS vs SoA layout convention.  A function
# whose C2PY_BEGIN block has a "layout" key gets a reference to this page.
LAYOUT_GUIDE = "guide/layout.md"


def _load_function_meta():
    """Map exported function name -> metadata from its primary C2PY_BEGIN block.

    Scans every .c/.cpp C2PY_BEGIN block and, for each function name, prefers
    the block that carries a 'doc'/'params' (the descriptive primary block)
    over bare variant blocks. Returns {name: {"doc", "params", "checks",
    "tags", "layout", "path"}}.
    """
    import ast

    result = {}
    for root, dirs, files in os.walk(FUNC_DIR):
        for f in sorted(files):
            if not f.endswith((".c", ".cpp")):
                continue
            fp = os.path.join(root, f)
            with open(fp) as fh:
                text = fh.read()
            begin_re = re.compile(r"C2PY_BEGIN")
            end_re = re.compile(r"C2PY_END")
            lines = text.splitlines(True)
            i = 0
            while i < len(lines):
                if begin_re.search(lines[i]):
                    i += 1
                    content_lines = []
                    while i < len(lines) and not end_re.search(lines[i]):
                        content_lines.append(lines[i])
                        i += 1
                    comment_prefix = re.compile(r"^\s*\*\s?")
                    cleaned = []
                    for ln in content_lines:
                        s = ln.rstrip("\n\r")
                        s = comment_prefix.sub("", s, count=1)
                        cleaned.append(s)
                    block_text = "\n".join(cleaned).strip()
                    if block_text:
                        block_text = re.sub(r"(?<=[\s,\[{:])true(?=\s*[\s,\]\}:])", "True", block_text)
                        block_text = re.sub(r"(?<=[\s,\[{:])false(?=\s*[\s,\]\}:])", "False", block_text)
                        try:
                            obj = ast.literal_eval(block_text)
                        except (ValueError, SyntaxError):
                            obj = None
                        if isinstance(obj, dict) and "py_sig" in obj:
                            m = re.match(r"(\w+)\(", obj["py_sig"])
                            name = m.group(1) if m else None
                            if not name:
                                continue
                            entry = {
                                "doc": obj.get("doc", ""),
                                "params": obj.get("params", {}),
                                "checks": obj.get("checks", []),
                                "tags": obj.get("tags", []),
                                "layout": obj.get("layout", ""),
                                "path": fp,
                            }
                            prev = result.get(name)
                            if prev is None:
                                result[name] = entry
                            else:
                                # Prefer the block that carries descriptive doc/params.
                                if (entry["doc"] or entry["params"]) and not (prev["doc"] or prev["params"]):
                                    result[name] = entry
                    if i < len(lines):
                        i += 1
                else:
                    i += 1
    return result


def _gen_cpu_features_page():
    """Generate docs/api/cpu.md for the _c2py_has/set functions."""
    import c2ImageD11 as ci
    mod = ci._cImageD11

    arches = {
        "x86_64": [("sse4_1", "SSE4.1"), ("avx2", "AVX2"), ("avx512f", "AVX-512F")],
        "arm64": [("asimd", "ASIMD/NEON"), ("sve", "SVE"), ("sve2", "SVE2")],
        "ppc64": [("altivec", "AltiVec"), ("vsx", "VSX")],
    }

    lines = ["# CPU Feature Flags", ""]
    lines.append("Probed at module init by `c2py.h`. Modifiable at runtime for benchmarking.")
    lines.append("")

    for arch, flags in sorted(arches.items()):
        lines.append("## {}".format(arch))
        lines.append("")
        lines.append("| Flag | Supported | Has Function | Set Function |")
        lines.append("|------|-----------|-------------|-------------|")
        for suffix, label in flags:
            has_name = "_c2py_has_{}".format(suffix)
            set_name = "_c2py_set_{}".format(suffix)
            has_fn = getattr(mod, has_name, None)
            set_fn = getattr(mod, set_name, None)
            if has_fn:
                val = has_fn()
                status = "yes (current: {})".format(val) if val else "no"
                lines.append("| {} | {} | `{}()` | `{}(val)` |".format(
                    label, status, has_name, set_name))
        lines.append("")

    return "\n".join(lines)


def _format_help(name, fn):
    """Extract just the Python-level signature line from help(fn)."""
    from io import StringIO
    old = sys.stdout
    s = StringIO()
    try:
        sys.stdout = s
        help(fn)
        sys.stdout = sys.__stdout__
        text = s.getvalue()
    except Exception:
        sys.stdout = old
        return ""
    sys.stdout = old
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and "(" in stripped and ")" in stripped:
            return "**`{}`**".format(stripped)
    return ""


def _overloads_table(c2py_data, func_name):
    """Build a markdown table from c_overloads entries for a function."""
    functions = c2py_data.get("functions", [])
    for f in functions:
        py_sig = f.get("py_sig", "")
        if py_sig.split("(")[0].strip() == func_name:
            overloads = f.get("c_overloads", [])
            if not overloads:
                return None
            lines = []
            lines.append("## Overloads")
            lines.append("")
            lines.append("| C function | Condition |")
            lines.append("|-----------|-----------|")
            for ol in overloads:
                sig = ol.get("sig", "")
                when = ol.get("when", "")
                # Extract just the function name from the sig
                fn_match = re.match(r'^\w+\s+(\w+)\(', sig)
                c_fn = fn_match.group(1) if fn_match else sig[:40]
                # Simplify ISA conditions for readability
                cond = _simplify_condition(when)
                lines.append("| `{}` | {} |".format(c_fn, cond))
            return "\n".join(lines)
    return None


def _load_c2py_data():
    """Parse the assembled .c2py file and return the dict."""
    c2py_path = os.path.join(HERE, "lib", "interface", "_cImageD11.c2py")
    if not os.path.exists(c2py_path):
        return {}
    with open(c2py_path) as f:
        text = f.read()
    # Strip header comments
    lines = text.split("\n")
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    content = "\n".join(lines)
    try:
        return ast.literal_eval(content)
    except Exception:
        return {}


def _simplify_condition(when_expr):
    """Convert a when: expression to a readable label."""
    if not when_expr:
        return "always"
    parts = []
    if "gv.format == 'd'" in when_expr or "gv.format == 'd'" in when_expr:
        parts.append("f64")
    if "gv.format == 'f'" in when_expr:
        parts.append("f32")
    if "gv.shape[1] == 3" in when_expr and "gv.shape[0]" not in when_expr:
        parts.append("AoS")
    if "gv.shape[0] == 3" in when_expr:
        parts.append("SoA")
    if "c2py_amd64_avx512f" in when_expr:
        parts.append("AVX-512")
    elif "c2py_amd64_avx2" in when_expr:
        parts.append("AVX2")
    elif "c2py_amd64_sse4_1" in when_expr:
        parts.append("SSE4.1")
    if "gb.format == 'd'" in when_expr:
        parts.append("f64")
    return " ".join(parts) if parts else "always"
    """Run an examples.py file and capture stdout. Returns (code, output) or None."""
    if not os.path.exists(examples_path):
        return None
    result = subprocess.run(
        [sys.executable, examples_path],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.dirname(examples_path))
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _run_bench(bench_path):
    """Run a bench.py --json and capture stdout. Returns parsed JSON dict or None."""
    if not os.path.exists(bench_path):
        return None
    result = subprocess.run(
        [sys.executable, bench_path, "--json"],
        capture_output=True, text=True, timeout=120,
        cwd=os.path.dirname(bench_path))
    if result.returncode != 0:
        return None
    try:
        import json as _json
        return _json.loads(result.stdout.strip())
    except Exception:
        return None


BENCH_DIR = os.path.join(HERE, "docs", "bench")


def _load_bench_data(name):
    """Load committed benchmark JSON for a function. Returns dict or None."""
    path = os.path.join(BENCH_DIR, "{}.json".format(name))
    if not os.path.exists(path):
        return None
    try:
        import json as _json
        with open(path) as f:
            return _json.load(f)
    except (ValueError, IOError):
        return None


def _format_bench_table(data):
    """Format benchmark JSON dict as a markdown table."""
    if not data:
        return None
    lines = []
    date = data.get("generated_at", "unknown date")
    cpu = data.get("cpu_info", "unknown CPU")
    lines.append("Measured on {} ({})".format(date, cpu))
    lines.append("")

    # Look for measurements table
    measurements = data.get("measurements", {})
    if not measurements:
        # Try legacy format: flat keys
        rows = []
        for k, v in sorted(data.items()):
            if k in ("function", "generated_at", "cpu_info", "f2py_baseline"):
                continue
            if isinstance(v, dict):
                rows.append([k] + [str(v.get(rk, "")) for rk in sorted(v.keys())])
        if rows:
            lines.append("| Variant | " + " | ".join(rows[0][1:]) + " |")
            lines.append("|---" + "|" * (len(rows[0]) - 1) + " |")
            for row in rows:
                lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines) if rows else None

    # Structured measurements
    cols = set()
    ng_val = None
    for v in measurements.values():
        if isinstance(v, dict):
            cols.update(v.keys())
            if v.get("ng"):
                ng_val = v["ng"]
    cols = sorted(c for c in cols if c != "ng")
    col_headers = ["Variant"] + cols
    header_line = "| " + " | ".join(col_headers) + " |"
    sep_line = "|" + "---|" * len(col_headers)
    lines.append(header_line)
    lines.append(sep_line)
    for variant, v in sorted(measurements.items()):
        row = [variant]
        for c in cols:
            row.append(str(v.get(c, "")) if isinstance(v, dict) else str(v))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _get_version():
    """Return (version, short_sha, full_sha) from the project."""
    version = "unknown"
    short_sha = "unknown"
    full_sha = "unknown"
    init_path = os.path.join(HERE, "c2ImageD11", "__init__.py")
    try:
        with open(init_path) as f:
            for line in f:
                m = re.match(r'^__version__\s*=\s*"(.+)"\s*$', line)
                if m:
                    version = m.group(1)
                    break
    except IOError:
        pass
    try:
        full_sha = subprocess.check_output(
            ["git", "-C", HERE, "rev-parse", "HEAD"],
            text=True, timeout=5).strip()
        short_sha = subprocess.check_output(
            ["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
            text=True, timeout=5).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        pass
    return version, short_sha, full_sha


def _build_module_toc(functions, meta=None):
    """Return a grouped table-of-contents, one section per category.

    A function can belong to several categories and appears under each.
    Functions without a known tag fall into an "Uncategorized" section.
    """
    meta = meta or _load_function_meta()
    lines = ["# API Index", ""]
    lines.append("{} C functions exported by `c2ImageD11`.".format(len(functions)))
    lines.append("")
    lines.append("Grouped by theme. A function may appear in more than one category.")
    lines.append("")

    def _tags(name):
        entry = meta.get(name)
        return list(entry.get("tags") or []) if entry else []

    known = set(BROWSE_CATEGORIES)
    # Assign each function to its categories (plus an implicit "Uncategorized").
    buckets = {}
    for name in sorted(functions):
        tags = [t for t in _tags(name) if t in known]
        if not tags:
            tags = ["Uncategorized"]
        for t in tags:
            buckets.setdefault(t, []).append(name)

    for cat in list(BROWSE_CATEGORIES) + ["Uncategorized"]:
        names = buckets.get(cat)
        if not names:
            continue
        lines.append("## {0}".format(cat))
        lines.append("")
        for name in sorted(names):
            lines.append("- [{}]({}.md)".format(name, name))
        lines.append("")
    return "\n".join(lines)



def generate(dry_run=False):
    """Generate all API documentation pages.

    Args:
        dry_run: If True, validate only; do not write files.
    Returns:
        Number of functions documented.
    """
    import c2ImageD11 as ci
    mod = ci._cImageD11

    # Identify C functions: not starting with _, callable, not the Python imports
    skip = {"absolute_import", "division", "print_function", "unicode_literals",
            "check_multiprocessing", "np", "os", "sys", "platform", "warnings",
            "importlib"}
    functions = sorted(n for n in dir(mod)
                       if not n.startswith("_")
                       and n not in skip
                       and callable(getattr(mod, n, None)))

    meta = _load_function_meta()
    c2py_data = _load_c2py_data()
    version, short_sha, full_sha = _get_version()
    pages = {}
    for name in functions:
        fn = getattr(mod, name)
        sig_line = _format_help(name, fn)
        lines = ["# `{}`".format(name), ""]
        if sig_line:
            lines.append(sig_line)
            lines.append("")

        entry = meta.get(name)
        if entry is not None:
            tags = entry.get("tags") or []
            layout = entry.get("layout") or ""
            if tags:
                lines.append("**Categories:** " + ", ".join("`{}`".format(t) for t in tags))
                lines.append("")
            if layout:
                lines.append(
                    "**Layout:** {}  "
                    "See [AoS/SoA guide]({}).".format(layout, LAYOUT_GUIDE))
                lines.append("")

        # Description and params from C2PY_BEGIN
        if entry is not None:
            doc = entry.get("doc") or ""
            params = entry.get("params") or {}
            checks = entry.get("checks") or []
            if doc:
                lines.append("## Description")
                lines.append("")
                lines.append(doc)
                lines.append("")
            if params:
                lines.append("## Parameters")
                lines.append("")
                lines.append("| Name | Description |")
                lines.append("|------|-------------|")
                for pname, pdesc in sorted(params.items()):
                    lines.append("| `{}` | {} |".format(pname, pdesc))
                lines.append("")
            if checks:
                lines.append("## Constraints")
                lines.append("")
                for c in checks:
                    lines.append("- `{}`".format(c))
                lines.append("")

        # Overloads table from .c2py data
        overloads = _overloads_table(c2py_data, name)
        if overloads:
            lines.append(overloads)
            lines.append("")

        # Example section
        examples_path = os.path.join(FUNC_DIR, name, "examples.py")
        if os.path.exists(examples_path):
            example_output = _run_example(examples_path)
            if example_output:
                lines.append("## Example")
                lines.append("")
                lines.append("```")
                lines.append(example_output)
                lines.append("```")
                lines.append("")

        # Benchmark section (from committed JSON data, never from live bench run)
        bench_data = _load_bench_data(name)
        if bench_data:
            bench_table = _format_bench_table(bench_data)
            if bench_table:
                lines.append("## Performance")
                lines.append("")
                lines.append(bench_table)
                lines.append("")

        pages[name] = "\n".join(lines)

    # Write pages
    version_line = "v{} ({})".format(version, full_sha[:7])
    if dry_run:
        print("  Dry run: would write {} API pages (version {})".format(len(pages), version_line))
    else:
        os.makedirs(API_DIR, exist_ok=True)
        for name, content in sorted(pages.items()):
            path = os.path.join(API_DIR, "{}.md".format(name))
            content += "\n\n---\nBuilt from commit `{}` (v{})".format(full_sha, version)
            with open(path, "w") as f:
                f.write(content)
        print("  Wrote {} function pages to {} (version {})".format(len(pages), API_DIR, version_line))

    # API index
    index_content = _build_module_toc(functions, meta)
    if dry_run:
        print("  Dry run: would write API index")
    else:
        path = os.path.join(API_DIR, "index.md")
        index_content += "\n\n---\nBuilt from commit `{}` (v{})".format(full_sha, version)
        with open(path, "w") as f:
            f.write(index_content)

    # CPU features page
    cpu_content = _gen_cpu_features_page()
    if dry_run:
        print("  Dry run: would write CPU features page")
    else:
        path = os.path.join(API_DIR, "cpu.md")
        with open(path, "w") as f:
            f.write(cpu_content)

    return len(pages)


def write_mkdocs_nav(functions, meta=None):
    """Write docs/mkdocs.yml with a category-grouped nav from the function list."""
    meta = meta or _load_function_meta()
    version, short_sha, full_sha = _get_version()

    def _tags(name):
        entry = meta.get(name)
        return list(entry.get("tags") or []) if entry else []

    known = set(BROWSE_CATEGORIES)
    buckets = {}
    for name in sorted(functions):
        tags = [t for t in _tags(name) if t in known]
        if not tags:
            tags = ["Uncategorized"]
        for t in tags:
            buckets.setdefault(t, []).append(name)

    api_pages = ["- API Index: api/index.md", "- CPU Features: api/cpu.md"]
    for cat in list(BROWSE_CATEGORIES) + ["Uncategorized"]:
        names = buckets.get(cat)
        if not names:
            continue
        sub = ["    - {}: api/{}.md".format(n, n) for n in sorted(names)]
        api_pages.append("- {}:".format(cat))
        api_pages.extend(sub)

    nav = """site_name: c2ImageD11
site_description: Standalone C extensions for ImageD11 (c2py23 binding) -- v{} ({})
repo_url: https://github.com/FABLE-3DXRD/c2ImageD11
repo_name: FABLE-3DXRD/c2ImageD11
docs_dir: .
site_dir: ../site
theme:
  name: material
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - content.code.copy
    - navigation.instant
    - navigation.sections
    - search.highlight

plugins:
  - search

copyright: Built from commit <a href="https://github.com/FABLE-3DXRD/c2ImageD11/commit/{}"><code>{}</code></a> (v{})

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences

nav:
  - Home: index.md
  - Guide:
    - AoS vs SoA Layout: guide/layout.md
    - ISA Variants: guide/variants.md
    - Compiler Selection: guide/compiler.md
  - API Reference:
{}
    """.format(version, short_sha, full_sha, short_sha, version, "\n".join("    " + p for p in api_pages))

    mkdocs_path = os.path.join(HERE, "docs", "mkdocs.yml")
    with open(mkdocs_path, "w") as f:
        f.write(nav)
    print("  Wrote docs/mkdocs.yml")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate c2ImageD11 API documentation")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate without writing files")
    args = p.parse_args()

    print("Generating documentation...")
    count = generate(dry_run=args.dry_run)

    if not args.dry_run:
        # Identify functions for mkdocs nav
        import c2ImageD11 as ci
        mod = ci._cImageD11
        skip = {"absolute_import", "division", "print_function", "unicode_literals",
                "check_multiprocessing", "np", "os", "sys", "platform", "warnings",
                "importlib"}
        functions = sorted(n for n in dir(mod)
                           if not n.startswith("_")
                           and n not in skip
                           and callable(getattr(mod, n, None)))
        write_mkdocs_nav(functions, _load_function_meta())

    print("Done: {} functions documented".format(count))


if __name__ == "__main__":
    main()
