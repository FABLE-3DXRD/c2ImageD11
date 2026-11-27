#!/usr/bin/env python
"""Check that every C function definition matches the wrapper's prototype.

The generated wrapper declares each C function from its C2PY signature
(``extern ...`` lines in lib/interface/_cImageD11_wrapper.c) but never sees
the definition, so the compiler cannot compare them.  A mismatch can still
work by luck on one ABI and break on another: ``coverlaps`` took ``int``
where the wrapper passed ``intptr_t`` in a stack argument, which was fine
on Linux and x86_64 but read garbage on Apple arm64.

This script recompiles every source in a meson build directory with
``-fsyntax-only`` and the wrapper's prototypes force-included, then reads
gcc's "conflicting types" errors.  Differences that do not change the
calling convention or the pointed-to element type are accepted:
qualifiers (const, restrict) and array spellings (``double x[]``,
``double (*x)[3]`` and ``double *x`` all count as ``double *``).  Anything
else, including any error this script cannot parse, fails.

Needs gcc (clang's notes do not print the previous type).

Usage:  python tools/check_prototypes.py build/libc2ImageD11
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import io
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WRAPPER = os.path.join(HERE, '..', 'lib', 'interface', '_cImageD11_wrapper.c')

# gcc 11+ formats.  C: error + note with both types (optionally {aka ...}).
C_ERR = re.compile(r"error: conflicting types for '(\w+)'; have '(.*?)'(?: \{aka '(.*?)'\})?$")
C_NOTE = re.compile(r"note: previous declaration of '(\w+)' with type '(.*?)'(?: \{aka '(.*?)'\})?$")
# C++ (extern "C" redeclared with different parameters).
CXX_ERR = re.compile(r"error: conflicting declaration of C function '(.*)'$")
CXX_NOTE = re.compile(r"note: previous declaration '(.*)'$")
# Per-parameter notes gcc adds after a C conflict; informational only.
PARAM_NOTE = re.compile(r"note: previously declared as ")


def normalize(t):
    """Reduce a function type to what matters for the call."""
    t = re.sub(r'\b(const|volatile|restrict|__restrict__|__restrict)\b', ' ', t)
    t = re.sub(r'\(\s*\*\s*\)\s*\[\d*\]', '*', t)   # T (*)[N] -> T *
    t = re.sub(r'\[\d*\]', '*', t)                  # T [N]    -> T *
    return re.sub(r'\s+', '', t)


def compile_args(entry):
    if 'arguments' in entry:
        args = list(entry['arguments'])
    else:
        args = shlex.split(entry['command'])
    out = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ('-o', '-MF', '-MQ', '-MT'):
            skip = True
            continue
        if a in ('-c', '-MD', '-MMD'):
            continue
        out.append(a)
    return out


def check_output(stderr, src):
    """Return (n_benign, problems) for one compiler run."""
    lines = [l.rstrip() for l in stderr.splitlines()]
    benign = 0
    problems = []
    pending = None   # (kind, name, type) waiting for its note
    for l in lines:
        m = C_ERR.search(l)
        if m:
            pending = ('c', m.group(1), m.group(3) or m.group(2))
            continue
        m = CXX_ERR.search(l)
        if m:
            pending = ('cxx', None, m.group(1))
            continue
        m = C_NOTE.search(l)
        if m and pending and pending[0] == 'c' and pending[1] == m.group(1):
            have, want = pending[2], m.group(3) or m.group(2)
            if normalize(have) == normalize(want):
                benign += 1
            else:
                problems.append('%s: %s\n    definition: %s\n    wrapper:    %s'
                                % (src, m.group(1), have, want))
            pending = None
            continue
        m = CXX_NOTE.search(l)
        if m and pending and pending[0] == 'cxx':
            have, want = pending[2], m.group(1)
            if normalize(have) == normalize(want):
                benign += 1
            else:
                problems.append('%s:\n    definition: %s\n    wrapper:    %s'
                                % (src, have, want))
            pending = None
            continue
        if PARAM_NOTE.search(l):
            continue
        if 'error:' in l:
            problems.append('%s: unexpected compiler error:\n    %s' % (src, l))
    if pending:
        problems.append('%s: conflict without a matching note: %s' % (src, pending[2]))
    return benign, problems


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    builddir = os.path.abspath(argv[1])
    with io.open(os.path.join(builddir, 'compile_commands.json'), encoding='utf-8') as f:
        entries = json.load(f)
    with io.open(WRAPPER, encoding='utf-8') as f:
        protos = [l for l in f.read().splitlines()
                  if l.startswith('extern ') and not l.startswith('extern "C"')]
    if not protos:
        print('No extern prototypes found in %s' % WRAPPER)
        return 1

    workdir = os.path.join(builddir, 'check_prototypes')
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    body = '\n'.join(protos) + '\n'
    hdr_c = os.path.join(workdir, 'protos_c.h')
    hdr_cxx = os.path.join(workdir, 'protos_cxx.h')
    with io.open(hdr_c, 'w', encoding='utf-8') as f:
        f.write('#include <stdint.h>\n' + body)
    with io.open(hdr_cxx, 'w', encoding='utf-8') as f:
        f.write('#include <stdint.h>\nextern "C" {\n' + body + '}\n')

    nfiles = 0
    nbenign = 0
    problems = []
    for e in entries:
        src = e['file']
        base = os.path.basename(src)
        if base.startswith('_cImageD11_wrapper') or base.startswith('c2py_runtime'):
            continue
        if not (src.endswith('.c') or src.endswith('.cpp')):
            continue
        hdr = hdr_cxx if src.endswith('.cpp') else hdr_c
        cmd = compile_args(e) + ['-fsyntax-only', '-fdiagnostics-color=never',
                                 '-include', hdr]
        p = subprocess.Popen(cmd, cwd=e['directory'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = p.communicate()
        err = err.decode('utf-8', 'replace')
        name = os.path.relpath(os.path.join(e['directory'], src))
        b, probs = check_output(err, name)
        if p.returncode != 0 and b == 0 and not probs:
            probs = ['%s: compiler failed:\n%s' % (name, err)]
        nfiles += 1
        nbenign += b
        problems.extend(probs)

    print('Checked %d sources against %d wrapper prototypes '
          '(%d differ only in qualifiers/array spelling).'
          % (nfiles, len(protos), nbenign))
    if problems:
        print('\nDefinitions that do not match the wrapper:\n')
        print('\n'.join(problems))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
