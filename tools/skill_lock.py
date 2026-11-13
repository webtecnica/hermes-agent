"""User skill locks, enforced on every agent write path.

``hermes curator lock <skill>`` (or the Desktop lock switch) records ``locked: true`` beside
``pinned`` in the profile's ``skills/.usage.json``. While a skill is locked the agent cannot change
it: ``skill_manage`` refuses every mutating action, ``write_file``/``patch`` refuse any path inside
the skill's directory, and ``terminal`` refuses commands that write there. The lock record itself
is protected the same way, and the agent cannot run ``hermes curator unlock``. The user's own edits
(Desktop editor, their own editor, the CLI) are not gated.

The terminal check reads the command's write targets (redirections, tee, cp/mv/rm/sed -i and
friends, git in the skill's tree, ``cd`` into it, inline interpreter code naming it). It is a
policy check, not a sandbox: a script that writes there through its own logic is not seen.
"""

from __future__ import annotations

import ast
import glob
import itertools
import os
import re
import shlex
from typing import Iterable

# Verbs whose every path operand is written (or removed/renamed).
_WRITES_EVERY_OPERAND = frozenset({
    "rm", "rmdir", "unlink", "truncate", "shred", "touch", "chmod", "chown", "chgrp", "chattr",
    "setfacl", "mkdir", "mv", "patch", "dos2unix", "unix2dos", "xattr", "ed", "ex", "vi", "vim",
    "nvim", "nano", "emacs"})
# Verbs that write only to their LAST operand (the destination).
_WRITES_LAST_OPERAND = frozenset({"cp", "install", "ln", "rsync", "scp"})
# Verbs that reach INTO a directory operand recursively, so an ancestor of a locked skill counts.
_RECURSIVE = frozenset({"rm", "mv", "chmod", "chown", "chgrp", "chattr", "setfacl", "rsync", "find"})
_INPLACE_EDITORS = frozenset({"sed", "perl", "ruby"})
_GIT_MUTATIONS = frozenset({
    "add", "am", "apply", "checkout", "clean", "commit", "merge", "mv", "pull", "rebase", "reset",
    "restore", "revert", "rm", "stash", "switch", "cherry-pick"})
_INTERPRETERS = re.compile(
    r"^(?:python[0-9.]*|node|nodejs|deno|bun|ruby|perl|php|bash|sh|zsh|dash|ksh|pwsh|powershell)$")
_WRAPPERS = frozenset({"sudo", "env", "command", "builtin", "exec", "nohup", "setsid", "time", "nice", "xargs"})
# Inline code that writes: open() with a write/append/create mode, pathlib/fs write calls, deletes,
# renames, copies, and shell write forms. A plain open(path) / read stays allowed.
_WRITEISH_CODE = re.compile(
    r"open\s*\([^)]*['\"][rbt]*[wax+][^'\"]*['\"]|\.write(?:_text|_bytes|lines)?\s*\(|"
    r"\b(?:unlink|rmtree|removedirs|remove|rmdir|rename|truncate|chmod|chown|mkdir|makedirs|symlink|"
    r"copyfile|copytree|copy2?|move|touch)\s*\(|\bos\.replace\s*\(|\.replace\s*\(\s*(?:Path|['\"]/)|"
    r"(?:write|append)File(?:Sync)?|(?:rm|unlink|rename|mkdir|copyFile|cp)Sync|"
    r"(?:^|[\s;|&])>{1,2}|\bsed\s+-i|\b(?:mv|rm|cp|tee|touch)\s", re.IGNORECASE)
_SEPARATORS = frozenset({";", "&", "&&", "|", "||", "(", ")", "|&", ";;"})
# Any of these in an open() mode string means the file is written.
_WRITE_MODES = "wax+"


def _skill_usage():
    from tools import skill_usage
    return skill_usage


def locked_skill_dirs() -> list[tuple[str, str]]:
    """``(name, realpath)`` of every locked skill in the active profile, from ONE walk of the
    skills dirs (locks are keyed by directory name, so no frontmatter is read)."""
    names = set(_skill_usage().locked_skill_names())
    if not names:
        return []
    from agent.skill_utils import get_all_skills_dirs, is_excluded_skill_path
    found: dict[str, str] = {}
    for base in get_all_skills_dirs():
        if not base.is_dir():
            continue
        for skill_md in base.rglob("SKILL.md"):
            name = skill_md.parent.name
            if name in names and name not in found and not is_excluded_skill_path(skill_md, root=base):
                found[name] = os.path.realpath(skill_md.parent)
    return sorted(found.items())


def is_locked_skill(name: str, locked_names: Iterable[str] | None = None) -> bool:
    """Whether *name* (directory, ``cat/name`` or frontmatter name) is locked. Locks are keyed by
    directory name, while curator rows and some callers carry the frontmatter name."""
    names = set(_skill_usage().locked_skill_names() if locked_names is None else locked_names)
    if not names:
        return False
    if name in names or os.path.basename(name) in names:
        return True
    from tools.skill_manager_tool import _find_skill
    found = _find_skill(name)
    return bool(found) and found["path"].name in names


def _record_path() -> str:
    return os.path.realpath(_skill_usage()._usage_file())


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def skill_refusal(name: str) -> str:
    return (f"Skill '{name}' is locked by the user, so the agent cannot change it: skill_manage, "
            f"write_file, patch and terminal all refuse. Don't retry or work around it (no copying it "
            f"under another name); tell the user what you would change. They can unlock it with "
            f"`hermes curator unlock {name}` or the lock switch in Desktop.")


RECORD_REFUSAL = ("skills/.usage.json holds the user's skill locks, so the agent cannot change it "
                  "while any skill is locked.")
UNLOCK_REFUSAL = ("Only the user can unlock a skill. Ask them to run `hermes curator unlock <skill>` "
                  "themselves or use the lock switch in Desktop.")


def _skills_roots() -> list[str]:
    from agent.skill_utils import get_all_skills_dirs
    return [os.path.realpath(d) for d in get_all_skills_dirs()]


def _hit(path: str, locked: list[tuple[str, str]], record: str, *, ancestor: str = "") -> str | None:
    """Refusal text when *path* is the lock record or inside a locked skill. ``ancestor="any"`` also
    counts a directory ABOVE a locked skill (recursive deletes/moves); ``ancestor="skills"`` counts it
    only inside a skills root (a recursive copy or extraction there can merge into the skill)."""
    real = os.path.realpath(path)
    if real == record or (ancestor == "any" and real != os.sep and _inside(record, real)):
        return RECORD_REFUSAL
    for name, root in locked:
        if _inside(real, root):
            return skill_refusal(name)
        if ancestor and real != os.sep and _inside(root, real) and (
                ancestor == "any" or any(_inside(real, r) for r in _skills_roots())):
            return skill_refusal(name)
    return None


def path_refusal(paths: Iterable[str]) -> str | None:
    """write_file/patch guard: refusal text when any path lands in a locked skill or the lock record."""
    locked = locked_skill_dirs()
    if not locked:
        return None
    record = _record_path()
    for path in paths:
        if path and (reason := _hit(path, locked, record)):
            return reason
    return None


# ── terminal ──────────────────────────────────────────────────────────────────────────────────

_VAR = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _shell_env(assigned: dict[str, str] | None = None) -> dict[str, str]:
    from hermes_constants import get_hermes_home
    return {**os.environ, "HOME": os.path.expanduser("~"), "HERMES_HOME": str(get_hermes_home()),
            **(assigned or {})}


def _expand(word: str, cwd: str, env: dict[str, str] | None = None) -> list[str]:
    """Shell word -> absolute path(s): $VARS (environment + assignments earlier in the command), ~,
    relative to *cwd*; a glob expands against the filesystem the way the shell would (unmatched:
    its literal directory prefix)."""
    env = env if env is not None else _shell_env()
    word = _VAR.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), word)
    home = env["HOME"]
    if word == "~" or word.startswith("~/"):
        word = home + word[1:]
    path = os.path.normpath(os.path.join(cwd, word))
    if not any(ch in word for ch in "*?["):
        return [path]
    if matches := glob.glob(path):
        return matches
    return [path[: min(path.index(ch) for ch in "*?[" if ch in path)].rsplit(os.sep, 1)[0] or os.sep]


def _unquoted_newlines_to_semicolons(command: str) -> str:
    """Newlines separate commands like ``;`` does; shlex would read them as plain whitespace."""
    out, quote = [], None
    for ch in command:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "\n":
            ch = ";"
        out.append(ch)
    return "".join(out)


def _tokens(command: str) -> list[str]:
    lexer = shlex.shlex(_unquoted_newlines_to_semicolons(command), posix=True, punctuation_chars=";&|<>()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:  # unbalanced quotes: fall back to whitespace words
        return command.split()


def _segments(tokens: list[str]) -> Iterable[tuple[list[str], list[str], str]]:
    """Yield ``(argv, redirect_targets, separator_that_ended_it)`` per simple command."""
    argv: list[str] = []
    targets: list[str] = []
    it = iter(tokens)
    for tok in it:
        if ">" in tok and set(tok) <= set("&>|0123456789"):
            if not tok.endswith("&"):
                target = next(it, "")
                if target and not target.startswith("&") and target != "/dev/null":
                    targets.append(target)
            continue
        if tok in ("<", "<<", "<<<", "<<-"):
            next(it, "")  # input file / heredoc delimiter / here-string: read, never written
            continue
        if tok and set(tok) <= set(";&|()"):
            yield argv, targets, tok
            argv, targets = [], []
            continue
        argv.append(tok)
    yield argv, targets, ""


_XARGS_OPTIONS_WITH_ARG = frozenset({"-n", "-I", "-L", "-P", "-d", "-E", "-s", "-a"})


def _strip_wrappers(argv: list[str]) -> list[str]:
    from tools.self_repo_guard import _WRAPPER_OPTIONS_WITH_ARG
    while argv:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]):
            argv = argv[1:]
            continue
        head = os.path.basename(argv[0])
        if head not in _WRAPPERS:
            break
        takes_value = _XARGS_OPTIONS_WITH_ARG if head == "xargs" else _WRAPPER_OPTIONS_WITH_ARG.get(head, frozenset())
        argv = argv[1:]
        while argv and argv[0].startswith("-"):
            argv = argv[2:] if argv[0] in takes_value else argv[1:]
    return argv


def _operands(args: list[str]) -> list[str]:
    return [a for a in args if a and not a.startswith("-")]


def _target_directory(args: list[str]) -> tuple[str | None, list[str]]:
    """``cp/mv/install/ln -t DIR`` / ``--target-directory[=]DIR``: (DIR, remaining args)."""
    rest = list(args)
    for i, arg in enumerate(rest):
        if arg.startswith("--target-directory="):
            return arg.split("=", 1)[1], rest[:i] + rest[i + 1:]
        if arg in ("-t", "--target-directory") and i + 1 < len(rest):
            return rest[i + 1], rest[:i] + rest[i + 2:]
        if arg.startswith("-t") and len(arg) > 2 and not arg.startswith("--"):
            return arg[2:], rest[:i] + rest[i + 1:]
    return None, rest


class _Scope:
    """What a command is checked against: its working dir, the locked skills and the lock record."""

    def __init__(self, cwd: str, locked: list[tuple[str, str]], record: str, depth: int,
                 env: dict[str, str] | None = None) -> None:
        self.cwd, self.locked, self.record, self.depth = cwd, locked, record, depth
        self.env = env if env is not None else _shell_env()

    def hit(self, word: str, ancestor: str = "") -> str | None:
        return next((r for path in _expand(word, self.cwd, self.env)
                     if (r := _hit(path, self.locked, self.record, ancestor=ancestor))), None)

    def first_hit(self, words: Iterable[str], ancestor: str = "") -> str | None:
        return next((r for w in words if (r := self.hit(w, ancestor))), None)


def _check_unlock(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb == "hermes" or (verb.startswith("python") and "hermes_cli.main" in args):
        if "curator" in args and "unlock" in args[args.index("curator"):]:
            return UNLOCK_REFUSAL
    return None


_RECURSIVE_COPY_FLAGS = re.compile(r"^(?:-[A-Za-z]*[rRa][A-Za-z]*|--recursive|--archive)$")


def _check_file_verbs(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb in _WRITES_EVERY_OPERAND or verb == "tee":
        return scope.first_hit(_operands(args), ancestor="any" if verb in _RECURSIVE else "")
    if verb not in _WRITES_LAST_OPERAND:
        return None
    target, rest = _target_directory(args)
    ops = _operands(rest)
    dest = target if target is not None else (ops[-1] if ops else None)
    if dest is None:
        return None
    # A recursive copy into a directory above the skill merges into it; rsync is recursive with -a/-r.
    merges = verb == "rsync" or any(_RECURSIVE_COPY_FLAGS.match(a) for a in rest)
    return scope.hit(dest, ancestor="skills" if merges else "")


def _check_inplace(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb in _INPLACE_EDITORS and any(
            a.startswith("--in-place") or (a.startswith("-") and not a.startswith("--") and "i" in a) for a in args):
        return scope.first_hit(_operands(args))
    return None


def _find_exec_commands(args: list[str]) -> Iterable[list[str]]:
    group: list[str] | None = None
    for arg in args:
        if arg in ("-exec", "-execdir", "-ok", "-okdir"):
            group = []
        elif group is not None and arg in (";", "+"):
            yield group
            group = None
        elif group is not None:
            group.append(arg)
    if group:
        yield group


def _check_find(verb: str, args: list[str], scope: _Scope) -> str | None:
    """``find -delete``, or ``-exec`` of a command that writes, over a tree holding a locked skill."""
    if verb != "find":
        return None
    starts = list(itertools.takewhile(lambda a: not a.startswith("-"), args)) or ["."]
    writes = "-delete" in args
    for cmd in _find_exec_commands(args):
        # Judge the executed command as if `{}` were a file in the locked skill.
        probe = [w if w != "{}" else os.path.join(scope.locked[0][1], "SKILL.md") for w in cmd] if scope.locked else cmd
        writes = writes or (bool(probe) and _argv_refusal(
            probe, scope.cwd, scope.locked, scope.record, scope.depth, scope.env) is not None)
    return scope.first_hit(starts, ancestor="any") if writes else None


def _check_git(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb != "git":
        return None
    where, rest = scope.cwd, list(args)
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag == "-C" and rest:
            where = _expand(rest.pop(0), scope.cwd, scope.env)[0]
        elif flag in ("-c", "--git-dir", "--work-tree") and rest:
            rest.pop(0)
    if rest and rest[0] in _GIT_MUTATIONS:
        return _hit(where, scope.locked, scope.record, ancestor="any")
    return None


def _check_extract(verb: str, args: list[str], scope: _Scope) -> str | None:
    if verb not in ("tar", "unzip", "bsdtar"):
        return None
    dest = scope.cwd
    for flag, value in itertools.pairwise(args):
        if flag in ("-C", "--directory", "-d"):
            dest = _expand(value, scope.cwd, scope.env)[0]
    bundled = bool(args) and not args[0].startswith("-") and "x" in args[0]  # `tar xzf a.tgz`
    extracting = verb == "unzip" or bundled or any(
        a == "--extract" or (a.startswith("-") and not a.startswith("--") and "x" in a) for a in args)
    # The archive's paths are unknown: extracting into a directory above the skill can overwrite it.
    return _hit(dest, scope.locked, scope.record, ancestor="skills") if extracting else None


def _check_interpreter(verb: str, args: list[str], scope: _Scope) -> str | None:
    if not _INTERPRETERS.match(verb):
        return None
    for flag, code in itertools.pairwise(args):
        if flag not in ("-c", "-e", "-E", "--eval", "-Command", "-r"):
            continue
        if verb in ("bash", "sh", "zsh", "dash", "ksh") and scope.depth < 3:
            reason = _command_refusal(code, scope.cwd, scope.locked, scope.record, scope.depth + 1)
        else:
            lang = "python" if _PYTHON_HEADER.match(verb) else "other"
            reason = _code_refusal(code, scope.locked, scope.record, scope.cwd, lang, scope.depth + 1)
        if reason:
            return reason
    return None


_ARGV_CHECKS = (_check_unlock, _check_file_verbs, _check_inplace, _check_find, _check_git,
                _check_extract, _check_interpreter)


def _argv_refusal(argv: list[str], cwd: str, locked, record, depth: int,
                  env: dict[str, str] | None = None) -> str | None:
    scope = _Scope(cwd, locked, record, depth, env)
    verb = os.path.basename(argv[0])
    return next((r for check in _ARGV_CHECKS if (r := check(verb, argv[1:], scope))), None)


_MAX_CODE_DEPTH = 3
_PYTHON_HEADER = re.compile(r"^python[0-9.]*$")
_SHELL_HEADER = re.compile(r"^(?:bash|sh|zsh|dash|ksh)$")
_HEREDOC_TAG = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_OS_PATH_WRITES = frozenset({
    "rename", "renames", "replace", "remove", "unlink", "rmdir", "removedirs", "mkdir", "makedirs",
    "symlink", "link"})
_SHUTIL_COPY_LAST = frozenset({"copy", "copy2", "copyfile", "copytree", "move"})
_SHUTIL_REMOVE = frozenset({"rmtree", "rmdir"})
_BARE_PATH_WRITES = frozenset({"rmtree", "remove", "unlink", "rename", "replace"})
_PATH_METHOD_WRITES = frozenset({"write_text", "write_bytes", "unlink", "rmdir", "touch", "mkdir",
                                 "rename", "replace", "symlink_to", "hardlink_to"})
_PATH_TYPES = frozenset({"Path", "PurePath", "PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"})
_DYNAMIC_WRITES = frozenset({"system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv",
                             "spawnve", "spawnvp", "spawnvpe", "execv", "execve", "execvp", "execvpe",
                             "execl", "execle", "execlp", "eval", "exec", "execfile"})


def _py_strings(node, names: dict[str, list[str]]) -> list[str] | None:
    """Literal strings an AST expression may denote (None: not statically known)."""
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.JoinedStr):
        out = [""]
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                vals: list[str] | None = [part.value]
            elif isinstance(part, ast.FormattedValue):
                vals = _py_strings(part.value, names)
            else:
                return None
            if vals is None:
                return None
            out = [head + tail for head in out for tail in vals]
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        left, right = _py_strings(node.left, names), _py_strings(node.right, names)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Div):
            return [os.path.join(a, b) for a in left for b in right]
        return [a + b for a in left for b in right]
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if name in _PATH_TYPES and node.args:
            return _py_strings(node.args[0], names)
        if name in ("join", "joinpath"):
            out = [""]
            for arg in node.args:
                vals = _py_strings(arg, names)
                if vals is None:
                    return None
                out = [os.path.join(head, tail) for head in out for tail in vals]
            return out
        if name == "cwd" and not node.args:
            return [os.getcwd()]
        if name == "expanduser" and node.args:
            vals = _py_strings(node.args[0], names)
            return [os.path.expanduser(v) for v in vals] if vals else None
    return None


def _py_path_like(node, names: dict[str, list[str]]) -> bool:
    """Whether *node* plausibly denotes a path (a Path(...) call, a literal-bound name, `x / y`)."""
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.BinOp):
        return isinstance(node.op, ast.Div)
    if isinstance(node, ast.Call):
        func = node.func
        return (func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")) in _PATH_TYPES
    return isinstance(node, ast.Attribute)


def _py_write_destinations(tree: ast.Module) -> tuple[list[str], bool]:
    """``(literal write destinations, saw a write whose destination is not statically known)``.

    Only the argument of a write call is a destination: the path named by a read is not.
    """
    names: dict[str, list[str]] = {}
    dests: list[str] = []
    unresolved = False
    nodes = sorted(ast.walk(tree), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            vals = _py_strings(node.value, names)
            if vals:
                names[node.targets[0].id] = vals
            else:
                names.pop(node.targets[0].id, None)
            continue
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        attr = func.attr if isinstance(func, ast.Attribute) else ""
        name = func.id if isinstance(func, ast.Name) else attr
        owner = func.value.id if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) else ""
        path_owner = isinstance(func, ast.Attribute) and _py_path_like(func.value, names)
        targets: list[ast.expr | None] = []
        mode_node: ast.expr | None = None
        if attr == "open" and path_owner:
            targets = [func.value]
            mode_node = node.args[0] if node.args else None
        elif attr == "open" or name == "open":
            if owner == "os":  # os.open(path, flags): the flags are not a mode string
                unresolved = True
                continue
            targets = [node.args[0] if node.args else None]
            mode_node = node.args[1] if len(node.args) > 1 else None
        elif attr in ("write_text", "write_bytes"):
            targets = [func.value]
        elif owner == "shutil" and attr in _SHUTIL_COPY_LAST:
            targets = [node.args[1] if len(node.args) > 1 else None]
        elif (owner == "shutil" and attr in _SHUTIL_REMOVE) or (owner == "os" and attr in _OS_PATH_WRITES):
            targets = list(node.args)
        elif attr in _PATH_METHOD_WRITES and path_owner:
            targets = [func.value]
        elif isinstance(func, ast.Name) and name in _BARE_PATH_WRITES:
            targets = list(node.args)
        elif name in _DYNAMIC_WRITES and (not isinstance(func, ast.Attribute) or owner in ("os", "subprocess")):
            unresolved = True
            continue
        else:
            continue
        if mode_node is None and name == "open":
            for kw in node.keywords:
                if kw.arg == "mode":
                    mode_node = kw.value
        if mode_node is None and name == "open":
            continue  # no mode: open() defaults to "r", a read
        if mode_node is not None:
            modes = _py_strings(mode_node, names)
            if not modes or not any(any(ch in m for ch in _WRITE_MODES) for m in modes):
                if not modes:
                    unresolved = True
                continue
        for target in targets:
            vals = _py_strings(target, names) if target is not None else None
            if vals is None:
                unresolved = True
            else:
                dests.extend(vals)
    return dests, unresolved


def _mention_refusal(text: str, locked) -> str | None:
    """Conservative fallback: any mention of a locked skill's root or `skills/<name>` segment."""
    skills_root = os.path.realpath(_skill_usage()._skills_dir())
    for name, root in locked:
        rel = os.path.relpath(root, skills_root) if skills_root and _inside(root, skills_root) else ""
        if root in text or (rel and f"skills/{rel}" in text) or f"skills/{os.path.basename(root)}" in text:
            return skill_refusal(name)
    return None


def _heredoc_body(code: str) -> tuple[str, str] | None:
    """``(header line, body)`` for the first ``<<TAG`` heredoc in *code*; None when there is none."""
    match = _HEREDOC_TAG.search(code)
    if match is None:
        return None
    head_end = code.find("\n", match.end())
    if head_end < 0:
        return None
    tag, body = match.group(2), []
    for line in code[head_end + 1:].splitlines():
        if line.strip() == tag:
            return code[:head_end], "\n".join(body)
        body.append(line)
    return code[:head_end], "\n".join(body)


def _header_verb(header: str) -> str:
    for argv, _targets, _separator in reversed(list(_segments(_tokens(header)))):
        argv = _strip_wrappers(argv)
        if argv:
            return os.path.basename(argv[0])
    return ""


def _leading_cwd(header: str, cwd: str) -> str:
    """The cwd the heredoc-fed interpreter runs in: replay the ``cd``s that precede it."""
    env = _shell_env()
    for argv, _targets, _separator in _segments(_tokens(header)):
        argv = _strip_wrappers(argv)
        if not argv:
            continue
        if argv[0] in ("cd", "pushd") and len(argv) > 1 and argv[1] != "-":
            cwd = _expand(argv[1], cwd, env)[0]
            continue
        break
    return cwd


def _code_refusal(code: str, locked, record, cwd: str = "", lang: str = "python",
                  depth: int = 0) -> str | None:
    """Inline interpreter code that WRITES into a locked skill (or the record), or runs with its
    working directory inside one. Destinations come from the write calls themselves, so naming a
    locked skill as a read source (or an unrelated checkout with the same ``skills/<name>`` segment)
    is not a hit. Code whose write destination cannot be resolved falls back to the mention check."""
    from hermes_constants import get_hermes_home
    if not _WRITEISH_CODE.search(code):
        return None
    if cwd and (reason := _hit(cwd, locked, record)):
        return reason
    home, hermes_home = os.path.expanduser("~"), str(get_hermes_home())
    text = code.replace("$HERMES_HOME", hermes_home).replace("~", home)
    if ".usage.json" in text:
        return RECORD_REFUSAL
    fallback = _mention_refusal(text, locked)
    if depth >= _MAX_CODE_DEPTH:
        return fallback  # too deeply nested to parse: fail closed on any mention
    heredoc = _heredoc_body(code)
    if heredoc is not None:
        header, body = heredoc
        body_cwd = _leading_cwd(header, cwd or os.getcwd())
        if reason := _hit(body_cwd, locked, record):
            return reason
        verb = _header_verb(header)
        if _SHELL_HEADER.match(verb):
            return _command_refusal(body, body_cwd, locked, record, depth + 1)
        if _PYTHON_HEADER.match(verb):
            return _code_refusal(body, locked, record, body_cwd, "python", depth + 1)
        return fallback if not body.strip() else _code_refusal(
            body, locked, record, body_cwd, "other", depth + 1)
    if lang != "python":
        return fallback  # JS/Ruby/PowerShell inline: not parsed, mention check only
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return fallback
    dests, unresolved = _py_write_destinations(tree)
    if unresolved:
        return fallback
    base = cwd or os.getcwd()
    for dest in dests:
        for path in _expand(dest, base):
            if reason := _hit(path, locked, record):
                return reason
    return None


_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)


def _command_refusal(command: str, cwd: str, locked, record, depth: int = 0) -> str | None:
    if "<<" in command and (reason := _code_refusal(command, locked, record, cwd, depth=depth)):
        return reason  # heredoc-fed interpreter body
    env = _shell_env()
    subshells: list[str] = []  # cwd to restore when each ( ... ) closes
    for argv, targets, separator in _segments(_tokens(command)):
        words = argv[1:] if argv[:1] == ["export"] else argv
        if words and all(_ASSIGNMENT.match(w) for w in words):  # `p=...` / `export p=...` persist
            for w in words:
                name, value = _ASSIGNMENT.match(w).groups()
                env[name] = _VAR.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), value)
            argv = []
        for target in targets:
            if reason := next((r for path in _expand(target, cwd, env)
                               if (r := _hit(path, locked, record))), None):
                return reason
        argv = _strip_wrappers(argv)
        if argv and argv[0] in ("cd", "pushd") and len(argv) > 1:
            cwd = _expand(argv[1], cwd, env)[0]
        elif argv and (reason := _argv_refusal(argv, cwd, locked, record, depth, env)):
            return reason
        for ch in separator:
            if ch == "(":
                subshells.append(cwd)
            elif ch == ")" and subshells:
                cwd = subshells.pop()
    return None


def command_refusal(command: str, cwd: str) -> str | None:
    """terminal guard: refusal text when *command* writes into a locked skill, the lock record, or
    runs ``hermes curator unlock``. None when nothing is locked."""
    locked = locked_skill_dirs()
    if not locked:
        return None
    return _command_refusal(command, cwd or os.getcwd(), locked, _record_path())
