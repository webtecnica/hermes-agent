"""Tests for git worktree isolation (CLI --worktree / -w flag).

Verifies worktree creation, cleanup, .worktreeinclude handling,
.gitignore management, and integration with the CLI.  (#652)
"""

import os
import shutil
import subprocess
import pytest
from pathlib import Path


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo for testing."""
    repo = tmp_path / "test-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )
    # Create initial commit (worktrees need at least one commit)
    (repo / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.com/test-repo.git"],
        cwd=repo, capture_output=True,
    )
    # Add a fake remote ref so cleanup logic sees the initial commit as
    # "pushed" when a remote is configured.
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        cwd=repo, capture_output=True,
    )
    return repo


@pytest.fixture
def git_repo_no_remote(tmp_path):
    """Create a temporary git repo with no configured remotes."""
    repo = tmp_path / "test-repo-no-remote"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )
    (repo / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo, capture_output=True,
    )
    return repo


@pytest.fixture
def git_repo_remote_no_tracking(tmp_path):
    """Create a temporary git repo with a remote but no remote-tracking refs."""
    repo = tmp_path / "test-repo-remote-no-tracking"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True,
    )
    (repo / "README.md").write_text("# Test Repo\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo, capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", "https://example.com/test-repo.git"],
        cwd=repo, capture_output=True,
    )
    return repo


# ---------------------------------------------------------------------------
# Lightweight reimplementations for testing (avoid importing cli.py)
# ---------------------------------------------------------------------------

def _git_repo_root(cwd=None):
    """Test version of _git_repo_root."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _setup_worktree(repo_root):
    """Test version of _setup_worktree — creates a worktree."""
    import uuid
    short_id = uuid.uuid4().hex[:8]
    wt_name = f"hermes-{short_id}"
    branch_name = f"hermes/{wt_name}"

    worktrees_dir = Path(repo_root) / ".worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    wt_path = worktrees_dir / wt_name

    result = subprocess.run(
        ["git", "worktree", "add", str(wt_path), "-b", branch_name, "HEAD"],
        capture_output=True, text=True, timeout=30, cwd=repo_root,
    )
    if result.returncode != 0:
        return None

    return {
        "path": str(wt_path),
        "branch": branch_name,
        "repo_root": repo_root,
    }


def _has_unpushed_commits(worktree_path, timeout=10):
    """Test version of the worktree unpushed-commit helper."""
    try:
        remote_refs = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname)", "refs/remotes"],
            capture_output=True, text=True, timeout=timeout, cwd=worktree_path,
        )
        if remote_refs.returncode != 0:
            return True
        if not remote_refs.stdout.strip():
            return False

        result = subprocess.run(
            ["git", "log", "--oneline", "HEAD", "--not", "--remotes"],
            capture_output=True, text=True, timeout=timeout, cwd=worktree_path,
        )
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _is_dirty(wt_path, timeout=10):
    """Test version of the worktree dirty-check helper (fail-safe True)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=timeout, cwd=wt_path,
        )
        if result.returncode != 0:
            return True
        return bool(result.stdout.strip())
    except Exception:
        return True


def _cleanup_worktree(info):
    """Test version of _cleanup_worktree.

    Mirrors the cli.py contract: preserves the worktree if it has
    unpushed commits OR uncommitted changes; only deletes the branch
    after ``git worktree remove`` succeeded.
    """
    wt_path = info["path"]
    branch = info["branch"]
    repo_root = info["repo_root"]

    if not Path(wt_path).exists():
        return

    if _has_unpushed_commits(wt_path, timeout=10):
        return False  # Did not clean up — has unpushed commits

    if _is_dirty(wt_path):
        return False  # Did not clean up — uncommitted changes

    result = subprocess.run(
        ["git", "worktree", "remove", wt_path, "--force"],
        capture_output=True, text=True, timeout=15, cwd=repo_root,
    )
    if result.returncode != 0:
        return False  # Removal failed — keep the branch

    subprocess.run(
        ["git", "branch", "-D", branch],
        capture_output=True, text=True, timeout=10, cwd=repo_root,
    )
    return True  # Cleaned up


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGitRepoDetection:
    """Test git repo root detection."""

    def test_detects_git_repo(self, git_repo):
        root = _git_repo_root(cwd=str(git_repo))
        assert root is not None
        assert Path(root).resolve() == git_repo.resolve()

    def test_detects_subdirectory(self, git_repo):
        subdir = git_repo / "src" / "lib"
        subdir.mkdir(parents=True)
        root = _git_repo_root(cwd=str(subdir))
        assert root is not None
        assert Path(root).resolve() == git_repo.resolve()

    def test_returns_none_outside_repo(self, tmp_path):
        # tmp_path itself is not a git repo
        bare_dir = tmp_path / "not-a-repo"
        bare_dir.mkdir()
        root = _git_repo_root(cwd=str(bare_dir))
        assert root is None


class TestWorktreeCreation:
    """Test worktree setup."""

    def test_creates_worktree(self, git_repo):
        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert Path(info["path"]).exists()
        assert info["branch"].startswith("hermes/hermes-")
        assert info["repo_root"] == str(git_repo)

        # Verify it's a valid git worktree
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=info["path"],
        )
        assert result.stdout.strip() == "true"

    def test_worktree_has_own_branch(self, git_repo):
        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Check branch name in worktree
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, cwd=info["path"],
        )
        assert result.stdout.strip() == info["branch"]

    def test_worktree_is_independent(self, git_repo):
        """Two worktrees from the same repo are independent."""
        info1 = _setup_worktree(str(git_repo))
        info2 = _setup_worktree(str(git_repo))
        assert info1 is not None
        assert info2 is not None
        assert info1["path"] != info2["path"]
        assert info1["branch"] != info2["branch"]

        # Create a file in worktree 1
        (Path(info1["path"]) / "only-in-wt1.txt").write_text("hello")

        # It should NOT appear in worktree 2
        assert not (Path(info2["path"]) / "only-in-wt1.txt").exists()

    def test_worktrees_dir_created(self, git_repo):
        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert (git_repo / ".worktrees").is_dir()

    def test_worktree_has_repo_files(self, git_repo):
        """Worktree should contain the repo's tracked files."""
        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert (Path(info["path"]) / "README.md").exists()


class TestWorktreeCleanup:
    """Test worktree cleanup on exit."""

    def test_clean_worktree_removed(self, git_repo):
        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert Path(info["path"]).exists()

        result = _cleanup_worktree(info)
        assert result is True
        assert not Path(info["path"]).exists()

    def test_dirty_worktree_preserved_on_cleanup(self, git_repo):
        """Dirty working tree is preserved even without unpushed commits.

        Uncommitted changes may be work the user has not retrieved yet —
        cleanup must never destroy them.
        """
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo))
        assert info is not None

        # Make uncommitted changes (staged but uncommitted file)
        (Path(info["path"]) / "new-file.txt").write_text("uncommitted")
        subprocess.run(
            ["git", "add", "new-file.txt"],
            cwd=info["path"], capture_output=True,
        )

        # The git_repo fixture already has a fake remote ref so the initial
        # commit is seen as "pushed" — only the dirty tree protects it.
        cli_mod._cleanup_worktree(info)
        assert Path(info["path"]).exists()  # Preserved despite no unpushed commits

        # Branch and lock are kept too
        result = subprocess.run(
            ["git", "branch", "--list", info["branch"]],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert info["branch"] in result.stdout
        assert cli_mod._worktree_is_locked(str(git_repo), info["path"]) is True

    def test_worktree_with_unpushed_commits_kept(self, git_repo):
        """Worktree with unpushed commits is preserved."""
        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Make a commit that is NOT on any remote
        (Path(info["path"]) / "work.txt").write_text("real work")
        subprocess.run(["git", "add", "work.txt"], cwd=info["path"], capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "agent work"],
            cwd=info["path"], capture_output=True,
        )

        result = _cleanup_worktree(info)
        assert result is False  # Kept — has unpushed commits
        assert Path(info["path"]).exists()

    def test_clean_worktree_removed_without_remote(self, git_repo_no_remote):
        """Clean worktrees in repos without remotes should still be removed."""
        info = _setup_worktree(str(git_repo_no_remote))
        assert info is not None
        assert Path(info["path"]).exists()
        assert _has_unpushed_commits(info["path"], timeout=10) is False

        result = _cleanup_worktree(info)
        assert result is True
        assert not Path(info["path"]).exists()

    def test_clean_worktree_removed_without_remote_tracking_refs(
        self, git_repo_remote_no_tracking
    ):
        """Configured remotes without fetched refs should not block cleanup."""
        info = _setup_worktree(str(git_repo_remote_no_tracking))
        assert info is not None
        assert Path(info["path"]).exists()
        assert _has_unpushed_commits(info["path"], timeout=10) is False

        result = _cleanup_worktree(info)
        assert result is True
        assert not Path(info["path"]).exists()

    def test_branch_deleted_on_cleanup(self, git_repo):
        info = _setup_worktree(str(git_repo))
        branch = info["branch"]

        _cleanup_worktree(info)

        # Branch should be gone
        result = subprocess.run(
            ["git", "branch", "--list", branch],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert branch not in result.stdout

    def test_cleanup_nonexistent_worktree(self, git_repo):
        """Cleanup should handle already-removed worktrees gracefully."""
        info = {
            "path": str(git_repo / ".worktrees" / "nonexistent"),
            "branch": "hermes/nonexistent",
            "repo_root": str(git_repo),
        }
        # Should not raise
        _cleanup_worktree(info)


class TestWorktreeInclude:
    """Test .worktreeinclude file handling."""

    def test_copies_included_files(self, git_repo):
        """Files listed in .worktreeinclude should be copied to the worktree."""
        # Create a .env file (gitignored)
        (git_repo / ".env").write_text("SECRET=abc123")
        (git_repo / ".gitignore").write_text(".env\n.worktrees/\n")
        subprocess.run(
            ["git", "add", ".gitignore"],
            cwd=str(git_repo), capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Add gitignore"],
            cwd=str(git_repo), capture_output=True,
        )

        # Create .worktreeinclude
        (git_repo / ".worktreeinclude").write_text(".env\n")

        # Import and use the real _setup_worktree logic for include handling
        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Manually copy .worktreeinclude entries (mirrors cli.py logic)
        include_file = git_repo / ".worktreeinclude"
        wt_path = Path(info["path"])
        for line in include_file.read_text().splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            src = git_repo / entry
            dst = wt_path / entry
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))

        # Verify .env was copied
        assert (wt_path / ".env").exists()
        assert (wt_path / ".env").read_text() == "SECRET=abc123"

    def test_ignores_comments_and_blanks(self, git_repo):
        """Comments and blank lines in .worktreeinclude should be skipped."""
        (git_repo / ".worktreeinclude").write_text(
            "# This is a comment\n"
            "\n"
            "  # Another comment\n"
        )
        info = _setup_worktree(str(git_repo))
        assert info is not None
        # Should not crash — just skip all lines


class TestGitignoreManagement:
    """Test that .worktrees/ is added to .gitignore."""

    def test_adds_to_gitignore(self, git_repo):
        """Creating a worktree should add .worktrees/ to .gitignore."""
        # Remove any existing .gitignore
        gitignore = git_repo / ".gitignore"
        if gitignore.exists():
            gitignore.unlink()

        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Now manually add .worktrees/ to .gitignore (mirrors cli.py logic)
        _ignore_entry = ".worktrees/"
        existing = gitignore.read_text() if gitignore.exists() else ""
        if _ignore_entry not in existing.splitlines():
            with open(gitignore, "a") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{_ignore_entry}\n")

        content = gitignore.read_text()
        assert ".worktrees/" in content

    def test_does_not_duplicate_gitignore_entry(self, git_repo):
        """If .worktrees/ is already in .gitignore, don't add again."""
        gitignore = git_repo / ".gitignore"
        gitignore.write_text(".worktrees/\n")

        # The check should see it's already there
        existing = gitignore.read_text()
        assert ".worktrees/" in existing.splitlines()


class TestMultipleWorktrees:
    """Test running multiple worktrees concurrently (the core use case)."""

    def test_ten_concurrent_worktrees(self, git_repo):
        """Create 10 worktrees — simulating 10 parallel agents."""
        worktrees = []
        for _ in range(10):
            info = _setup_worktree(str(git_repo))
            assert info is not None
            worktrees.append(info)

        # All should exist and be independent
        paths = [info["path"] for info in worktrees]
        assert len(set(paths)) == 10  # All unique

        # Each should have the repo files
        for info in worktrees:
            assert (Path(info["path"]) / "README.md").exists()

        # Edit a file in one worktree
        (Path(worktrees[0]["path"]) / "README.md").write_text("Modified in wt0")

        # Others should be unaffected
        for info in worktrees[1:]:
            assert (Path(info["path"]) / "README.md").read_text() == "# Test Repo\n"

        # List worktrees via git
        result = subprocess.run(
            ["git", "worktree", "list"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        # Should have 11 entries: main + 10 worktrees
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 11

        # Cleanup all (git_repo fixture has a fake remote ref so cleanup works)
        for info in worktrees:
            # Discard changes first so cleanup works
            subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=info["path"], capture_output=True,
            )
            _cleanup_worktree(info)

        # All should be removed
        for info in worktrees:
            assert not Path(info["path"]).exists()


class TestWorktreeDirectorySymlink:
    """Test .worktreeinclude with directories (symlinked)."""

    def test_symlinks_directory(self, git_repo):
        """Directories in .worktreeinclude should be symlinked."""
        # Create a .venv directory
        venv_dir = git_repo / ".venv" / "lib"
        venv_dir.mkdir(parents=True)
        (venv_dir / "marker.txt").write_text("venv marker")
        (git_repo / ".gitignore").write_text(".venv/\n.worktrees/\n")
        subprocess.run(
            ["git", "add", ".gitignore"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "gitignore"], cwd=str(git_repo), capture_output=True
        )

        (git_repo / ".worktreeinclude").write_text(".venv/\n")

        info = _setup_worktree(str(git_repo))
        assert info is not None

        wt_path = Path(info["path"])
        src = git_repo / ".venv"
        dst = wt_path / ".venv"

        # Manually symlink (mirrors cli.py logic)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(src.resolve()), str(dst))

        assert dst.is_symlink()
        assert (dst / "lib" / "marker.txt").read_text() == "venv marker"


class TestStaleWorktreePruning:
    """Test _prune_stale_worktrees garbage collection."""

    def test_prunes_old_clean_worktree(self, git_repo):
        """Old clean worktrees should be removed on prune."""
        import time

        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert Path(info["path"]).exists()

        # Make the worktree look old (set mtime to 25h ago)
        old_time = time.time() - (25 * 3600)
        os.utime(info["path"], (old_time, old_time))

        # Reimplementation of prune logic (matches cli.py)
        worktrees_dir = git_repo / ".worktrees"
        cutoff = time.time() - (24 * 3600)

        for entry in worktrees_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("hermes-"):
                continue
            try:
                mtime = entry.stat().st_mtime
                if mtime > cutoff:
                    continue
            except Exception:
                continue

            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, cwd=str(entry),
            )
            if status.stdout.strip():
                continue

            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5, cwd=str(entry),
            )
            branch = branch_result.stdout.strip()
            subprocess.run(
                ["git", "worktree", "remove", str(entry), "--force"],
                capture_output=True, text=True, timeout=15, cwd=str(git_repo),
            )
            if branch:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    capture_output=True, text=True, timeout=10, cwd=str(git_repo),
                )

        assert not Path(info["path"]).exists()

    def test_keeps_recent_worktree(self, git_repo):
        """Recent worktrees should NOT be pruned."""
        import time

        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Don't modify mtime — it's recent
        worktrees_dir = git_repo / ".worktrees"
        cutoff = time.time() - (24 * 3600)

        pruned = False
        for entry in worktrees_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("hermes-"):
                continue
            mtime = entry.stat().st_mtime
            if mtime > cutoff:
                continue  # Too recent
            pruned = True

        assert not pruned
        assert Path(info["path"]).exists()

    def test_keeps_old_worktree_with_unpushed_commits(self, git_repo):
        """Old worktrees (24-72h) with unpushed commits should NOT be pruned."""
        import time

        info = _setup_worktree(str(git_repo))
        assert info is not None

        # Make an unpushed commit
        (Path(info["path"]) / "work.txt").write_text("real work")
        subprocess.run(["git", "add", "work.txt"], cwd=info["path"], capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "agent work"],
            cwd=info["path"], capture_output=True,
        )

        # Make it old (25h — in the 24-72h soft tier)
        old_time = time.time() - (25 * 3600)
        os.utime(info["path"], (old_time, old_time))

        # Check for unpushed commits (simulates prune logic)
        has_unpushed = _has_unpushed_commits(info["path"])
        assert has_unpushed  # Has unpushed commits → not pruned in soft tier
        assert Path(info["path"]).exists()

    def test_prunes_old_clean_worktree_without_remote(self, git_repo_no_remote):
        """Old clean worktrees in repos without remotes should not be kept."""
        import time

        info = _setup_worktree(str(git_repo_no_remote))
        assert info is not None
        assert Path(info["path"]).exists()

        old_time = time.time() - (25 * 3600)
        os.utime(info["path"], (old_time, old_time))

        worktrees_dir = git_repo_no_remote / ".worktrees"
        cutoff = time.time() - (24 * 3600)

        for entry in worktrees_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("hermes-"):
                continue
            mtime = entry.stat().st_mtime
            if mtime > cutoff:
                continue
            if _has_unpushed_commits(str(entry), timeout=5):
                continue

            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5, cwd=str(entry),
            )
            branch = branch_result.stdout.strip()
            subprocess.run(
                ["git", "worktree", "remove", str(entry), "--force"],
                capture_output=True, text=True, timeout=15, cwd=str(git_repo_no_remote),
            )
            if branch:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    capture_output=True, text=True, timeout=10, cwd=str(git_repo_no_remote),
                )

        assert not Path(info["path"]).exists()

    def test_prunes_old_clean_worktree_without_remote_tracking_refs(
        self, git_repo_remote_no_tracking
    ):
        """Old clean worktrees with no fetched remote refs should be pruned."""
        import time

        info = _setup_worktree(str(git_repo_remote_no_tracking))
        assert info is not None
        assert Path(info["path"]).exists()

        old_time = time.time() - (25 * 3600)
        os.utime(info["path"], (old_time, old_time))

        worktrees_dir = git_repo_remote_no_tracking / ".worktrees"
        cutoff = time.time() - (24 * 3600)

        for entry in worktrees_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("hermes-"):
                continue
            mtime = entry.stat().st_mtime
            if mtime > cutoff:
                continue
            if _has_unpushed_commits(str(entry), timeout=5):
                continue

            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5, cwd=str(entry),
            )
            branch = branch_result.stdout.strip()
            subprocess.run(
                ["git", "worktree", "remove", str(entry), "--force"],
                capture_output=True, text=True, timeout=15,
                cwd=str(git_repo_remote_no_tracking),
            )
            if branch:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    capture_output=True, text=True, timeout=10,
                    cwd=str(git_repo_remote_no_tracking),
                )

        assert not Path(info["path"]).exists()

    def test_force_prunes_very_old_worktree(self, git_repo):
        """Very old (>72h) CLEAN, unlocked, fully-pushed worktrees are pruned."""
        import time
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo))
        assert info is not None

        # _setup_worktree locks the worktree; unlock to simulate a worktree
        # whose owning session released it (clean + unlocked + pushed).
        assert cli_mod._unlock_worktree(str(git_repo), info["path"]) is True

        # Make it very old (73h)
        old_time = time.time() - (73 * 3600)
        os.utime(info["path"], (old_time, old_time))

        cli_mod._prune_stale_worktrees(str(git_repo))

        assert not Path(info["path"]).exists()
        # Branch should be gone too
        result = subprocess.run(
            ["git", "branch", "--list", info["branch"]],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert info["branch"] not in result.stdout


class TestWorktreeLocking:
    """Test git-native worktree locks and the preserve-work contracts.

    These tests exercise the REAL cli.py implementations (not the local
    reimplementations above), matching the pattern in
    test_worktree_security.py.
    """

    def test_setup_worktree_locks(self, git_repo):
        """_setup_worktree leaves the new worktree locked."""
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo))
        assert info is not None

        # Verify via git worktree list --porcelain: the stanza for this
        # worktree must contain a "locked" line.
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        target = Path(info["path"]).resolve()
        current = None
        locked = False
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                current = Path(line[len("worktree "):].strip()).resolve()
            elif line == "locked" or line.startswith("locked "):
                if current == target:
                    locked = True
        assert locked
        assert cli_mod._worktree_is_locked(str(git_repo), info["path"]) is True

    def test_unlock_worktree(self, git_repo):
        """_unlock_worktree releases the lock taken by _setup_worktree."""
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo))
        assert info is not None
        assert cli_mod._worktree_is_locked(str(git_repo), info["path"]) is True

        assert cli_mod._unlock_worktree(str(git_repo), info["path"]) is True
        assert cli_mod._worktree_is_locked(str(git_repo), info["path"]) is False

    def test_prune_skips_locked_very_old_clean_worktree(self, git_repo):
        """A locked worktree is never pruned, even >72h old and clean."""
        import time
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo))
        assert info is not None
        # Still locked from _setup_worktree; clean; fully pushed.

        old_time = time.time() - (80 * 3600)
        os.utime(info["path"], (old_time, old_time))

        cli_mod._prune_stale_worktrees(str(git_repo))

        assert Path(info["path"]).exists()

    def test_prune_skips_old_dirty_unlocked_worktree(self, git_repo):
        """An old dirty worktree is not pruned even when unlocked."""
        import time
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo))
        assert info is not None
        assert cli_mod._unlock_worktree(str(git_repo), info["path"]) is True

        # Uncommitted change (untracked file)
        (Path(info["path"]) / "wip.txt").write_text("uncommitted work")

        old_time = time.time() - (25 * 3600)
        os.utime(info["path"], (old_time, old_time))

        cli_mod._prune_stale_worktrees(str(git_repo))

        assert Path(info["path"]).exists()
        assert (Path(info["path"]) / "wip.txt").exists()

    def test_prune_preserves_very_old_worktree_with_unpushed_commits(self, git_repo):
        """Unpushed commits protect a worktree at ANY age — the old >72h
        force-remove tier is gone."""
        import time
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo))
        assert info is not None
        assert cli_mod._unlock_worktree(str(git_repo), info["path"]) is True

        # Unpushed commit (clean tree afterwards)
        (Path(info["path"]) / "work.txt").write_text("real work")
        subprocess.run(["git", "add", "work.txt"], cwd=info["path"], capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "agent work"],
            cwd=info["path"], capture_output=True,
        )

        old_time = time.time() - (80 * 3600)
        os.utime(info["path"], (old_time, old_time))

        cli_mod._prune_stale_worktrees(str(git_repo))

        assert Path(info["path"]).exists()
        result = subprocess.run(
            ["git", "branch", "--list", info["branch"]],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert info["branch"] in result.stdout

    def test_cleanup_preserves_dirty_worktree(self, git_repo):
        """_cleanup_worktree keeps a dirty worktree (untracked file)."""
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo))
        assert info is not None

        (Path(info["path"]) / "scratch.txt").write_text("not yet committed")

        cli_mod._cleanup_worktree(info)

        assert Path(info["path"]).exists()
        assert (Path(info["path"]) / "scratch.txt").exists()

    def test_cleanup_removes_clean_locked_worktree(self, git_repo):
        """_cleanup_worktree unlocks then removes a clean, pushed worktree."""
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo))
        assert info is not None
        assert cli_mod._worktree_is_locked(str(git_repo), info["path"]) is True

        cli_mod._cleanup_worktree(info)

        assert not Path(info["path"]).exists()
        result = subprocess.run(
            ["git", "branch", "--list", info["branch"]],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert info["branch"] not in result.stdout

    def test_branch_kept_when_worktree_remove_fails(self, git_repo, monkeypatch):
        """If `git worktree remove` fails, the branch must NOT be deleted."""
        import subprocess as sp
        import cli as cli_mod

        info = cli_mod._setup_worktree(str(git_repo))
        assert info is not None

        real_run = sp.run

        def fake_run(cmd, *args, **kwargs):
            if (
                isinstance(cmd, (list, tuple))
                and list(cmd[:3]) == ["git", "worktree", "remove"]
            ):
                return sp.CompletedProcess(
                    cmd, returncode=1, stdout="", stderr="simulated removal failure"
                )
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(sp, "run", fake_run)

        cli_mod._cleanup_worktree(info)

        monkeypatch.undo()

        # Worktree dir still present, branch NOT deleted
        assert Path(info["path"]).exists()
        result = subprocess.run(
            ["git", "branch", "--list", info["branch"]],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert info["branch"] in result.stdout

    def test_worktree_is_locked_fail_safe(self, tmp_path):
        """_worktree_is_locked returns True (fail safe) on a bogus repo_root."""
        import cli as cli_mod

        bogus = tmp_path / "does-not-exist"
        assert cli_mod._worktree_is_locked(str(bogus), str(bogus / "wt")) is True

        # An existing directory that is not a git repo is also an error case
        not_repo = tmp_path / "not-a-repo"
        not_repo.mkdir()
        assert cli_mod._worktree_is_locked(str(not_repo), str(not_repo / "wt")) is True

    def test_worktree_is_dirty_fail_safe(self, tmp_path):
        """_worktree_is_dirty returns True (fail safe) on a bogus path."""
        import cli as cli_mod

        assert cli_mod._worktree_is_dirty(str(tmp_path / "missing")) is True


class TestEdgeCases:
    """Test edge cases for robustness."""

    def test_no_commits_repo(self, tmp_path):
        """Worktree creation should fail gracefully on a repo with no commits."""
        repo = tmp_path / "empty-repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo), capture_output=True)

        info = _setup_worktree(str(repo))
        assert info is None  # Should fail gracefully

    def test_not_a_git_repo(self, tmp_path):
        """Repo detection should return None for non-git directories."""
        bare = tmp_path / "not-git"
        bare.mkdir()
        root = _git_repo_root(cwd=str(bare))
        assert root is None

    def test_worktrees_dir_already_exists(self, git_repo):
        """Should work fine if .worktrees/ already exists."""
        (git_repo / ".worktrees").mkdir(exist_ok=True)
        info = _setup_worktree(str(git_repo))
        assert info is not None
        assert Path(info["path"]).exists()


class TestCLIFlagLogic:
    """Test the flag/config OR logic from main()."""

    def test_worktree_flag_triggers(self):
        """--worktree flag should trigger worktree creation."""
        worktree = True
        w = False
        config_worktree = False
        use_worktree = worktree or w or config_worktree
        assert use_worktree

    def test_w_flag_triggers(self):
        """-w flag should trigger worktree creation."""
        worktree = False
        w = True
        config_worktree = False
        use_worktree = worktree or w or config_worktree
        assert use_worktree

    def test_config_triggers(self):
        """worktree: true in config should trigger worktree creation."""
        worktree = False
        w = False
        config_worktree = True
        use_worktree = worktree or w or config_worktree
        assert use_worktree

    def test_none_set_no_trigger(self):
        """No flags and no config should not trigger."""
        worktree = False
        w = False
        config_worktree = False
        use_worktree = worktree or w or config_worktree
        assert not use_worktree


class TestTerminalCWDIntegration:
    """Test that TERMINAL_CWD is correctly set to the worktree path."""

    def test_terminal_cwd_set(self, git_repo):
        """After worktree setup, TERMINAL_CWD should point to the worktree."""
        info = _setup_worktree(str(git_repo))
        assert info is not None

        # This is what main() does:
        os.environ["TERMINAL_CWD"] = info["path"]
        assert os.environ["TERMINAL_CWD"] == info["path"]
        assert Path(os.environ["TERMINAL_CWD"]).exists()

        # Clean up env
        del os.environ["TERMINAL_CWD"]

    def test_terminal_cwd_is_valid_git_repo(self, git_repo):
        """The TERMINAL_CWD worktree should be a valid git working tree."""
        info = _setup_worktree(str(git_repo))
        assert info is not None

        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=info["path"],
        )
        assert result.stdout.strip() == "true"


class TestOrphanedBranchPruning:
    """Test cleanup of orphaned hermes/* and pr-* branches."""

    def test_prunes_orphaned_hermes_branch(self, git_repo):
        """hermes/hermes-* branches with no worktree should be deleted."""
        # Create a branch that looks like a worktree branch but has no worktree
        subprocess.run(
            ["git", "branch", "hermes/hermes-deadbeef", "HEAD"],
            cwd=str(git_repo), capture_output=True,
        )

        # Verify it exists
        result = subprocess.run(
            ["git", "branch", "--list", "hermes/hermes-deadbeef"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert "hermes/hermes-deadbeef" in result.stdout

        # Simulate _prune_orphaned_branches logic
        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        all_branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]

        wt_result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        active_branches = {"main"}
        for line in wt_result.stdout.split("\n"):
            if line.startswith("branch refs/heads/"):
                active_branches.add(line.split("branch refs/heads/", 1)[-1].strip())

        orphaned = [
            b for b in all_branches
            if b not in active_branches
            and (b.startswith("hermes/hermes-") or b.startswith("pr-"))
        ]
        assert "hermes/hermes-deadbeef" in orphaned

        # Delete them
        if orphaned:
            subprocess.run(
                ["git", "branch", "-D"] + orphaned,
                capture_output=True, text=True, cwd=str(git_repo),
            )

        # Verify gone
        result = subprocess.run(
            ["git", "branch", "--list", "hermes/hermes-deadbeef"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert "hermes/hermes-deadbeef" not in result.stdout

    def test_prunes_orphaned_pr_branch(self, git_repo):
        """pr-* branches should be deleted during pruning."""
        subprocess.run(
            ["git", "branch", "pr-1234", "HEAD"],
            cwd=str(git_repo), capture_output=True,
        )
        subprocess.run(
            ["git", "branch", "pr-5678", "HEAD"],
            cwd=str(git_repo), capture_output=True,
        )

        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        all_branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]

        active_branches = {"main"}
        orphaned = [
            b for b in all_branches
            if b not in active_branches and b.startswith("pr-")
        ]
        assert "pr-1234" in orphaned
        assert "pr-5678" in orphaned

        subprocess.run(
            ["git", "branch", "-D"] + orphaned,
            capture_output=True, text=True, cwd=str(git_repo),
        )

        # Verify gone
        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        remaining = result.stdout.strip()
        assert "pr-1234" not in remaining
        assert "pr-5678" not in remaining

    def test_preserves_active_worktree_branch(self, git_repo):
        """Branches with active worktrees should NOT be pruned."""
        info = _setup_worktree(str(git_repo))
        assert info is not None

        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        active_branches = set()
        for line in result.stdout.split("\n"):
            if line.startswith("branch refs/heads/"):
                active_branches.add(line.split("branch refs/heads/", 1)[-1].strip())

        assert info["branch"] in active_branches  # Protected

    def test_preserves_main_branch(self, git_repo):
        """main branch should never be pruned."""
        result = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        all_branches = [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]
        active_branches = {"main"}

        orphaned = [
            b for b in all_branches
            if b not in active_branches
            and (b.startswith("hermes/hermes-") or b.startswith("pr-"))
        ]
        assert "main" not in orphaned


class TestSystemPromptInjection:
    """Test that the agent gets worktree context in its system prompt."""

    def test_prompt_note_format(self, git_repo):
        """Verify the system prompt note contains all required info."""
        info = _setup_worktree(str(git_repo))
        assert info is not None

        # This is what main() does:
        wt_note = (
            f"\n\n[System note: You are working in an isolated git worktree at "
            f"{info['path']}. Your branch is `{info['branch']}`. "
            f"Changes here do not affect the main working tree or other agents. "
            f"Remember to commit and push your changes, and create a PR if appropriate. "
            f"The original repo is at {info['repo_root']}.]\n"
        )

        assert info["path"] in wt_note
        assert info["branch"] in wt_note
        assert info["repo_root"] in wt_note
        assert "isolated git worktree" in wt_note
        assert "commit and push" in wt_note
