"""Check repository-relative Markdown links and balanced Mermaid fences."""

from __future__ import annotations

import re
from pathlib import Path

LINK_PATTERN = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)]+)\)")


def main() -> None:
    """Fail when local Markdown targets or Mermaid fences are invalid."""

    root = Path.cwd()
    failures: list[str] = []
    markdown_files = list(root.glob("*.md")) + list((root / "docs").glob("*.md"))
    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        mermaid_blocks = re.findall(r"```mermaid\s+.*?```", content, flags=re.DOTALL)
        if len(mermaid_blocks) != content.count("```mermaid") or content.count("```") % 2:
            failures.append(f"{path}: unbalanced Mermaid fence")
        for match in LINK_PATTERN.finditer(content):
            target_text = match.group(1).split("#", maxsplit=1)[0]
            if not target_text:
                continue
            target = (path.parent / target_text).resolve()
            if not target.exists():
                failures.append(f"{path}: missing local link target {target_text}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Documentation check passed for {len(markdown_files)} Markdown files")


if __name__ == "__main__":
    main()
