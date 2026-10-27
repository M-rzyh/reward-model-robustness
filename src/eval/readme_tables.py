"""Refresh the generated tables in README.md: each <!-- BEGIN path --> ... <!-- END path --> block is replaced
by the current contents of `path`, so no results number in the README is typed by hand."""
import re
import sys
from pathlib import Path

BLOCK = re.compile(r"(<!-- BEGIN (\S+) -->\n).*?(<!-- END \2 -->)", re.S)


def refresh(text: str) -> tuple[str, list[str]]:
    used = []

    def fill(m):
        used.append(m.group(2))
        return m.group(1) + Path(m.group(2)).read_text().rstrip("\n") + "\n" + m.group(3)

    return BLOCK.sub(fill, text), used


def main():
    readme = Path(sys.argv[1] if len(sys.argv) > 1 else "README.md")
    new, used = refresh(readme.read_text())
    readme.write_text(new)
    print(f"refreshed {len(used)} tables in {readme}: {', '.join(used)}")


if __name__ == "__main__":
    main()
