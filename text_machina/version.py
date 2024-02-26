_MAJOR = "0"
_MINOR = "2"
_REVISION = "5"

VERSION_SHORT = "{0}.{1}".format(_MAJOR, _MINOR)
VERSION = "{0}.{1}.{2}".format(_MAJOR, _MINOR, _REVISION)


def _is_newer_than(version: str) -> bool:
    """True if current version is newer than 'version'."""
    from pkg_resources import parse_version as parse

    return parse(VERSION) > parse(version)


def _main():
    """
    For use inside an Azure pipeline.

    Usage:

    $ pip index versions text-machina | python -m text_machina.version

    Outputs:
     - 1 if code version is newer than latest version on pip.
     - 0 otherwise.
    """
    import sys

    for line in sys.stdin.readlines():
        line = line.strip()
        if line.startswith("LATEST"):
            latest_version = line.split()[-1]
            print(int(_is_newer_than(latest_version)))
            return
    else:
        raise ValueError("No LATEST line in stdin.")


if __name__ == "__main__":
    _main()
