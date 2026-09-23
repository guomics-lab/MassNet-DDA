"""Bounded, link-free extraction for source dependency archives (Python 3.10+).

This prevents unsafe archive writes; it does not authenticate downloaded source.
Only use a trusted download origin until independently verified digests are pinned.
"""

import ntpath
import os
from pathlib import Path
import shutil
import tarfile
import tempfile


_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
# Windows also recognizes these three superscript digits in device names.
# Keep this check independent of ntpath.isreserved, added after Python 3.10.
_RESERVED_NAMES.update("COM" + digit for digit in "123456789\u00b9\u00b2\u00b3")
_RESERVED_NAMES.update("LPT" + digit for digit in "123456789\u00b9\u00b2\u00b3")
_WINDOWS_FORBIDDEN = frozenset(':<>"|?*')


def _member_parts(name):
    if not name or "\x00" in name or "\\" in name:
        raise ValueError("Unsafe archive member name")
    if name.startswith("/") or ntpath.splitdrive(name)[0]:
        raise ValueError("Absolute archive paths are not allowed")
    parts = tuple(part for part in name.split("/") if part not in ("", "."))
    if not parts or ".." in parts or any(":" in part for part in parts):
        raise ValueError("Unsafe archive member path")
    if any(part.rstrip(" .") != part
           or part.partition(".")[0].rstrip(" ").upper() in _RESERVED_NAMES
           or any(ord(character) < 32 or character in _WINDOWS_FORBIDDEN for character in part)
           for part in parts):
        raise ValueError("Archive contains a platform-reserved path")
    return parts


def extract_source_archive(archive_path, destination, expected_root, *,
                           max_members=100000, max_total_size=4 * 1024 ** 3,
                           max_file_size=1024 ** 3):
    """Validate the entire archive, then publish its single expected source tree.

    Reject links, devices, sparse files, duplicate/conflicting paths and excessive
    declared sizes. Extraction uses exclusive writes inside a private temporary
    directory, never ``extractall`` or archive-supplied ownership/permissions.
    Existing source trees are not overwritten, including symbolic links.
    """
    if _member_parts(expected_root) != (expected_root,):
        raise ValueError("Expected root must be one directory name")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / expected_root
    if os.path.lexists(target):
        raise FileExistsError("Refusing to overwrite an existing source tree")

    with tarfile.open(archive_path, "r:gz") as archive:
        members = []
        kinds = {}
        total_size = 0
        for count, member in enumerate(archive, 1):
            if count > max_members:
                raise ValueError("Archive has too many members")
            parts = _member_parts(member.name)
            if parts[0] != expected_root:
                raise ValueError("Archive member is outside its expected source tree")
            if not (member.isdir() or member.isreg()) or member.issparse():
                raise ValueError("Only regular files and directories are allowed")
            if len(parts) == 1 and not member.isdir():
                raise ValueError("The archive root must be a directory")
            if member.size < 0 or member.size > max_file_size:
                raise ValueError("Archive member size exceeds the limit")
            total_size += member.size
            if total_size > max_total_size:
                raise ValueError("Archive total size exceeds the limit")
            kind = "directory" if member.isdir() else "file"
            if parts in kinds:
                if kinds[parts] == kind == "directory":
                    continue
                raise ValueError("Archive contains duplicate or conflicting paths")
            kinds[parts] = kind
            members.append((member, parts))

        if not members:
            raise ValueError("The source archive is empty")
        for parts in kinds:
            for end in range(1, len(parts)):
                if kinds.get(parts[:end]) == "file":
                    raise ValueError("Archive file conflicts with a parent directory")

        with tempfile.TemporaryDirectory(prefix=".source-extract-", dir=destination) as staging:
            staging = Path(staging)
            for member, parts in members:
                output = staging.joinpath(*parts)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, output.open("xb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
                output.chmod(0o755 if member.mode & 0o111 else 0o644)
            # Fail if another build created a target while this archive was read.
            if os.path.lexists(target):
                raise FileExistsError("Source tree appeared during archive extraction")
            (staging / expected_root).rename(target)
    return target
