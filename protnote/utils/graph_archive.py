"""Indexed archive format for consolidated graph .pt files.

Binary format:
    [8 bytes IDENTIFIER "TNGRPH01"]
    [8 bytes header_len (uint64 LE)]
    [JSON header (utf-8)]
    [concatenated torch.save() byte buffers]

The JSON header contains:
    {"version": 1, "entries": {"seq_id": {"offset": int, "size": int}, ...}}

Offsets are relative to the start of the data blob (after the header).
"""

import io
import json
import logging
import struct
from pathlib import Path

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

IDENTIFIER = b"TNGRPH01"
HEADER_LEN_FMT = "<Q"  # uint64 little-endian
HEADER_LEN_SIZE = struct.calcsize(HEADER_LEN_FMT)


def consolidate_to_archive(
    graph_dir: str | Path,
    graph_index: dict[str, str],
    archive_path: str | Path,
) -> int:
    """Read individual .pt files and write a single indexed archive.

    Args:
        graph_dir: Directory containing individual .pt files.
        graph_index: Mapping of seq_id -> filename (relative to graph_dir).
        archive_path: Output archive file path.

    Returns:
        Number of entries written.
    """
    graph_dir = Path(graph_dir)
    archive_path = Path(archive_path)

    entries = {}
    buffers = {}

    for seq_id, filename in tqdm(graph_index.items(), desc="Reading .pt files"):
        pt_path = graph_dir / filename
        if not pt_path.exists():
            logger.warning(f"Missing .pt file for {seq_id}: {pt_path}")
            continue
        buf = io.BytesIO()
        data = torch.load(pt_path, weights_only=False)
        torch.save(data, buf)
        buffers[seq_id] = buf.getvalue()

    # Build header with offsets
    offset = 0
    for seq_id, raw in buffers.items():
        entries[seq_id] = {"offset": offset, "size": len(raw)}
        offset += len(raw)

    header = json.dumps({"version": 1, "entries": entries}).encode("utf-8")
    header_len = len(header)

    with open(archive_path, "wb") as f:
        f.write(IDENTIFIER)
        f.write(struct.pack(HEADER_LEN_FMT, header_len))
        f.write(header)
        for seq_id in entries:
            f.write(buffers[seq_id])

    logger.info(f"Archive written: {archive_path} ({len(entries)} entries, {archive_path.stat().st_size / 1e6:.1f} MB)")
    return len(entries)


class GraphArchiveReader:
    """Random-access reader for graph archive files.

    Lazy initialization: the file is not opened until the first access,
    making this safe to pass through DataLoader's multiprocessing spawn.
    """

    def __init__(self, archive_path: str | Path):
        self._archive_path = str(archive_path)
        self._file = None
        self._entries = None
        self._data_offset = None

    def _ensure_open(self):
        if self._file is not None:
            return
        f = open(self._archive_path, "rb")
        identifier = f.read(len(IDENTIFIER))
        if identifier != IDENTIFIER:
            f.close()
            raise ValueError(f"Invalid archive identifier: {identifier!r} (expected {IDENTIFIER!r})")
        header_len = struct.unpack(HEADER_LEN_FMT, f.read(HEADER_LEN_SIZE))[0]
        header_raw = f.read(header_len)
        header = json.loads(header_raw.decode("utf-8"))
        if header.get("version") != 1:
            f.close()
            raise ValueError(f"Unsupported archive version: {header.get('version')}")
        self._entries = header["entries"]
        self._data_offset = len(IDENTIFIER) + HEADER_LEN_SIZE + header_len
        self._file = f

    def __contains__(self, seq_id: str) -> bool:
        self._ensure_open()
        return seq_id in self._entries

    def __getitem__(self, seq_id: str) -> dict:
        self._ensure_open()
        entry = self._entries.get(seq_id)
        if entry is None:
            raise KeyError(seq_id)
        self._file.seek(self._data_offset + entry["offset"])
        raw = self._file.read(entry["size"])
        return torch.load(io.BytesIO(raw), weights_only=False)

    def keys(self):
        self._ensure_open()
        return self._entries.keys()

    def __len__(self) -> int:
        self._ensure_open()
        return len(self._entries)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()
