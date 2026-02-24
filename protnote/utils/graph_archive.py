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

import hashlib
import io
import json
import logging
import os
import struct
from pathlib import Path

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

IDENTIFIER = b"TNGRPH01"
HEADER_LEN_FMT = "<Q"  # uint64 little-endian
HEADER_LEN_SIZE = struct.calcsize(HEADER_LEN_FMT)


def shard_index_for_seq_id(seq_id, num_shards):
    """Deterministic shard assignment based on MD5 hash of sequence ID."""
    return int(hashlib.md5(seq_id.encode()).hexdigest(), 16) % num_shards


def consolidate_to_archive(
    graph_dir,
    graph_index,
    archive_path,
):
    """Read individual .pt files and write a single indexed archive.

    Uses a two-pass streaming approach to avoid loading all files into memory:
    Pass 1 — write a placeholder header then stream each .pt file to disk.
    Pass 2 — seek back and overwrite the placeholder with the real header.

    Args:
        graph_dir: Directory containing individual .pt files.
        graph_index: Mapping of seq_id -> {"filename": str, ...}.
        archive_path: Output archive file path.

    Returns:
        Number of entries written.
    """
    graph_dir = Path(graph_dir)
    archive_path = Path(archive_path)

    num_entries = len(graph_index)
    # Generous header estimate: ~120 bytes per JSON entry + overhead
    max_header_size = max(1_000_000, num_entries * 120 + 1000)
    preamble_size = len(IDENTIFIER) + HEADER_LEN_SIZE  # 16 bytes

    entries = {}

    with open(archive_path, "wb") as f:
        # Pass 1: write placeholder header, then stream data
        f.write(IDENTIFIER)
        f.write(struct.pack(HEADER_LEN_FMT, max_header_size))
        f.write(b"\x00" * max_header_size)

        offset = 0
        for seq_id, entry in tqdm(graph_index.items(), desc="Archiving .pt files"):
            pt_path = graph_dir / entry["filename"]
            if not pt_path.exists():
                logger.warning(f"Missing .pt file for {seq_id}: {pt_path}")
                continue
            data = torch.load(pt_path, weights_only=False)
            buf = io.BytesIO()
            torch.save(data, buf)
            raw = buf.getvalue()
            f.write(raw)
            entries[seq_id] = {"offset": offset, "size": len(raw)}
            offset += len(raw)

        # Pass 2: write real header over the placeholder
        header_json = json.dumps({"version": 1, "entries": entries}).encode("utf-8")
        if len(header_json) > max_header_size:
            raise RuntimeError(
                f"Header size ({len(header_json)}) exceeds reserved space "
                f"({max_header_size}). This is a bug — please report it."
            )
        # Pad with spaces (valid trailing whitespace for JSON)
        padded_header = header_json + b" " * (max_header_size - len(header_json))
        f.seek(preamble_size)
        f.write(padded_header)

    logger.info(
        f"Archive written: {archive_path} ({len(entries)} entries, "
        f"{archive_path.stat().st_size / 1e6:.1f} MB)"
    )
    return len(entries)


def consolidate_to_sharded_archive(graph_dir, graph_index, output_dir, num_shards=16):
    """Partition entries by hash and write one archive per shard.

    Args:
        graph_dir: Directory containing individual .pt files.
        graph_index: Mapping of seq_id -> {"filename": str, ...}.
        output_dir: Directory where shard files will be written.
        num_shards: Number of shards to create.

    Returns:
        Total number of entries written across all shards.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Partition entries by shard
    shard_indices = [dict() for _ in range(num_shards)]
    for seq_id, entry in graph_index.items():
        shard = shard_index_for_seq_id(seq_id, num_shards)
        shard_indices[shard][seq_id] = entry

    total = 0
    for i in range(num_shards):
        shard_path = output_dir / f"graphs.shard-{i:04d}-of-{num_shards:04d}.pngrph"
        if shard_indices[i]:
            n = consolidate_to_archive(graph_dir, shard_indices[i], shard_path)
            total += n
            logger.info(f"Shard {i + 1}/{num_shards}: {n} entries -> {shard_path.name}")
        else:
            logger.info(f"Shard {i + 1}/{num_shards}: empty, skipping")

    logger.info(
        f"Sharded archive complete: {total} entries across {num_shards} shards "
        f"in {output_dir}"
    )
    return total


class GraphArchiveReader:
    """Random-access reader for graph archive files.

    Lazy initialization: the file is not opened until the first access,
    making this safe to pass through DataLoader's multiprocessing spawn.
    """

    def __init__(self, archive_path):
        self._archive_path = str(archive_path)
        self._file = None
        self._entries = None
        self._data_offset = None
        self._pid = None

    def _ensure_open(self):
        # Re-open if we were forked into a new process (shared fd is not safe)
        if self._file is not None and self._pid == os.getpid():
            return
        if self._file is not None:
            # Forked — close inherited fd and re-open
            self._file.close()
            self._file = None
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
        self._pid = os.getpid()

    def __contains__(self, seq_id):
        self._ensure_open()
        return seq_id in self._entries

    def __getitem__(self, seq_id):
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

    def __len__(self):
        self._ensure_open()
        return len(self._entries)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


_MISSING_SHARD = object()


class ShardedGraphArchiveReader:
    """Random-access reader that routes lookups across multiple shard archives.

    Lazy initialization per shard: no file handles are held at construction,
    making this safe to pass through DataLoader's multiprocessing spawn.
    Empty shards (no file on disk) are tolerated and cached as missing.
    """

    def __init__(self, shard_dir, num_shards):
        self._shard_dir = str(shard_dir)
        self._num_shards = num_shards
        self._readers = [None] * num_shards

    @classmethod
    def from_directory(cls, shard_dir):
        """Auto-detect shard count from filenames in directory."""
        shard_dir = Path(shard_dir)
        shard_files = sorted(shard_dir.glob("graphs.shard-*-of-*.pngrph"))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {shard_dir}")
        # Format: graphs.shard-NNNN-of-TTTT.pngrph
        name = shard_files[0].stem  # e.g. "graphs.shard-0000-of-0016"
        num_shards = int(name.split("-of-")[-1])
        return cls(shard_dir, num_shards)

    def _get_reader(self, shard_idx):
        reader = self._readers[shard_idx]
        if reader is _MISSING_SHARD:
            return None
        if reader is None:
            shard_path = (
                Path(self._shard_dir)
                / f"graphs.shard-{shard_idx:04d}-of-{self._num_shards:04d}.pngrph"
            )
            if shard_path.exists():
                self._readers[shard_idx] = GraphArchiveReader(shard_path)
            else:
                self._readers[shard_idx] = _MISSING_SHARD
                return None
        return self._readers[shard_idx]

    def __contains__(self, seq_id):
        shard_idx = shard_index_for_seq_id(seq_id, self._num_shards)
        reader = self._get_reader(shard_idx)
        return reader is not None and seq_id in reader

    def __getitem__(self, seq_id):
        shard_idx = shard_index_for_seq_id(seq_id, self._num_shards)
        reader = self._get_reader(shard_idx)
        if reader is None:
            raise KeyError(seq_id)
        return reader[seq_id]

    def keys(self):
        all_keys = []
        for i in range(self._num_shards):
            reader = self._get_reader(i)
            if reader is not None:
                all_keys.extend(reader.keys())
        return all_keys

    def __len__(self):
        total = 0
        for i in range(self._num_shards):
            reader = self._get_reader(i)
            if reader is not None:
                total += len(reader)
        return total

    def close(self):
        for reader in self._readers:
            if reader is not None and reader is not _MISSING_SHARD:
                reader.close()
        self._readers = [None] * self._num_shards

    def __del__(self):
        self.close()


def open_archive(path):
    """Factory: return the appropriate reader for a single-file or sharded archive.

    Args:
        path: Path to a ``.pngrph`` file or a directory containing shard files.

    Returns:
        GraphArchiveReader or ShardedGraphArchiveReader.
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".pngrph":
        return GraphArchiveReader(path)
    if path.is_dir():
        shard_files = list(path.glob("graphs.shard-*-of-*.pngrph"))
        if shard_files:
            return ShardedGraphArchiveReader.from_directory(path)
    raise FileNotFoundError(f"No valid archive found at {path}")
