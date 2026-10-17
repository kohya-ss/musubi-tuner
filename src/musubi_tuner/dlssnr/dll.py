# PE/WEIGHTS_HT format adapted from MLX-DLSS (Apache-2.0).
# Modified: bounded parsing and same-size payload replacement, not serialization.
# See NOTICE.md and LICENSE.MLX-DLSS in this directory.

"""Read external PE32+ DLLs without loading or executing their code."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field


def _read(data, offset: int, fmt: str, end: int | None = None) -> tuple:
    size = struct.calcsize(fmt)
    end = len(data) if end is None else min(end, len(data))
    if offset < 0 or offset + size > end:
        raise ValueError(f"truncated DLL/resource at byte {offset}")
    return struct.unpack_from(fmt, data, offset)


def _resource_location(data: bytes, resource_name: str = "WEIGHTS_HT", *, required: bool = True) -> tuple[int, int] | None:
    if data[:2] != b"MZ":
        raise ValueError("source is not a PE DLL (missing MZ)")
    pe = _read(data, 0x3C, "<I")[0]
    signature, machine, count = _read(data, pe, "<4sHH")
    if signature != b"PE\0\0" or machine != 0x8664:
        raise ValueError("expected an x64 PE DLL")
    optional_size = _read(data, pe + 20, "<H")[0]
    optional = pe + 24
    if optional_size < 136 or optional + optional_size > len(data):
        raise ValueError("truncated PE optional header")
    if _read(data, optional, "<H")[0] != 0x20B:
        raise ValueError("expected a PE32+ DLL")
    if _read(data, optional + 108, "<I")[0] < 3:
        raise ValueError("PE header has no resource directory")
    rva, size = _read(data, optional + 128, "<II")
    if not rva or not size:
        raise ValueError("PE image has no resource directory")
    sections = []
    for index in range(count):
        header = optional + optional_size + index * 40
        _read(data, header, "<40s")
        sections.append(_read(data, header + 8, "<IIII"))

    def file_offset(address, length):
        matches = []
        for virtual_size, virtual_address, raw_size, raw_offset in sections:
            relative = address - virtual_address
            if 0 <= relative < max(virtual_size, raw_size) and relative + length <= raw_size:
                offset = raw_offset + relative
                if offset + length > len(data):
                    raise ValueError("truncated PE file-backed section")
                matches.append(offset)
        if len(matches) != 1:
            raise ValueError("resource RVA is outside or ambiguous in PE file-backed sections")
        return matches[0]

    base = file_offset(rva, size)

    def read(relative, fmt):
        if relative < 0:
            raise ValueError("negative resource offset")
        return _read(data, base + relative, fmt, base + size)

    def entries(relative):
        named, numeric = read(relative + 12, "<HH")
        result = []
        for index in range(named + numeric):
            identifier, target = read(relative + 16 + 8 * index, "<II")
            if identifier & 0x80000000:
                text_offset = identifier & 0x7FFFFFFF
                length = read(text_offset, "<H")[0]
                identifier = read(text_offset + 2, f"<{length * 2}s")[0].decode("utf-16-le")
            result.append((identifier, target))
        return result

    directory = 0
    for identifier in (10, resource_name):
        matches = [target for key, target in entries(directory) if key == identifier]
        if not matches and not required:
            return None
        if len(matches) != 1 or not matches[0] & 0x80000000:
            raise ValueError(f"missing or ambiguous resource directory {identifier}")
        directory = matches[0] & 0x7FFFFFFF
    languages = entries(directory)
    if len(languages) != 1 or languages[0][1] & 0x80000000:
        raise ValueError(f"{resource_name} must have exactly one language/data entry")
    payload_rva, payload_size, _, _ = read(languages[0][1], "<IIII")
    return file_offset(payload_rva, payload_size), payload_size


def has_weight_resource(dll: bytes, resource_name: str) -> bool:
    """Check a resource name without hiding malformed or ambiguous PE directories."""
    return _resource_location(dll, resource_name, required=False) is not None


@dataclass(frozen=True)
class SerializedWeight:
    offset: int
    payload: memoryview
    dtype_code: int
    metadata0: int
    metadata1: int
    dimensions: tuple[int, ...]


@dataclass(frozen=True)
class WeightResource:
    offset: int
    size: int
    records: dict[str, SerializedWeight]
    source: bytes = field(repr=False)

    def replace(self, dll: bytes, payloads: dict[str, bytes]) -> bytes:
        if dll != self.source:
            raise ValueError("resource does not belong to this DLL")
        if payloads.keys() != self.records.keys():
            raise ValueError("replacement record inventory differs from the DLL")
        result = bytearray(dll)
        for name, record in self.records.items():
            payload = payloads[name]
            if len(payload) != len(record.payload):
                raise ValueError(f"{name}: replacement payload size differs from the DLL")
            result[record.offset : record.offset + len(payload)] = payload
        return bytes(result)


def read_weights(dll: bytes, *, resource_name: str = "WEIGHTS_HT") -> WeightResource:
    """Locate the resource through PE RVAs; preserve every serialized metadata byte."""
    start, size = _resource_location(dll, resource_name)
    end = start + size
    if _read(dll, start, "<Q", end)[0] != size:
        raise ValueError("serialized weight map size differs from resource size")
    records = {}
    cursor = start + 8
    view = memoryview(dll)
    while cursor < end:
        name_length = _read(dll, cursor, "<Q", end)[0]
        cursor += 8
        if not 1 <= name_length <= 4096:
            raise ValueError("invalid serialized tensor name length")
        name = _read(dll, cursor, f"<{name_length}s", end)[0].decode("utf-8")
        cursor += name_length
        outer, inner, byte_count, dtype_code = _read(dll, cursor, "<QQQI", end)
        record_end = cursor + 8 + outer
        cursor += 28
        if outer != inner or outer < 40 or record_end > end:
            raise ValueError(f"{name}: invalid serialized record size")
        if dtype_code != 1 or byte_count > record_end - cursor - 16:
            raise ValueError(f"{name}: invalid packed dtype or payload size")
        payload_offset = cursor
        cursor += byte_count
        metadata0, metadata1, dimension_count = _read(dll, cursor, "<IIQ", record_end)
        cursor += 16
        if not 1 <= dimension_count <= 16 or cursor + 4 * dimension_count != record_end:
            raise ValueError(f"{name}: invalid serialized dimensions")
        dimensions = _read(dll, cursor, f"<{dimension_count}I", record_end)
        if not all(dimensions) or math.prod(dimensions) * 2 != byte_count:
            raise ValueError(f"{name}: dimensions do not match payload size")
        if name in records:
            raise ValueError(f"duplicate serialized tensor {name}")
        records[name] = SerializedWeight(
            payload_offset, view[payload_offset : payload_offset + byte_count], dtype_code, metadata0, metadata1, dimensions
        )
        cursor = record_end
    if not records:
        raise ValueError("empty serialized weight map")
    return WeightResource(start, size, records, dll)
