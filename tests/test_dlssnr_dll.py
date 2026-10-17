import importlib
import struct

import pytest


def weight_map(rows):
    """Small independently serialized native records, in caller-specified order."""
    result = bytearray(8)
    for name, payload in rows:
        encoded = name.encode("utf-8")
        size = 40 + len(payload)
        result.extend(struct.pack("<Q", len(encoded)) + encoded)
        result.extend(struct.pack("<QQQI", size, size, len(payload), 1))
        result.extend(payload)
        result.extend(struct.pack("<IIQI", 7, 9, 1, len(payload) // 2))
    struct.pack_into("<Q", result, 0, len(result))
    return bytes(result)


def pe_image(resource):
    """Minimal PE32+ with a real RCDATA/name/language resource tree."""
    result = bytearray(0x300 + len(resource))
    result[:2] = b"MZ"
    struct.pack_into("<I", result, 0x3C, 0x80)
    result[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HH", result, 0x84, 0x8664, 1)
    struct.pack_into("<H", result, 0x94, 0xF0)
    struct.pack_into("<H", result, 0x98, 0x20B)
    struct.pack_into("<I", result, 0x98 + 108, 16)
    struct.pack_into("<II", result, 0x98 + 128, 0x1000, 0x100 + len(resource))
    result[0x188:0x190] = b".rsrc\0\0\0"
    struct.pack_into("<IIII", result, 0x190, 0x100 + len(resource), 0x1000, 0x100 + len(resource), 0x200)
    struct.pack_into("<HHII", result, 0x20C, 0, 1, 10, 0x80000020)
    struct.pack_into("<HHII", result, 0x22C, 1, 0, 0x80000080, 0x80000040)
    struct.pack_into("<HHII", result, 0x24C, 0, 1, 0x409, 0x60)
    struct.pack_into("<IIII", result, 0x260, 0x1100, len(resource), 0, 0)
    name = "WEIGHTS_HT".encode("utf-16-le")
    struct.pack_into("<H", result, 0x280, len(name) // 2)
    result[0x282 : 0x282 + len(name)] = name
    result[0x300:] = resource
    return bytes(result)


def test_extract_and_patch_preserve_headers_metadata_order_and_overlay():
    native = importlib.import_module("musubi_tuner.dlssnr.dll")
    resource = weight_map([("block2.layer0.layer", b"\x01\x80"), ("block1.layer0.layer", b"\x02\x03\x04\x05")])
    dll = pe_image(resource) + b"certificate-and-overlay"
    parsed = native.read_weights(dll)
    assert parsed.offset == 0x300 and parsed.size == len(resource)
    assert list(parsed.records) == ["block2.layer0.layer", "block1.layer0.layer"]
    row = parsed.records["block2.layer0.layer"]
    assert (row.dtype_code, row.metadata0, row.metadata1, row.dimensions) == (1, 7, 9, (1,))
    original = {name: bytes(record.payload) for name, record in parsed.records.items()}
    assert parsed.replace(dll, original) == dll
    original["block1.layer0.layer"] = b"\xa1\xa2\xa3\xa4"
    updated = parsed.replace(dll, original)
    expected = dll.replace(b"\x02\x03\x04\x05", b"\xa1\xa2\xa3\xa4")
    assert updated == expected
    assert bytes(native.read_weights(updated).records["block1.layer0.layer"].payload) == original["block1.layer0.layer"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "size"])
def test_patch_rejects_record_inventory_or_size_changes(mutation):
    native = importlib.import_module("musubi_tuner.dlssnr.dll")
    dll = pe_image(weight_map([("a", b"\x00\x00")]))
    replacements = {"a": b"\x00\x00"}
    if mutation == "missing":
        replacements.clear()
    elif mutation == "extra":
        replacements["b"] = b"\x00\x00"
    else:
        replacements["a"] = b"\x00\x00\x00\x00"
    with pytest.raises(ValueError, match="inventory|size"):
        native.read_weights(dll).replace(dll, replacements)


@pytest.mark.parametrize("mutation", ["magic", "pe32", "directory", "language", "size", "dtype", "duplicate", "truncated"])
def test_malformed_dlls_fail_closed(mutation):
    native = importlib.import_module("musubi_tuner.dlssnr.dll")
    resource = weight_map([("a", b"\x01\x02")])
    dll = bytearray(pe_image(resource))
    if mutation == "magic":
        dll[:2] = b"NO"
    elif mutation == "pe32":
        struct.pack_into("<H", dll, 0x98, 0x10B)
    elif mutation == "directory":
        struct.pack_into("<I", dll, 0x214, 0xFFFFFFF0)
    elif mutation == "language":
        struct.pack_into("<H", dll, 0x24E, 2)
    elif mutation == "size":
        struct.pack_into("<Q", dll, 0x300, 8)
    elif mutation == "dtype":
        struct.pack_into("<I", dll, 0x300 + 8 + 8 + 1 + 24, 99)
    elif mutation == "duplicate":
        dll = pe_image(weight_map([("a", b"\x01\x02"), ("a", b"\x03\x04")]))
    else:
        dll = dll[:-1]
    with pytest.raises(ValueError):
        native.read_weights(bytes(dll))
