"""Placement and explicitly modeled scalar 32-bit shared-memory warp accesses."""

from collections import defaultdict
from typing import Literal

from pydantic import Field

from .model import Model


class Access(Model):
    lane: int = Field(ge=0, le=31)
    byte_offset: int = Field(ge=0, le=2**63 - 1)
    active: bool = True


class BankQuery(Model):
    mode: Literal["placement", "scalar-warp"] = "placement"
    bank_count: int = Field(default=32, ge=1, le=128)
    bank_width_bytes: int = Field(default=4, ge=1, le=32)
    base_byte_offset: int = Field(default=0, ge=0, le=2**63 - 1)
    byte_offsets: list[int] = Field(default_factory=list, max_length=4096)
    access_width_bytes: int = Field(default=4, ge=1, le=128)
    operation: Literal["read", "write"] = "read"
    accesses: list[Access] = Field(default_factory=list, max_length=32)


def analyze_banks(query: BankQuery):
    if any(o < 0 or o > 2**63 - 1 for o in query.byte_offsets):
        raise ValueError("Byte offsets must be nonnegative 64-bit integers")

    def placement(offset):
        address = query.base_byte_offset + offset
        banks = sorted(
            {
                (address + b) // query.bank_width_bytes % query.bank_count
                for b in range(query.access_width_bytes)
            }
        )
        return {"byte_offset": offset, "word": address // query.bank_width_bytes, "banks": banks}

    response = {
        "mode": query.mode,
        "bank_count": query.bank_count,
        "bank_width_bytes": query.bank_width_bytes,
        "base_byte_offset": query.base_byte_offset,
        "placement": [placement(o) for o in query.byte_offsets],
        "assumptions": [
            "Byte addresses relative to the explicitly declared base offset",
            "Bank placement alone does not establish a conflict",
        ],
    }
    if query.mode == "placement":
        return response
    if (query.access_width_bytes, query.bank_count, query.bank_width_bytes) != (4, 32, 4):
        raise ValueError(
            "Conflict analysis supports only scalar 32-bit accesses on 32 four-byte banks"
        )
    if not query.accesses:
        raise ValueError("Scalar analysis needs an explicit warp access group")
    lanes = [a.lane for a in query.accesses]
    if len(set(lanes)) != len(lanes):
        raise ValueError("Each lane may contribute only one scalar access")
    active = [a for a in query.accesses if a.active]
    if any((query.base_byte_offset + a.byte_offset) % 4 for a in active):
        raise ValueError("Scalar 32-bit accesses must be four-byte aligned")
    banks, words = defaultdict(set), defaultdict(list)
    for access in active:
        item = placement(access.byte_offset)
        banks[item["banks"][0]].add(item["word"])
        words[item["word"]].append(access.lane)
    collisions = [ls for ls in words.values() if len(ls) > 1]
    if query.operation == "write" and collisions:
        response.update(
            status="unsupported",
            reason="Same-word writes have a data race; no conflict count is reported",
        )
        return response
    response.update(
        status="known",
        active_lanes=len(active),
        serialization_rounds=max(map(len, banks.values()), default=0),
        extra_bank_transactions=sum(max(0, len(ws) - 1) for ws in banks.values()),
        broadcasts=collisions if query.operation == "read" else [],
    )
    response["assumptions"] += [
        "One scalar 32-bit instruction for the declared active lanes of one warp",
        "Same-word reads broadcast; distinct words in one bank serialize",
        "Counts describe this access group, not measured kernel performance",
    ]
    return response
