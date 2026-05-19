"""
Tenant-specific schemas and enums.

Each tenant belongs to a sector that determines which aspects
are relevant and what domain rules apply.
"""

from enum import Enum


class TenantSector(str, Enum):
    """Sector that a tenant operates in.

    Using a string-based Enum guarantees serialisation to JSON
    and prevents accidental comparison mismatches.

    Members:
        FINTECH:   Financial technology sector.
        ECOMMERCE: Online retail / e-commerce sector.
    """

    FINTECH = "fintech"
    ECOMMERCE = "ecommerce"
