"""Management-plane vocabulary frozen for the EPIC-30 contract tests."""

from __future__ import annotations

from enum import StrEnum


class ManagementRole(StrEnum):
    """RBAC roles frozen by the accepted T-306-A RFC."""

    VIEWER = "Viewer"
    DEVELOPER = "Developer"
    OPERATOR = "Operator"
    ADMIN = "Admin"
