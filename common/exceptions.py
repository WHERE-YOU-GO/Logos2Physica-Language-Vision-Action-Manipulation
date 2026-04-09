from __future__ import annotations


class ProjectError(Exception):
    """Base exception for the project."""

    def __init__(self, message: str = "") -> None:
        self.message = message or self.__class__.__name__
        super().__init__(self.message)


class PerceptionError(ProjectError):
    """Raised when perception processing fails."""


class DetectionNotFoundError(PerceptionError):
    """Raised when a requested detection cannot be found."""


class TargetResolutionError(ProjectError):
    """Raised when semantic target resolution fails."""


class PlanningError(ProjectError):
    """Raised when planning cannot produce a valid result."""


class SafetyViolationError(ProjectError):
    """Raised when a command violates a safety boundary."""


class ExecutionError(ProjectError):
    """Raised when motion or gripper execution fails."""


class VerificationError(ProjectError):
    """Raised when post-action verification fails."""


__all__ = [
    "ProjectError",
    "PerceptionError",
    "DetectionNotFoundError",
    "TargetResolutionError",
    "PlanningError",
    "SafetyViolationError",
    "ExecutionError",
    "VerificationError",
]
