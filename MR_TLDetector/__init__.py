"""Reusable components for MR_TLDetector."""

__all__ = ["DetectionConfig", "InsulatorDetector"]


def __getattr__(name):
    """Load the inference backend only when its public API is requested."""
    if name in __all__:
        from .inference import DetectionConfig, InsulatorDetector

        return {"DetectionConfig": DetectionConfig, "InsulatorDetector": InsulatorDetector}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
