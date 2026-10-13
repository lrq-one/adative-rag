class CommerceModelUnavailable(ValueError):
    """A real model failed or lacks credentials; never fall back to a fake response."""
