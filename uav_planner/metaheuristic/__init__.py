"""Metaheuristic planners for continuous UAV path optimization."""

__all__ = ["ao_optimize", "mso_optimize", "iamso_optimize"]


def __getattr__(name):
    if name == "ao_optimize":
        from .ao import ao_optimize

        return ao_optimize
    if name == "mso_optimize":
        from .mso import mso_optimize

        return mso_optimize
    if name == "iamso_optimize":
        from .iamso import iamso_optimize

        return iamso_optimize
    raise AttributeError(name)
