import equinox as eqx


class AbstractPotentialIntegrator(eqx.Module, strict=True):
    """Base class for a method of integrating a potential onto
    the exit plane.
    """

    requires_inverse_rotation: eqx.AbstractClassVar[bool]
