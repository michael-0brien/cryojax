import cryojax.simulator as cxs
from cryojax.simulator import DiscreteStructuralEnsemble


def test_conformation(potential, pose, projection_method, transfer_theory, config):
    potential = tuple([potential for _ in range(3)])
    structure = DiscreteStructuralEnsemble(potential, pose, conformation=0)
    theory = cxs.LinearImageModel(structure, projection_method, transfer_theory, config)
    _ = theory.render()
