from .two_sp import TwoStageStocProg


def factory_two_sp(problem, inst, sampler=None):
    if 'cflp' in problem:
        from .cflp import FacilityLocationProblem
        return FacilityLocationProblem(inst)

    if 'dblrp' in problem:
        from .dblrp import DroneBaseLocationRoutingProblem
        return DroneBaseLocationRoutingProblem(inst)

    else:
        raise Exception(f"nsp.utils not defined for problem class {problem}")
