from nsp.model2mip import factory_model2mip


def factory_approximator(args, two_sp, model, model_type):
    mipper = factory_model2mip(model_type)

    if 'cflp' in args.problem:
        from .cflp import FacilityLocationProblemApproximator
        return FacilityLocationProblemApproximator(two_sp, model, model_type, mipper)

    elif 'dblrp' in args.problem:
        from .dblrp import DroneBaseLocationRoutingProblemApproximator
        return DroneBaseLocationRoutingProblemApproximator(two_sp, model, model_type, mipper)


    else:
        raise Exception(f"nsp.utils not defined for problem class {args.problem}")
