import nsp.params as params


def factory_dm(problem):
    cfg = getattr(params, problem)

    if "cflp" in problem:
        print("Loading CFLP data manager...")
        from .cflp import FacilityLocationDataManager
        return FacilityLocationDataManager(cfg)

    if "dblrp" in problem:
        print("Loading DBLRP data manager...")
        from .dblrp import DroneBaseLocationRoutingDataManager
        return DroneBaseLocationRoutingDataManager(cfg)

    else:
        raise ValueError("Invalid problem type!")
