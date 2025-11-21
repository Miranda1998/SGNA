from types import SimpleNamespace

# -----------------------------------------#
#   Capacited Facility Location Problem   #
# -----------------------------------------#

# cflp_10_10 = SimpleNamespace(
#     n_facilities=10,
#     n_customers=10,
#     ratio=2.0,
#     flag_integer_second_stage=True,
#     flag_bound_tightening=True,
#     n_samples_p=10000,              # NN-P specific data generation
#     n_samples_per_scenario=10,      # NN-P specific data generation
#     n_samples_e=5000,               # NN-E specific data generation
#     n_max_scenarios_in_tr=100,      # NN-E specific data generation
#     time_limit=60,                  # data generation
#     mip_gap=0.01,                   # data generation
#     tr_split=0.80,                  # data generation
#     verbose=0,                      # data generation
#     seed=7,
#     data_path='./data'
# )

cflp_10_10 = SimpleNamespace(
    n_facilities=10,
    n_customers=10,
    ratio=2.0,
    flag_integer_second_stage=True,
    flag_bound_tightening=True,
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=2,      # NN-P specific data generation
    n_samples_e=5,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=10,                  # data generation
    mip_gap=0.01,                   # data generation
    tr_split=0.80,                  # data generation
    verbose=0,                      # data generation
    seed=7,
    data_path='./data'
)

my_drone_each_base = 1
dblrp_10_10 = SimpleNamespace(
    n_bases=10,
    n_vessels=10,
    ratio=2.0,
    flag_integer_second_stage=True,
    flag_bound_tightening=True,
    n_samples_p=1,              # NN-P specific data generation
    n_samples_per_scenario=1,      # NN-P specific data generation
    n_samples_e=5,               # NN-E specific data generation
    n_max_scenarios_in_tr=100,      # NN-E specific data generation
    time_limit=60,                  # data generation
    mip_gap=0.01,                   # data generation
    tr_split=0.80,                  # data generation
    verbose=0,                      # data generation
    seed=7,
    base_costs_each=10,
    T=72,
    n_drones=10 * my_drone_each_base,
    drone_each_base=my_drone_each_base,
    max_route_time=36,
    drone_speed=74,  # 40节换算成km/h
    observation_time=1,
    data_path='./data',
    cVAE_model_path='./nsp/scenario_gen/cvae_epoch180_20251013_151341.pt',
    x_hist_path='./data/dblrp/dblrp_10_10_scenarios_xhist.npy',
    minmax_norm='./data/dblrp/minmax_norm.json',
    base_pos_path='./data/dblrp/dblrp_10_10_base_positions.npy'
)

dblrp_5_10 = SimpleNamespace(
    n_bases=5,
    n_vessels=10,
    ratio=2.0,
    flag_integer_second_stage=True,
    flag_bound_tightening=True,
    n_samples_p=1000,              # NN-P specific data generation
    n_samples_per_scenario=10,      # NN-P specific data generation
    n_samples_e=2,               # NN-E specific data generation
    n_max_scenarios_in_tr=2,      # NN-E specific data generation
    time_limit=60,                  # data generation
    mip_gap=0.01,                   # data generation
    tr_split=0.80,                  # data generation
    verbose=0,                      # data generation
    seed=7,
    base_costs_each=10,
    T=36,
    n_drones=5 * my_drone_each_base,
    drone_each_base=my_drone_each_base,
    max_route_time=36,
    drone_speed=74,  # 40节换算成km/h
    observation_time=1,
    data_path='./data',
    cVAE_model_path='./nsp/scenario_gen/cvae_epoch180_20251013_151341.pt',
    x_hist_path='./data/dblrp/dblrp_10_10_scenarios_xhist.npy',
    minmax_norm='./data/dblrp/minmax_norm.json',
    base_pos_path='./data/dblrp/dblrp_5_10_base_positions.npy'
)
