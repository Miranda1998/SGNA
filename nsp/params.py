from types import SimpleNamespace

# 算法相关参数
time_limit=60
mip_gap=0.01
tr_split=0.80
verbose=0
seed=7

# base相关参数
base_costs_each=20
my_drone_each_base = 3

# 无人机相关参数
max_route_time=24
drone_speed=55.56
observation_time=1

# 其他参数
T=36
time_slot=10

n_bases=10
n_vessels=10
dblrp_10_10 = SimpleNamespace(
    n_bases=n_bases,
    fixed_bases=-1,  # fixed_base = -1 代表不固定基站数量
    n_vessels=n_vessels,
    ratio=2.0,
    flag_integer_second_stage=True,
    flag_bound_tightening=True,
    n_samples_p=10000,              # NN-P specific data generation
    n_samples_per_scenario=500,      # NN-P specific data generation
    n_samples_e=5000,               # NN-E specific data generation
    n_max_scenarios_in_tr=20,      # NN-E specific data generation
    time_limit=time_limit,         # data generation
    mip_gap=mip_gap,                   # data generation
    tr_split=tr_split,                  # data generation
    verbose=verbose,                      # data generation
    seed=seed,
    base_costs_each=base_costs_each,
    T=T,
    time_slot=time_slot,
    n_drones=n_bases * my_drone_each_base,
    drone_each_base=my_drone_each_base,
    max_route_time=max_route_time,
    drone_speed=drone_speed,  # 节换算成km/h
    observation_time=observation_time,
    data_path='./data',
    cVAE_model_path='./nsp/scenario_gen/best_model.pt',
    x_hist_path='./data/dblrp/x_hist_10.npy',
    minmax_norm='./data/dblrp/minmax_norm.json',
    base_pos_path='./data/dblrp/dblrp_10_base_positions.npy'
)

