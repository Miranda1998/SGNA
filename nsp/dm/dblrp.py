import pickle as pkl
import time
from multiprocessing import Manager, Pool
from tqdm import tqdm

import json

import numpy as np
from nsp.two_sp.dblrp import DroneBaseLocationRoutingProblem
from nsp.utils.dblrp import get_path
from .dm import DataManager

import torch
from nsp.scenario_gen.model import CVAE  # 导入定义的cVAE模型
from nsp.utils.dblrp import inverse_points_minmax

import pdb
import traceback

# def hash_expected(sol, scenario, scenario_subset):
#     hashed_scenario_subset = list(map(lambda x: tuple(x.flatten()[:20]), scenario_subset))
#     return frozenset(sol.items()), tuple(hashed_scenario_subset), tuple(scenario.flatten()[:20])
#

def hash_expected(sol, scenario, scenario_subset):
    # 只取前 100 个元素
    # scenario_flattened = scenario.flatten()[:10]  # 取前 100 个元素
    scenario_flattened = [tuple(x.flatten()[:10]) for x in scenario]
    hashed_scenario_subset = [tuple(x.flatten()[:10]) for x in scenario_subset]  # 只取前 100 个元素

    # 返回更新后的哈希值
    return frozenset(sol.items()), tuple(hashed_scenario_subset), tuple(scenario_flattened)


# def convert_to_hashable(obj, path=""):
#     """
#     Recursively converts an object into a hashable type (e.g., tuple, frozenset).
#     Converts numpy arrays, lists, dicts, and other iterable objects into tuple/frozenset.
#
#     :param obj: The object to convert.
#     :param path: The path for debugging and recursion tracing.
#     :return: A hashable version of the object.
#     """
#     # Handle numpy.ndarray objects: convert them into tuples
#     if isinstance(obj, np.ndarray):
#         return tuple(obj.flatten())  # Convert numpy.ndarray to a flat tuple
#
#     # Handle lists: recursively convert each element to a hashable type (tuple)
#     elif isinstance(obj, list):
#         return tuple(convert_to_hashable(x, path + "->list") for x in obj)  # Recursively convert list elements
#
#     # Handle dicts: recursively convert both keys and values to hashable types
#     elif isinstance(obj, dict):
#         return frozenset(
#             (convert_to_hashable(key, path + "->dict->key"), convert_to_hashable(value, path + "->dict->value"))
#             for key, value in obj.items())  # Convert both key and value to hashable
#
#     # Handle sets: sets can be converted to frozensets
#     elif isinstance(obj, set):
#         return frozenset(convert_to_hashable(x, path + "->set") for x in obj)  # Convert each element of set
#
#     # If it's any other object type, we assume it's already hashable and return it
#     else:
#         return obj
#
#
# def hash_expected(sol, scenario, scenario_subset):
#     """
#     Generate a hashable scenario hash based on the provided solution, scenario, and scenario_subset.
#     This function ensures all elements in sol, scenario, and scenario_subset are converted to hashable types.
#
#     :param sol: A dictionary containing solution information.
#     :param scenario: A list or other object that will be converted into a tuple.
#     :param scenario_subset: A list of scenarios to be converted into hashable forms.
#     :return: A tuple containing three hashable components.
#     """
#     # Convert all items in 'sol' to hashable types (keys and values)
#     sol_items = [(key, convert_to_hashable(value, "sol->" + str(key))) for key, value in sol.items()]
#
#     # Convert all items in 'scenario_subset' to hashable types
#     hashed_scenario_subset = [convert_to_hashable(x, "scenario_subset") for x in scenario_subset]
#
#     # Convert 'scenario' to hashable type
#     scenario_hashable = convert_to_hashable(scenario, "scenario")
#
#     # Return the final hashable representation (frozenset for sol items, tuples for other components)
#     return frozenset(sol_items), tuple(hashed_scenario_subset), scenario_hashable


class DroneBaseLocationRoutingDataManager(DataManager):
    def __init__(self, problem_config):

        self.cfg = problem_config

        self.rng = np.random.RandomState()
        self.rng.seed(self.cfg.seed)

        self.instance_path = get_path(self.cfg.data_path, self.cfg, "inst")
        self.ml_data_p_path = get_path(self.cfg.data_path, self.cfg, "ml_data_p")
        self.ml_data_e_path = get_path(self.cfg.data_path, self.cfg, "ml_data_e")

    def generate_instance(self):
        """
        Generate a Drone Base Location Routing problem.
        Outputs as a gurobi model.
        """
        print("Generating instance...")

        self.instance = {}
        self._get_problem_data(self.cfg, self.instance)
        self._generate_first_stage_data(self.cfg, self.instance, self.rng)

        pkl.dump(self.instance, open(self.instance_path, 'wb'))

    @staticmethod
    def _get_problem_data(cfg, inst):
        """ Stores generic problem information. """
        inst['n_vessels'] = cfg.n_vessels
        inst['n_bases'] = cfg.n_bases
        inst['integer_second_stage'] = cfg.flag_integer_second_stage
        inst['bound_tightening_constrs'] = cfg.flag_bound_tightening
        inst['n_samples_p'] = cfg.n_samples_p
        inst['n_samples_per_scenario'] = cfg.n_samples_per_scenario
        inst['n_samples_e'] = cfg.n_samples_e
        inst['n_max_scenarios_in_tr'] = cfg.n_max_scenarios_in_tr
        inst['tr_split'] = cfg.tr_split
        inst['time_limit'] = cfg.time_limit
        inst['mip_gap'] = cfg.mip_gap
        inst['verbose'] = cfg.verbose
        inst['cVAE_model_path'] = cfg.cVAE_model_path
        inst['x_hist_path'] = cfg.x_hist_path
        inst['minmax_norm'] = cfg.minmax_norm
        inst['base_costs_each'] = cfg.base_costs_each
        inst['T'] = cfg.T
        inst['n_drones'] = cfg.n_drones
        inst['drone_each_base'] = cfg.drone_each_base
        inst['max_route_time'] = cfg.max_route_time
        inst['base_pos_path'] = cfg.base_pos_path
        inst['drone_speed'] = cfg.drone_speed
        inst['observation_time'] = cfg.observation_time

    @staticmethod
    def _generate_first_stage_data(cfg, inst, rng):
        """ Computes and stores information for first stage problem. """

        inst['reward'] = rng.rand(cfg.n_vessels)

        bases_set = [f'u{i}' for i in range(inst['n_bases'])]
        vessels_set = [f'v{j}' for j in range(inst['n_vessels'])]
        # rng = np.random.default_rng(1998)  # 使用固定的种子

        inst['base_costs'] = {}
        for u in bases_set:
            inst['base_costs'][u] = inst['base_costs_each']  # base opening costs between 50

        inst['reward'] = {}
        for v in vessels_set:
            inst['reward'][v] = rng.rand() * 20 + 10  # reward between 10 and 30

        for u in bases_set:
            inst['reward'][u] = 0  # reward between 10 and 30

        inst['base_pos'] = np.load(inst['base_pos_path'])

    def _load_instances(self):
        """ Loads instances files. """
        self.instance = pkl.load(open(self.instance_path, 'rb'))

    def generate_dataset_per_scenario(self, n_procs):
        """ Generate dataset for training ML models. """
        self._load_instances()

        print("Generating NN-P dataset for machine learning... ")
        print(f"  PROBLEM: dblrp_{self.instance['n_bases']}_{self.instance['n_vessels']}")

        data = []
        total_time = time.time()
        probs = np.linspace(0.1, 0.9, 9)

        two_sp = DroneBaseLocationRoutingProblem(self.instance)

        # sample a set of scenarios
        n_scenarios_to_sample = self.instance['n_samples_p'] // self.instance['n_samples_per_scenario']
        scenarios = self._sample_n_scenarios(n_scenarios_to_sample, self.instance)

        with Manager() as manager:

            mp_data_list = manager.list()

            # Get costs for suboptimal solutions for each scenario
            procs_to_run = []
            for scenario in scenarios:
                for j in range(self.instance['n_samples_per_scenario']):
                    p = self.rng.choice(probs)  # prob. of zero
                    y_subopt = self._get_pure_random_x(prob=p, size=self.instance['n_bases'])
                    procs_to_run.append((np.array(scenario), y_subopt))

            if n_procs == -1:
                pool = Pool()
            else:
                pool = Pool(n_procs)
            for scenario, y_subopt in procs_to_run:

                pool.apply_async(self.solve_second_stage_subopt_mp,
                                 args=(
                                     scenario,
                                     y_subopt,
                                     two_sp,
                                     self.instance,
                                     mp_data_list))

            pool.close()
            pool.join()

            data = list(mp_data_list)

        total_time = time.time() - total_time

        # get train/validation split, then store

        tr_data, val_data = self._get_data_split(data, self.instance)
        ml_data = {
            "tr_data": tr_data,
            "val_data": val_data,
            "data": data,
            "total_time": total_time
        }

        print("Total Time:         ", total_time)
        print("Dataset size:       ", len(data))
        print("Train dataset size: ", len(tr_data))
        print("Valid dataset size: ", len(val_data))

        pkl.dump(ml_data, open(self.ml_data_p_path, 'wb'))


    def solve_second_stage_subopt_mp(self, scenario, y_subopt, two_sp, instance, mp_data_list):
        """ Obtains the cost of the suboptimal first stage solution.  """
        try:
            time_ = time.time()
            x_subopt_obj = two_sp.get_second_stage_objective(
                y_subopt,
                scenario,
                gap=instance['mip_gap'],
                time_limit=instance['time_limit'],
                verbose=instance['verbose'])

            x_subopt_features = self._get_feature_vector(y_subopt, scenario)

            time_ = time.time() - time_

            mp_data_list.append({
                "demands": scenario,
                "y": y_subopt,
                "obj": x_subopt_obj,
                "features": x_subopt_features,
                "time": time_})


        except Exception as e:
            print(f"Failed to get second stage objective for suboptimal x in time limit ({instance['time_limit']}s)")
            print(f"  exception: {e}")

            # Print the full traceback for better debugging
            print("Detailed error trace:")
            print(traceback.format_exc())

            print(f"  x: {x_subopt_obj}")

        return


    def generate_dataset_expected(self, n_procs):
        """ Generate dataset for training ML models. """
        self._load_instances()
        print("Generating NN-E dataset for machine learning... ")
        print(f"  PROBLEM: dblrp_{self.instance['n_bases']}_{self.instance['n_vessels']}")

        data = []
        total_time = time.time()
        probs = np.linspace(0.0, 0.9, 10)

        two_sp = DroneBaseLocationRoutingProblem(self.instance)

        with Manager() as manager:

            # Get costs from suboptimal first stage solutions.
            print("  Getting objective for first varying first stage. ")

            # get set of all processes to run
            procs_to_run = []
            n_procs_to_run = 0
            n_samples = self.instance['n_samples_e']
            print('n_samples', n_samples)

            for _ in range(n_samples):
                # choose a random first stage optimal solution and perturb it
                p = self.rng.choice(probs)  # prob. of swapping bits in sol
                first_stage_sol = self._get_pure_random_x(prob=p, size=self.instance['n_bases'])

                # choose a random subset of demands
                n_second_stage = self.rng.randint(1, self.instance['n_max_scenarios_in_tr'])

                print("Sampling scenarios...")
                print('n_second_stage', n_second_stage)
                scenario_subset = self._sample_n_scenarios(n_second_stage, self.instance)

                # update
                procs_to_run.append((first_stage_sol, scenario_subset))
                n_procs_to_run += n_second_stage

            # initialize data structures for storing solutions
            mp_cost_dict = manager.dict()
            mp_time_dict = manager.dict()
            mp_count = manager.Value('i', 0)

            if n_procs == -1:
                pool = Pool()
            else:
                pool = Pool(n_procs)

            # Initialize the progress bar
            total_tasks = sum(len(scenario_subset) for _, scenario_subset in procs_to_run)
            progress_bar = tqdm(total=total_tasks, desc="Processing tasks", unit="task")

            # Helper function to update progress
            def update_progress(_):
                progress_bar.update(1)

            for first_stage_sol, scenario_subset in procs_to_run:
                for scenario in scenario_subset:
                    # Apply each task asynchronously
                    pool.apply_async(self.solve_subset_second_stage_cost_mp_expected,
                                     args=(first_stage_sol,
                                           scenario,
                                           scenario_subset,
                                           two_sp,
                                           self.instance,
                                           mp_cost_dict,
                                           mp_time_dict,
                                           mp_count,
                                           n_procs_to_run),
                                     callback=update_progress)  # update progress after each task

            pool.close()
            pool.join()

            # Close the progress bar once the processing is done
            progress_bar.close()

            print("Storing results... ", end="")

            for first_stage_sol, scenario_subset in procs_to_run:
                objs, scens, times = [], [], []
                for scenario in scenario_subset:
                    # add items to subset if and only if no errors occurred
                    scenario_hash = hash_expected(first_stage_sol, scenario, scenario_subset)
                    if scenario_hash in mp_cost_dict:
                        objs.append(mp_cost_dict[scenario_hash])
                        times.append(mp_time_dict[scenario_hash])
                        scens.append(scenario)

                data.append({
                    "x": first_stage_sol,
                    "obj_vals": objs,
                    "obj_mean": np.mean(objs),
                    "demands": scens,
                    "time": np.sum(times),
                    "times": times})

            print("Done")

        total_time = time.time() - total_time

        mp_time = list(map(lambda x: x['time'], data))
        mp_time = np.sum(mp_time)

        # get train/validation split, then store
        tr_data, val_data = self._get_data_split(data, self.instance)
        ml_data = {
            "tr_data": tr_data,
            "val_data": val_data,
            "data": data,
            "total_time": total_time,
            "mp_time": mp_time,
        }

        print("Time (No MP):       ", mp_time)
        print("Total Time:         ", total_time)
        print("Dataset size:       ", len(data))
        print("Train dataset size: ", len(tr_data))
        print("Valid dataset size: ", len(val_data))

        pkl.dump(ml_data, open(self.ml_data_e_path, 'wb'))


    def _get_pure_random_x(self, prob, size):
        """ Modeify bits in a solution x with probability p.  """
        x_sub = self.rng.choice([0.0, 1.0], p=[prob, 1 - prob], size=size)
        y_sub_dict = self._sol_vect_to_dict(x_sub)
        return y_sub_dict

    def _sol_vect_to_dict(self, x_vect):
        """ Converts a solution vector to a vector. """
        bases_set = [f'u{i}' for i in range(self.instance['n_bases'])]

        y_sol = {}
        for index in range(x_vect.size):
            y_sol[f"y_u{index}"] = x_vect[index]

        return y_sol

    def _get_feature_vector(self, x_sol, demand):
        """ Gets the simple feature vector (x, deamnds). """
        x_vect = self._sol_dict_to_vect(x_sol)
        features = x_vect.tolist() + demand.tolist()
        return features

    def _sol_dict_to_vect(self, x_sol):
        """ Converts a solution dictionary to a vector. """
        x_vect = np.zeros(len(x_sol))

        for k, v in x_sol.items():
            index = int(k.split('_')[1][1:])
            x_vect[index] = v
        return x_vect

    @staticmethod
    def _sample_n_scenarios(n, inst):

        # 设备选择
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化模型
        model = CVAE(x_dim=5, y_dim=2, z_dim=32, hid_dim=128, horizon=72).to(device)

        # 加载训练好的模型参数
        model.load_state_dict(torch.load(inst['cVAE_model_path'], map_location=device))
        model.eval()  # 设置为评估模式

        x_hist_np = np.load(inst['x_hist_path'])  # x_hist 是一个包含客户历史位置的数据 [n, T, 2]

        # 将 Numpy 数组转换为 PyTorch 张量
        x_hist = torch.tensor(x_hist_np, dtype=torch.float32)
        # 将张量移到正确的设备上（例如 GPU 或 CPU）
        x_hist = x_hist.to(device)  # 假设 device 是 'cuda' 或 'cpu'

        # 假设 stat 是额外的条件，如果没有可以传 None
        stat = None  # 或者提供额外的统计数据

        # 使用模型生成未来轨迹，K 是生成轨迹的数量
        K = 1  # 生成1条轨迹,每次采样只采一次

        # 导入标准化min-max
        with open(inst['minmax_norm'], 'r') as f:
            minmax_norm = json.load(f)

        scenarios = []
        for _ in range(n):
            # 生成随机历史位置数据作为输入
            trajs, prior_logprob, mus, logvars = model.sample(x_hist, stat=stat, K=K)
            hist_real = inverse_points_minmax(trajs, minmax_norm)[0].detach().cpu().numpy()
            hist_real_half = hist_real[:, ::2, :].copy()  # 每隔一个时刻取一次，保留奇数索引
            scenarios.append(hist_real_half)
            print('Generated scenario shape:', hist_real_half.shape)

        return scenarios

    def _get_data_split(self, data, instance):
        """ Gets train/validation splits for the data.  """
        perm = self.rng.permutation(len(data))

        split_idx = int(instance['tr_split'] * (len(data)))
        tr_idx = perm[:split_idx].tolist()
        val_idx = perm[split_idx:].tolist()

        tr_data = [data[i] for i in tr_idx]
        val_data = [data[i] for i in val_idx]

        return tr_data, val_data


    def solve_subset_second_stage_cost_mp_expected(self, first_stage_sol, scenario, scenario_subset, two_sp, instance,
                                                     mp_cost_dict, mp_time_dict, mp_count, n_procs_to_run):
        """ Obtains the cost of the suboptimal first stage solution.  """
        try:

            time_ = time.time()

            second_stage_obj = two_sp.get_second_stage_objective(
                first_stage_sol,
                scenario,
                gap=instance['mip_gap'],
                time_limit=instance['time_limit'],
                verbose=instance['verbose'])

            time_ = time.time() - time_

            mp_count.value += 1
            count = mp_count.value

            if count % 1000 == 0:
                print(f'Solving LP {count} / {n_procs_to_run}')

            mp_cost_dict[hash_expected(first_stage_sol, scenario, scenario_subset)] = second_stage_obj
            mp_time_dict[hash_expected(first_stage_sol, scenario, scenario_subset)] = time_

        except Exception as e:
            print(f"Failed to get second stage objective for suboptimal x in time limit ({instance['time_limit']}s)")
            print(f"  exception: {e}")
            print(f"  x: {first_stage_sol}")

        return