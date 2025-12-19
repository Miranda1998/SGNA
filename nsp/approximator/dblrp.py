import time

import gurobipy as gp
import numpy as np
import torch

from .approximator import Approximator


class DroneBaseLocationRoutingProblemApproximator(Approximator):

    def __init__(self, two_sp, model, model_type, mipper):
        """ Constructor for Facility Location Approximator. """
        self.two_sp = two_sp
        self.model = model
        self.model_type = model_type
        self.mipper = mipper

        self.inst = self.two_sp.inst

    def get_master_mip(self):
        """Initialize MIP model with first-stage variables and constraints """
        mip = gp.Model('mipQ')
        x_in = mip.addVars(self.inst['n_bases'], vtype=gp.GRB.BINARY, name='x_in')

        # 添加约束，确保最多只有两个 x_in 等于 1
        mip.addConstr(gp.quicksum(x_in[i] for i in x_in.keys()) == self.inst['fixed_bases'], "fixed_bases_x_in_equals_1")

        mip.update()

        bases_set = [f'u{i}' for i in range(self.inst['n_bases'])]

        # set objective
        objective = 0
        for i in x_in.keys():
            objective += -self.inst['base_costs'][bases_set[i]] * x_in[i]

        mip.setObjective(objective, gp.GRB.MAXIMIZE)
        mip.update()

        return mip

    def get_scenario_embedding(self, n_scenarios, test_set):
        """ Gets the set of scenarios.  """
        scenario_embedding = self.two_sp.get_scenarios(n_scenarios, test_set)
        scenario_embedding = [np.array(scenario_embedding[x]).flatten() for x in range(len(scenario_embedding))]

        # Get embedding if NN-E model.
        if self.model_type == 'nn_e':
            x_scen = np.array(scenario_embedding)
            x_scen = torch.from_numpy(x_scen).float()
            x_scen = torch.reshape(x_scen, (1, x_scen.shape[0], x_scen.shape[1]))
            scenario_embedding = self.model.embed_scenarios(x_scen).detach().numpy().reshape(-1)

        # Normalize the scenario embedding
        normalized_embedding = []
        for scenario in scenario_embedding:
            norm_scen = self.custom_min_max_normalize(scenario)
            normalized_embedding.append(norm_scen)

        scenario_embedding = np.array(normalized_embedding)

        return scenario_embedding

    def approximate(self, n_scenarios, gap=0.02, time_limit=600, threads=1, log_dir=None, test_set="0"):
        """ Formulates and solves the surrogate problem.  """

        def callback(model, where):
            """ Callback function to log time, bounds, and first stage sol. """
            if where == gp.GRB.Callback.MIPSOL:
                solving_results['time'].append(model.cbGet(gp.GRB.Callback.RUNTIME))
                solving_results['primal'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
                solving_results['dual'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
                solving_results['incumbent'].append(model.cbGetSolution(first_stage_vars))

        # total time for embedding+solving
        total_time = time.time()

        # get first stage mip and scenario embedding
        master_mip = self.get_master_mip()
        first_stage_vars = self.get_first_stage_variables(master_mip)
        scenario_embedding = self.get_scenario_embedding(n_scenarios, test_set)

        # initialize embedding model
        approximator_mip = self.mipper(master_mip,
                                       first_stage_vars,
                                       self.model,
                                       scenario_embedding).get_mip()
        solving_results = {'time': [], 'primal': [], 'dual': [], 'incumbent': []}

        approximator_mip.setParam('TimeLimit', time_limit)
        approximator_mip.setParam('MipGap', gap)
        approximator_mip.setParam('Threads', threads)
        approximator_mip.setParam('LogFile', log_dir)
        # optimize
        approximator_mip.optimize(callback)
        total_time = time.time() - total_time

        solving_results['incumbent'] = [dict(x) for x in solving_results['incumbent']]
        # Calculate the results here
        try:
            first_stage_sol = self.get_first_stage_solution(approximator_mip)
        except:
            first_stage_sol = None

        results = {
            'time': total_time,
            'predicted_obj': approximator_mip.objVal,
            'sol': first_stage_sol,
            'solving_results': solving_results,
            'solving_time': approximator_mip.Runtime
        }

        return results

    def get_first_stage_solution(self, model):
        """ Recovers the first stage solution. """
        x_sol = {}
        for var in model.getVars():
            if "x_in" in var.varName:
                idx = int(var.varName.split('[')[-1][:-1])
                x_sol[f"y_u{idx}"] = np.abs(var.x)
        return x_sol

    def get_first_stage_variables(self, mip):
        return {k: mip.getVarByName(f'x_in[{k}]')
                for k in range(self.inst['n_bases'])}

    def custom_min_max_normalize(self, x):
        """
        对x进行归一化，后720维按照指定的范围进行Min-Max归一化。
        - 如果值在27到31之间，按照[27, 31]归一化
        - 如果值在-96到-90之间，按照[-96, -90]归一化
        """
        # 如果 x 是 numpy array 或标量，转换为 tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, np.float32) or isinstance(x, float):  # 处理标量
            x = torch.tensor(x, dtype=torch.float32)

        # 定义归一化的范围
        range_27_31 = (27, 31)
        range_neg96_neg90 = (-96, -90)

        # 创建一个新的 x，保持原有数据结构
        normalized_x = x.clone()

        # 计算 27 到 31 范围内的掩码
        mask_27_31 = (x >= range_27_31[0]) & (x <= range_27_31[1])

        # 计算 -96 到 -90 范围内的掩码
        mask_neg96_neg90 = (x >= range_neg96_neg90[0]) & (x <= range_neg96_neg90[1])

        # 对于在 27 到 31 范围内的值进行归一化
        normalized_x[mask_27_31] = (x[mask_27_31] - range_27_31[0]) / (range_27_31[1] - range_27_31[0])

        # 对于在 -96 到 -90 范围内的值进行归一化
        normalized_x[mask_neg96_neg90] = (x[mask_neg96_neg90] - range_neg96_neg90[0]) / (
                    range_neg96_neg90[1] - range_neg96_neg90[0])

        # 对于不在这两个范围内的值，可以保持原值，或者你也可以选择其他方式处理
        # normalized_x[~mask_27_31 & ~mask_neg96_neg90] = x[~mask_27_31 & ~mask_neg96_neg90]  # 保持原值

        return normalized_x
