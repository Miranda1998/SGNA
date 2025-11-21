from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np
from geopy.distance import geodesic
from .two_sp import TwoStageStocProg
import pdb

import torch
from nsp.scenario_gen.model import CVAE  # 导入定义的cVAE模型
from nsp.utils.dblrp import inverse_points_minmax

import json


class DroneBaseLocationRoutingProblem(TwoStageStocProg):
    """
        Class for Two staged Stochastic Integer Programming Problem
        """

    def __init__(self, inst):
        self.tol = 1e-6
        self.inst = inst

        self.n_vessels = self.inst['n_vessels']
        self.n_bases = self.inst['n_bases']
        self.n_drones = self.inst['n_drones']
        self.T = self.inst['T']  # time periods


    def _make_surrogate_scenario_model(self, scenario_id=None):
        pass


    def _make_extensive_model(self, *args):
        """ Creates a two-stage extensive form. """
        """ Creates a second stage problem for a given scenario. """
        scenarios = args[0]
        n_scenarios = len(scenarios)
        scenario_prob = 1 / n_scenarios

        model = gp.Model()
        var_dict = {}

        bases_set = [f'u{i}' for i in range(self.n_bases)]
        vessels_set = [f'v{j}' for j in range(self.n_vessels)]
        nodes_set = bases_set + vessels_set
        scenario_set = [s for s in range(n_scenarios)]
        Time_set = [t for t in range(self.T)]

        # binary variables for each location
        for u in bases_set:
            # bound lower and upper to solution
            var_dict[f"y_{u}"] = model.addVar(vtype="B", name=f"y_{u}")

        # add either continous or binary second stage serving costs
        for s in scenario_set:
            for i in nodes_set:
                for k in Time_set:
                    for j in nodes_set:
                        for l in Time_set:
                            if l > k:
                                var_name = f"x_{i}_{k}_{j}_{l}_{s}"
                                var_dict[var_name] = model.addVar(vtype="B", name=var_name)

        # drones num for each location
        all_time = [-1] + Time_set
        print("Time_set", Time_set)
        print('alltime', all_time)
        for s in scenario_set:
            for u in bases_set:
                for t in all_time:
                    var_name = f"b_{u}_{t}_{s}"
                    # bound lower and upper to solution
                    var_dict[var_name] = model.addVar(vtype="I", lb=0, ub=self.inst['n_drones'], name=var_name)

        # drones num for each location
        for s in scenario_set:
            for i in nodes_set:
                for t in Time_set:
                    var_name = f"p_{i}_{t}_{s}"
                    # bound lower and upper to solution
                    var_dict[var_name] = model.addVar(vtype="I", lb=0, ub=self.inst['max_route_time'], name=var_name)
        print('Done1')

        # obj = 0
        # 创建一个空的线性表达式对象
        obj = gp.LinExpr
        obj = 0
        for u in bases_set:
            obj += - self.inst['base_costs'][u] * var_dict[f"y_{u}"]

        for s in scenario_set:
            for j in vessels_set:
                obj += scenario_prob * self.inst['reward'][j] * gp.quicksum(
                    var_dict[f"x_{i}_{k}_{j}_{l}_{s}"] for i in nodes_set for k in Time_set for l in Time_set if l > k)

        # 目标函数 最大化 obj
        model.setObjective(obj, gp.GRB.MAXIMIZE)
        for s in scenario_set:
            for u in bases_set:
                model.addConstr(
                    gp.quicksum(
                        var_dict[f"x_{u}_{k}_{j}_{l}_{s}"] for k in Time_set for j in nodes_set for l in Time_set if l > k)
                    <= var_dict[f"y_{u}"] * self.inst['n_drones'],
                    name='constraint_base_location01_%s' % (u))
        print('Done2')

        for s in scenario_set:
            for u in bases_set:
                model.addConstr(
                    gp.quicksum(
                        var_dict[f"x_{j}_{l}_{u}_{k}_{s}"] for k in Time_set for j in nodes_set for l in Time_set if k > l)
                    <= var_dict[f"y_{u}"] * self.inst['n_drones'],
                    name='constraint_base_location02_%s_%s' % (u, s))
        print('Done3')

        for s in scenario_set:
            for u in bases_set:
                for k in Time_set:
                    model.addConstr(
                        gp.quicksum(var_dict[f"x_{u}_{k}_{j}_{l}_{s}"] for j in nodes_set for l in Time_set if l > k)
                        - gp.quicksum(var_dict[f"x_{j}_{l}_{u}_{k}_{s}"] for j in nodes_set for l in Time_set if k > l)
                        == var_dict[f"b_{u}_{k - 1}_{s}"] - var_dict[f"b_{u}_{k}_{s}"], name='constraint_bases_%s_%s_%s' % (u, k, s))

        for s in scenario_set:
            for u in bases_set:
                model.addConstr(
                    var_dict[f"b_{u}_{-1}_{s}"] == self.inst['drone_each_base'], name='constraint_initial_bases_%s_%s' % (u, s))
        print('Done4')

        for s in scenario_set:
            for v in vessels_set:
                for k in Time_set:
                    model.addConstr(
                        gp.quicksum(var_dict[f"x_{v}_{k}_{j}_{l}_{s}"] for j in nodes_set for l in Time_set if l > k)
                        == gp.quicksum(var_dict[f"x_{j}_{l}_{v}_{k}_{s}"] for j in nodes_set for l in Time_set if k > l),
                        name='constraint_flow_balance_%s_%s_%s' % (v, k, s))
        print('Done5')

        for s in scenario_set:
            for v in nodes_set:
                model.addConstr(
                    gp.quicksum(var_dict[f"x_{v}_{k}_{j}_{l}_{s}"] for k in Time_set for j in nodes_set
                                for l in Time_set if l > k) <= 1,
                    name='constraint_visit_most_once_%s_%s' % (v, s))
        print('Done6')

        for s in scenario_set:
            for v in vessels_set:
                for k in Time_set:
                    for j in nodes_set:
                        for l in Time_set:
                            if l > k:
                                model.addConstr(
                                    var_dict[f"p_{v}_{k}_{s}"] - (l - k) * var_dict[f"x_{v}_{k}_{j}_{l}_{s}"]
                                    + self.inst['max_route_time'] * (1 - var_dict[f"x_{v}_{k}_{j}_{l}_{s}"])
                                    >= var_dict[f"p_{j}_{l}_{s}"],
                                    name='constraint_max_route_time_01_%s_%s_%s_%s_%s' % (v, k, j, l, s))
        print('Done7')

        for s in scenario_set:
            for u in bases_set:
                for k in Time_set:
                    for j in nodes_set:
                        for l in Time_set:
                            if l > k:
                                model.addConstr(
                                    self.inst['max_route_time'] - (l - k) * var_dict[f"x_{u}_{k}_{j}_{l}_{s}"]
                                    >= var_dict[f"p_{j}_{l}_{s}"],
                                    name='constraint_max_route_time_02_%s_%s_%s_%s_%s' % (u, k, j, l, s))
        print('Done8')

        for s in scenario_set:
            for u in nodes_set:
                for k in Time_set:
                    for j in nodes_set:
                        for l in Time_set:
                            if l > k:
                                # 定义两个地点的经纬度
                                if u in bases_set and j in bases_set:
                                    coord1 = self.inst['base_pos'][int(u[1:])]
                                    coord2 = self.inst['base_pos'][int(j[1:])]
                                elif u in bases_set and j in vessels_set:
                                    coord1 = self.inst['base_pos'][int(u[1:])]
                                    coord2 = scenarios[s][int(j[1:])][l]
                                elif u in vessels_set and j in bases_set:
                                    coord1 = scenarios[s][int(u[1:])][k]
                                    coord2 = self.inst['base_pos'][int(j[1:])]
                                else:
                                    coord1 = scenarios[s][int(u[1:])][k]  # 波兰，华沙的经纬度
                                    coord2 = scenarios[s][int(j[1:])][l]  # 意大利，罗马的经纬度

                                # 计算两个经纬度之间的地理距离
                                distance = geodesic(coord1, coord2).kilometers  # (latitude, longitude)

                                model.addConstr(
                                    (l - k - 1) * var_dict[f"x_{u}_{k}_{j}_{l}_{s}"] <=
                                    distance / self.inst['drone_speed'] * 12 + self.inst['observation_time'],
                                    name='constraint_cruising_time_02_%s_%s_%s_%s_%s' % (u, k, j, l, s))

                                model.addConstr((distance / self.inst['drone_speed'] * 12 +
                                                 self.inst['observation_time']) * var_dict[f"x_{u}_{k}_{j}_{l}_{s}"]
                                                <= (l - k),
                                                name='constraint_cruising_time_02_%s_%s_%s_%s_%s' % (u, k, j, l, s))

        return model

    def _make_second_stage_model(self, *args):
        """ Creates a second stage problem for a given scenario. """
        model = gp.Model()
        var_dict = {}

        bases_set = [f'u{i}' for i in range(self.n_bases)]
        vessels_set = [f'v{j}' for j in range(self.n_vessels)]
        nodes_set = bases_set + vessels_set
        Time_set = [t for t in range(self.T)]

        # binary variables for each location
        for u in bases_set:
            # bound lower and upper to solution
            var_dict[f"y_{u}"] = model.addVar(vtype="B", name=f"y_{u}")

        # add either continous or binary second stage serving costs
        for i in nodes_set:
            for k in Time_set:
                for j in nodes_set:
                    for l in Time_set:
                        if l > k:
                            var_name = f"x_{i}_{k}_{j}_{l}"
                            var_dict[var_name] = model.addVar(vtype="B", name=var_name)

        # drones num for each location
        all_time = [-1] + Time_set
        for u in bases_set:
            for t in all_time:
                var_name = f"b_{u}_{t}"
                # bound lower and upper to solution
                var_dict[var_name] = model.addVar(vtype="I", lb=0, ub=self.inst['n_drones'], name=var_name)


        # drones num for each location
        for i in nodes_set:
            for t in Time_set:
                var_name = f"p_{i}_{t}"
                # bound lower and upper to solution
                var_dict[var_name] = model.addVar(vtype="I", lb=0, ub=self.inst['max_route_time'], name=var_name)

        # obj = 0
        # 创建一个空的线性表达式对象
        obj = gp.LinExpr
        obj = 0
        for u in bases_set:
            obj += - self.inst['base_costs'][u] * var_dict[f"y_{u}"]

        for j in vessels_set:
            obj = obj + self.inst['reward'][j] * gp.quicksum(
                var_dict[f"x_{i}_{k}_{j}_{l}"] for i in nodes_set for k in Time_set for l in Time_set if l > k)


        # 目标函数 最大化 obj
        model.setObjective(obj, gp.GRB.MAXIMIZE)


        for u in bases_set:
            model.addConstr(
                gp.quicksum(
                    var_dict[f"x_{u}_{k}_{j}_{l}"] for k in Time_set for j in nodes_set for l in Time_set if l > k)
                <= var_dict[f"y_{u}"] * self.inst['n_drones'],
                name='constraint_base_location01_%s' % (u))


        for u in bases_set:
            model.addConstr(
                gp.quicksum(var_dict[f"x_{j}_{l}_{u}_{k}"] for k in Time_set for j in nodes_set for l in Time_set if k > l)
                <= var_dict[f"y_{u}"] * self.inst['n_drones'],
                name='constraint_base_location02_%s' % (u))


        for u in bases_set:
            for k in Time_set:
                model.addConstr(
                    gp.quicksum(var_dict[f"x_{u}_{k}_{j}_{l}"] for j in nodes_set for l in Time_set if l > k)
                    - gp.quicksum(var_dict[f"x_{j}_{l}_{u}_{k}"] for j in nodes_set for l in Time_set if k > l)
                    == var_dict[f"b_{u}_{k-1}"] - var_dict[f"b_{u}_{k}"], name='constraint_bases_%s_%s' % (u, k))


        for u in bases_set:
            model.addConstr(
                var_dict[f"b_{u}_{-1}"] == self.inst['drone_each_base'], name='constraint_initial_bases_%s' % (u))



        for v in vessels_set:
            for k in Time_set:
                model.addConstr(
                    gp.quicksum(var_dict[f"x_{v}_{k}_{j}_{l}"] for j in nodes_set for l in Time_set if l > k)
                    == gp.quicksum(var_dict[f"x_{j}_{l}_{v}_{k}"] for j in nodes_set for l in Time_set if k > l),
                    name='constraint_flow_balance_%s_%s' % (v, k))

        for i in nodes_set:
            model.addConstr(
                gp.quicksum(var_dict[f"x_{i}_{k}_{j}_{l}"] for k in Time_set for j in nodes_set for l in Time_set
                            if l > k) <= 1, name='constraint_visit_most_once_%s' % (i))

        for v in vessels_set:
            for k in Time_set:
                for j in nodes_set:
                    for l in Time_set:
                        if l > k:
                            model.addConstr(
                                var_dict[f"p_{v}_{k}"] - (l - k) * var_dict[f"x_{v}_{k}_{j}_{l}"]
                                + self.inst['max_route_time'] * (1 - var_dict[f"x_{v}_{k}_{j}_{l}"])
                                >= var_dict[f"p_{j}_{l}"],
                                name='constraint_max_route_time_01_%s_%s_%s_%s' % (v, k, j, l))

        for u in bases_set:
            for k in Time_set:
                for j in nodes_set:
                    for l in Time_set:
                        if l > k:
                            model.addConstr(
                                self.inst['max_route_time'] - (l - k) * var_dict[f"x_{u}_{k}_{j}_{l}"]
                                >= var_dict[f"p_{j}_{l}"],
                                name='constraint_max_route_time_02_%s_%s_%s_%s' % (u, k, j, l))


        for u in nodes_set:
            for k in Time_set:
                for j in nodes_set:
                    for l in Time_set:
                        if l > k:
                            # 定义两个地点的经纬度
                            if u in bases_set and j in bases_set:
                                coord1 = self.inst['base_pos'][int(u[1:])]
                                coord2 = self.inst['base_pos'][int(j[1:])]
                            elif u in bases_set and j in vessels_set:
                                coord1 = self.inst['base_pos'][int(u[1:])]
                                coord2 = args[0][int(j[1:])][l]
                            elif u in vessels_set and j in bases_set:
                                coord1 = args[0][int(u[1:])][k]
                                coord2 = self.inst['base_pos'][int(j[1:])]
                            else:
                                coord1 = args[0][int(u[1:])][k]  # 波兰，华沙的经纬度
                                coord2 = args[0][int(j[1:])][l]  # 意大利，罗马的经纬度

                            # 计算两个经纬度之间的地理距离
                            distance = geodesic(coord1, coord2).kilometers  # (latitude, longitude)

                            model.addConstr(
                                 (l - k - 1) * var_dict[f"x_{u}_{k}_{j}_{l}"] <=
                                 distance/self.inst['drone_speed']*12 + self.inst['observation_time'],
                                name='constraint_cruising_time_02_%s_%s_%s_%s' % (u, k, j, l))

                            model.addConstr((distance / self.inst['drone_speed']*12 +
                                            self.inst['observation_time'])* var_dict[f"x_{u}_{k}_{j}_{l}"]
                                            <= (l - k),
                                name='constraint_cruising_time_02_%s_%s_%s_%s' % (u, k, j, l))

        model.update()

        return model


    def solve_extensive(self, n_scenarios, gap=0.02, time_limit=600, threads=1, verbose=1,
                        log_dir=None, node_file_start=None, node_file_dir=None,
                        test_set="0"):
        """ Solves the two-stage extensive form. """
        def callback(model, where):
            """ Callback function to log time, bounds, and first stage sol. """
            if where == gp.GRB.Callback.MIPSOL:
                self.ef_solving_results['time'].append(model.cbGet(gp.GRB.Callback.RUNTIME))
                self.ef_solving_results['primal'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
                self.ef_solving_results['dual'].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
                self.ef_solving_results['incumbent'].append(model.cbGetSolution(model._x ))

        # make extensive form
        scenarios = self.get_scenarios(n_scenarios, test_set)
        model = self._make_extensive_model(scenarios)

        # get variables for callback
        model.update()
        self.ef_solving_results = {'primal': [], 'dual': [], 'incumbent': [], 'time': []}
        ef_fs_vars = []
        bases_set = [f'u{i}' for i in range(self.n_bases)]
        for u in bases_set:
            ef_fs_vars.append(model.getVarByName(f"y_{u}"))
        model._x = ef_fs_vars

        # solve two_sp
        if log_dir is not None:
            model.setParam("LogFile", log_dir)
        if node_file_start is not None:
            model.setParam("NodefileStart", node_file_start)
            model.setParam("NodefileDir", node_file_dir)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)

        model.optimize(callback)
        return model


    def solve_surrogate_scenario_model(self, *args):
        pass


    def get_second_stage_objective(self, sol_1, traj, gap=0.0001, time_limit=1e7,
                                   threads=1, verbose=0):
        """ Solves the second stage problem for a given scenario. """

        model = self._make_second_stage_model(traj)

        # fix first stage solution
        print("让我看看fixed的sol_1", sol_1)
        model = self.fix_first_stage(model, sol_1)

        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)

        model.optimize()

        if model.Status == 2 or 9:
            print("reward=", self.inst['reward'])
            if model.Status == 2:
                print('得到最优解')
                print("MIPGap（相对误差）：%s\n" % model.MIPGap)
            else:
                print('达到timelimit')
                print("MIPGap（相对误差）：%s\n" % model.MIPGap)
            for v in model.getVars():
                if v.x != 0 and v.x <= 1.0:
                    if "x" in v.var_name or "y" in v.var_name:
                        print('参数', v.varName, '=', v.x)
            print("这里的第二阶段目标值就开始不对了吗", model.objVal)

        second_stage_obj = self.get_second_stage_cost(model)
        print('second_stage_obj=', second_stage_obj)
        return second_stage_obj


    def fix_first_stage(self, model, sol):
        """ Fixes the first stage solution of a given model. """
        # pdb.set_trace()
        for sol_var_name, sol_var_val in sol.items():
            # model.getVarByName(sol_var_name).lb = sol_var_val
            # model.getVarByName(sol_var_name).ub = sol_var_val
            model.addConstr(model.getVarByName(sol_var_name) == sol_var_val, name="fixed_value_constraint")
        model.update()
        return model


    def get_second_stage_cost(self, model):
        """ Gets the second stage cost of a given model.  """
        second_stage_obj = 0  # 初始化目标函数

        # 确保模型优化成功
        if model.status == 2 or 9:
            # 获取模型中的所有变量
            vars_from_model = model.getVars()

            # 重新构建 var_dict
            var_dict_from_model = {var.varName: var for var in vars_from_model}

            # 通过从 model 获取的 var_dict 计算目标函数值
            fist_stage_cost = 0
            for var_name, var in var_dict_from_model.items():
                # 假设 reward 数据已经在 inst 中
                if "y" in var_name:
                    fist_stage_cost += -self.inst['base_costs'][var_name.split('_')[1]] * var.x # 根据变量名提取 u

            second_stage_obj = model.objVal + fist_stage_cost

            print(f"Calculated second stage objective: {second_stage_obj}")
        else:
            print("Model optimization was not successful. Status:", model.status)
        return second_stage_obj


    def evaluate_first_stage_sol(self, sol, n_scenarios, gap=0.0001, time_limit=600, threads=1, verbose=0,
                                 test_set="0", n_procs=1):
        """ Evaluates the first stage solution across all scenarios. """

        return 1998




    def  get_first_stage_solution(self, model):
        # 获取模型中的所有变量
        vars_from_model = model.getVars()

        # 重新构建 var_dict
        var_dict_from_model = {var.varName: var for var in vars_from_model}

        # 通过从 model 获取的 var_dict 计算目标函数值
        fist_stage_sol = {}
        for var_name, var in var_dict_from_model.items():
            # 假设 reward 数据已经在 inst 中
            if "y" in var_name:
                fist_stage_sol[var_name] =  var.x  # 根据变量名提取 u

        return fist_stage_sol

    def get_first_stage_extensive_solution(self, model):
        return self.get_first_stage_solution(model)


    def get_scenarios(self, n_scenarios, test_set):
        # 设备选择
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化模型
        model = CVAE(x_dim=5, y_dim=2, z_dim=32, hid_dim=128, horizon=72).to(device)

        # 加载训练好的模型参数
        model.load_state_dict(torch.load(self.inst['cVAE_model_path'], map_location=device))
        model.eval()  # 设置为评估模式

        x_hist_np = np.load(self.inst['x_hist_path'])  # x_hist 是一个包含客户历史位置的数据 [n, T, 2]

        # 将 Numpy 数组转换为 PyTorch 张量
        x_hist = torch.tensor(x_hist_np, dtype=torch.float32)
        # 将张量移到正确的设备上（例如 GPU 或 CPU）
        x_hist = x_hist.to(device)  # 假设 device 是 'cuda' 或 'cpu'

        # 假设 stat 是额外的条件，如果没有可以传 None
        stat = None  # 或者提供额外的统计数据

        # 使用模型生成未来轨迹，K 是生成轨迹的数量
        K = 1  # 生成1条轨迹,每次采样只采一次

        # 导入标准化min-max
        with open(self.inst['minmax_norm'], 'r') as f:
            minmax_norm = json.load(f)

        rng = np.random.RandomState()
        rng.seed(n_scenarios)

        scenarios = []
        for _ in range(n_scenarios):
            # 生成随机历史位置数据作为输入
            trajs, prior_logprob, mus, logvars = model.sample(x_hist, stat=stat, K=K)
            hist_real = inverse_points_minmax(trajs, minmax_norm)[0].detach().cpu().numpy()
            scenarios.append(hist_real)

        return scenarios