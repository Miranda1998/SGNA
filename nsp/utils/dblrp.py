from pathlib import Path
import torch
import math


def get_path(data_path, cfg, ptype="inst", suffix=".pkl", as_str=False):
    p = Path(data_path) / "dblrp"
    p.mkdir(parents=True, exist_ok=True)

    p = p / f"{ptype}_f{cfg.n_bases}_c{cfg.n_vessels}_" \
            f"r{cfg.ratio}_iss{1 if cfg.flag_integer_second_stage else 0}" \
            f"_bt{1 if cfg.flag_bound_tightening else 0}_nsp{cfg.n_samples_p}_nse{cfg.n_samples_e}_sd{cfg.seed}{suffix}"

    if as_str:
        return str(p)
    return p


def inverse_points_minmax(points_norm: torch.Tensor, minmax_norm: dict):
    """
    仅反归一化前两维（lat, lon）。
    points_norm: [B,T,2]（标准化空间: 0~1）
    返回: [B,T,2]（经纬度真实值）
    """
    lat_min = float(minmax_norm["lat_min"])
    lat_max = float(minmax_norm["lat_max"])
    lon_min = float(minmax_norm["lon_min"])
    lon_max = float(minmax_norm["lon_max"])

    # 防止除零
    lat_rng = max(lat_max - lat_min, 1e-9)
    lon_rng = max(lon_max - lon_min, 1e-9)

    lat = points_norm[..., 0] * lat_rng + lat_min
    lon = points_norm[..., 1] * lon_rng + lon_min
    return torch.stack([lat, lon], dim=-1)


def haversine(coord1, coord2):
    # 地球半径 (单位：千米)
    R = 6371.0

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # 将经纬度从度转为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # 计算经度和纬度的差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # 使用 haversine 公式计算两点之间的距离
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 计算距离
    distance = R * c
    return distance