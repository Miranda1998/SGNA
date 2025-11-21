# src/model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)


def _chi2_quantile_df2(alpha: float) -> float:
    # df=2 时，CDF(x)=1-exp(-x/2) → x = -2 ln(1-alpha)
    return float(-2.0 * math.log(max(1e-12, 1.0 - alpha)))


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    tau_z = 1.5

    return mu + eps * (std * tau_z)


def kl_normal(mu_q, logvar_q, mu_p, logvar_p):
    LOGVAR_MIN = math.log(1e-5)
    LOGVAR_MAX = math.log(1e2)
    logvar_q = logvar_q.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
    logvar_p = logvar_p.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return kl.sum(dim=1).mean()


def nll_gaussian(y, mu, logvar):
    LOGVAR_MIN = math.log(1e-2)
    LOGVAR_MAX = math.log(1e2)
    logvar = logvar.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
    nll = 0.5 * (LOG_2PI + logvar + (y - mu) ** 2 / torch.exp(logvar))
    return nll.sum(dim=(1, 2)).mean()



@torch.no_grad()
def empirical_coverage(trajs, y_true, alpha=0.9):
    # trajs: [K,B,T,2], y_true: [B,T,2]
    mu = trajs.mean(dim=0)             # [B,T,2]
    Xc = trajs - mu.unsqueeze(0)       # [K,B,T,2]
    # 仅用均方对角方差作为简化
    var = (Xc.pow(2).mean(dim=0) + 1e-6)  # [B,T,2]
    d2 = ((y_true - mu)**2 / var).sum(-1)  # [B,T]
    thr = -2.0 * math.log(1.0 - alpha)     # df=2 卡方分位
    return (d2 <= thr).float().mean()

def nll_with_cov_regularizer(model, x, y, stat,
                             raw_logvar, mu,
                             lam_cov=1e-2, alpha=0.9, K=10):
    # 先算 NLL（用上面的 bounded 版本）
    nll = nll_gaussian(y, mu, raw_logvar)

    # 然后用少量采样做覆盖率
    with torch.no_grad():
        trajs, *_ = model.sample(x, stat, K=K)  # [K,B,T,2]
        cov = empirical_coverage(trajs, y)
    # 惩罚“低于目标覆盖率”的程度
    pen_cov = torch.relu(torch.tensor(alpha, device=x.device) - torch.tensor(cov, device=x.device))
    return nll + lam_cov * pen_cov


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, act=nn.ReLU, num_layers=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d, hidden), act()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim=128, out_dim=128, num_layers=1, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hid_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hid_dim * mult, out_dim)
        self.out_dim = out_dim

    def forward(self, seq):
        _, h = self.gru(seq)
        h = h[-1]
        return self.fc(h)


class PriorNet(nn.Module):
    def __init__(self, in_dim, z_dim, hidden=128):
        super().__init__()
        self.mlp = MLP(in_dim, z_dim * 2, hidden=hidden)

    def forward(self, cond):
        o = self.mlp(cond)
        mu, logvar = torch.chunk(o, 2, dim=-1)
        return mu, logvar


class PosteriorNet(nn.Module):
    def __init__(self, in_dim, z_dim, hidden=128):
        super().__init__()
        self.mlp = MLP(in_dim, z_dim * 2, hidden=hidden)

    def forward(self, cond):
        o = self.mlp(cond)
        mu, logvar = torch.chunk(o, 2, dim=-1)
        return mu, logvar

#==================Heads with shared trunk==================#
class SharedTrunkTwoHeads(nn.Module):
    def __init__(self, hid_dim, out_dim, width=256, depth=2, dropout=0.1, sigma0=1e-2):
        super().__init__()
        layers = []
        in_dim = hid_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, width), nn.GELU(), nn.LayerNorm(width)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            in_dim = width
        self.trunk = nn.Sequential(*layers)

        self.mu_head = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, out_dim)
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, out_dim)
        )

        self.reset_parameters(sigma0)

    def reset_parameters(self, sigma0: float = 1e-2):
        # 用 no_grad 包裹所有原地赋值 / 初始化
        with torch.no_grad():
            # 干路/中间层：用更通用的初始化
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            # 最后一层（输出层）做你原先的“特化初始化”
            mu_out = self.mu_head[-1]         # nn.Linear
            logvar_out = self.logvar_head[-1] # nn.Linear

            nn.init.normal_(mu_out.weight, mean=0.0, std=1e-3)
            nn.init.constant_(mu_out.bias, 0.0)

            # logvar：权重置 0，bias 设为 log(sigma0^2)
            nn.init.constant_(logvar_out.weight, 0.0)
            if logvar_out.bias is not None:
                logvar_out.bias.fill_(math.log(max(1e-6, sigma0 ** 2)))

    def forward(self, h):
        x = self.trunk(h)
        mu = self.mu_head(x)
        logvar = torch.clamp(self.logvar_head(x), -10, 10)
        return mu, logvar


#==================Heads with ResidualHeads==================#
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

class ResidualHeads(nn.Module):
    def __init__(self, hid_dim, out_dim, width=256, blocks=3, sigma0=1e-2):
        super().__init__()
        self.proj = nn.Linear(hid_dim, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, width * 2) for _ in range(blocks)])
        self.mu_head = nn.Linear(width, out_dim)
        self.logvar_head = nn.Linear(width, out_dim)
        self.reset_parameters(sigma0)

    def reset_parameters(self, sigma0=1e-2):
        with torch.no_grad():
            # 通用初始化
            nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
            for m in self.blocks.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            # 输出层：复刻你原始线性头的初始化
            nn.init.normal_(self.mu_head.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.mu_head.bias, 0.0)
            nn.init.constant_(self.logvar_head.weight, 0.0)
            self.logvar_head.bias.fill_(math.log(max(1e-6, sigma0 ** 2)))

    def forward(self, h):  # h: [B,H] 或 [B,T,H]
        x = self.proj(h)
        x = self.blocks(x)
        mu = self.mu_head(x)
        logvar = torch.clamp(self.logvar_head(x), -10, 10)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cond_dim, out_dim=2, hid_dim=128, num_layers=1, horizon=72,
                 max_step=3.5, sigma0=0.7, z_dim=16, mdn_K=5):
        """
        无 teacher forcing 的自回归解码器。
        现在每一步输入 = [上一时刻预测速度(Δ), z]，让 z 持续影响整段预测。
        """
        super().__init__()
        self.horizon = horizon
        self.max_step = max_step
        self.z_dim = z_dim

        # 条件特征 -> 初始 hidden
        self.init = nn.Linear(cond_dim, hid_dim)

        # 自回归 RNN：输入从原来的 out_dim 变为 out_dim + z_dim
        self.rnn = nn.GRU(input_size=out_dim + z_dim, hidden_size=hid_dim,
                          num_layers=num_layers, batch_first=True)

        # 残差输出头
        self.head = ResidualHeads(hid_dim, out_dim, sigma0=sigma0)

        # 可学习初始速度
        self.v0 = nn.Parameter(torch.zeros(out_dim))

    def forward(self, cond, z, last_pos=None):
        """
        cond: [B, cond_dim]
        z   : [B, z_dim]
        last_pos: [B, 2]  历史最后一个绝对位置（用于把累计Δ还原为绝对坐标）
        """
        if last_pos is None:
            raise ValueError("必须提供 last_pos 才能把相对位移累加成绝对坐标！")

        B = cond.size(0)
        h = torch.tanh(self.init(cond)).unsqueeze(0)  # [1, B, hid_dim]

        mu_list, logvar_list, delta_list = [], [], []

        # 初始位置和初始速度
        pos_t = last_pos  # [B,2]
        v_prev = self.v0.unsqueeze(0).expand(B, -1)  # [B,2]

        VAR_MIN, VAR_MAX = 1e-2, 10.0

        # 预备一个每步都复用的 z（常数注入）
        z_step = z  # [B, z_dim]

        for t in range(self.horizon):
            # 当前步输入：上一步速度 + z
            inp_t = torch.cat([v_prev, z_step], dim=-1)  # [B, 2 + z_dim]
            out_t, h = self.rnn(inp_t.unsqueeze(1), h)   # out_t: [B,1,hid]
            dec_out = out_t.squeeze(1)

            # 输出 Δμ 和 logvar
            # raw_delta = self.mu_head(dec_out)
            # raw_logvar = self.logvar_head(dec_out)

            raw_delta, raw_logvar = self.head(dec_out)

            # Δμ 经过 tanh 限幅（若你不想限幅，可改成：delta_mu_t = raw_delta）
            delta_mu_t = raw_delta

            # 方差约束（Softplus 保正、上限裁剪、再取 log）
            var_t = F.softplus(raw_logvar) + VAR_MIN
            var_t = torch.clamp(var_t, max=VAR_MAX)
            logvar_t = torch.log(var_t)

            # 更新位置（把相对位移累计成绝对坐标）
            pos_t = pos_t + delta_mu_t

            mu_list.append(pos_t)
            logvar_list.append(logvar_t)
            delta_list.append(delta_mu_t)

            # 下一步输入使用当前预测出来的速度（阻断梯度可稳定训练）
            v_prev = delta_mu_t.detach()

        # 拼回序列
        mu = torch.stack(mu_list, dim=1)         # [B, T, 2]
        logvar = torch.stack(logvar_list, dim=1) # [B, T, 2]
        delta_mu = torch.stack(delta_list, dim=1)# [B, T, 2]
        return mu, logvar, delta_mu


class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim=2, z_dim=16, hid_dim=128, horizon=72, stat_dim=0,
                 dec_max_step=3.5, dec_sigma0=0.7, lambda_cov=0.05):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.hid_dim = hid_dim
        self.horizon = horizon
        self.stat_dim = stat_dim

        self.enc_hist = Encoder(in_dim=x_dim, hid_dim=hid_dim, out_dim=hid_dim)
        self.enc_fut  = Encoder(in_dim=y_dim, hid_dim=hid_dim, out_dim=hid_dim)

        self.prior = PriorNet(self.enc_hist.out_dim + stat_dim, z_dim, hidden=hid_dim)
        self.post  = PosteriorNet(self.enc_hist.out_dim + self.enc_fut.out_dim + stat_dim, z_dim, hidden=hid_dim)

        cond_dim = self.enc_hist.out_dim + z_dim + stat_dim
        # 关键：把 z_dim 传给解码器，并在 RNN 每步拼接 z
        self.dec = Decoder(cond_dim=cond_dim, out_dim=y_dim, hid_dim=hid_dim, horizon=horizon,
                           max_step=dec_max_step, sigma0=dec_sigma0, z_dim=z_dim)

        self.teacher_forcing_ratio = 0.0
        self.lambda_cov = float(lambda_cov)


    def _soft_coverage_loss_prior_samples(self, h_enc, stat, last_pos, y_fut,
                                          mu_p, logvar_p,
                                          K=8, alpha=0.9, tau=0.5, eps=1e-6):
        """
        用 '先验采样K条' 构造经验均值/协方差（每步一个2D高斯），
        计算 GT 的软覆盖率，并取 (1 - mean_coverage) 作为损失。
        - h_enc: [B,H] 编码后的历史
        - stat:  [B,S]
        - last_pos: [B,2]
        - y_fut: [B,T,2]  (标准化空间)
        - mu_p, logvar_p: 先验参数 [B,Z]
        返回: loss_cov (scalar), soft_cov_mean (float)
        """
        B, T = y_fut.shape[:2]
        if stat is None:
            stat = y_fut.new_zeros(B, 0)

        # 1) 先验采样K条 + 无teacher解码 → [K,B,T,2]
        trajs = []
        for _ in range(K):
            z = reparameterize(mu_p, logvar_p)  # [B,Z]
            cond = torch.cat([h_enc, z, stat], dim=1)  # [B,H+Z+S]
            mu_k, _, _ = self.dec(cond, z, last_pos=last_pos)  # 不喂 teacher
            trajs.append(mu_k)
        trajs = torch.stack(trajs, dim=0)  # [K,B,T,2]

        # 2) 经验均值与协方差（每个 batch、每个时间步一个2x2协方差）
        mu_bar = trajs.mean(dim=0)  # [B,T,2]
        Xc = trajs - mu_bar.unsqueeze(0)  # [K,B,T,2]
        denom = max(K - 1, 1)
        # cov[bt] = sum_k Xc[k,bt,:] @ Xc[k,bt,:]^T / (K-1)
        cov = torch.einsum('kbti,kbtj->btij', Xc, Xc) / denom  # [B,T,2,2]
        # 稳定性：加 eps I
        eye = torch.eye(2, device=y_fut.device).view(1, 1, 2, 2)
        cov = cov + eps * eye

        # 3) Mahalanobis d^2，并用卡方阈值做软覆盖
        diff = (y_fut - mu_bar).unsqueeze(-1)  # [B,T,2,1]
        # 用 cholesky 更稳：d = L^{-1} diff；d2 = ||d||^2
        L = torch.linalg.cholesky(cov)  # [B,T,2,2]
        sol = torch.cholesky_solve(diff, L)  # [B,T,2,1]
        d2 = torch.matmul(diff.transpose(-2, -1), sol).squeeze(-1).squeeze(-1)  # [B,T]
        thr = _chi2_quantile_df2(alpha)  # 标量阈值
        # 软覆盖：I[d2<=thr] 的平滑近似 → sigmoid((thr - d2)/tau)
        soft_cov = torch.sigmoid((thr - d2) / max(1e-6, tau))  # [B,T]
        soft_cov_mean = soft_cov.mean()  # 标量

        # 损失：希望覆盖高 → 最小化 (1 - mean_cov)
        loss_cov = (1.0 - soft_cov_mean)
        return loss_cov, soft_cov_mean

    def forward(self, x_hist, y_fut, stat=None, beta=1.0):
        B = x_hist.size(0)
        if stat is None:
            stat = x_hist.new_zeros(B, 0)

        h_enc = self.enc_hist(x_hist)
        h_fut = self.enc_fut(y_fut)

        mu_p, logvar_p = self.prior(torch.cat([h_enc, stat], dim=1))
        mu_q, logvar_q = self.post(torch.cat([h_enc, h_fut, stat], dim=1))

        z = reparameterize(mu_q, logvar_q)
        cond = torch.cat([h_enc, z, stat], dim=1)

        last_pos = x_hist[:, -1, :2]
        # 关键：把 z 传入解码器
        mu, logvar, delta_mu = self.dec(cond, z, last_pos=last_pos)

        # === 基础重构 ===
        nll = nll_gaussian(y_fut, mu, logvar)
        kl  = kl_normal(mu_q, logvar_q, mu_p, logvar_p)

        # === 辅助 MSE（可保留原权重）===
        mse_pos = F.mse_loss(mu, y_fut[..., :2], reduction="mean")
        lambda_mse = 100.0 * mse_pos

        # === 覆盖率校准正则 ===
        var = torch.exp(logvar).clamp_min(1e-12)
        d2 = ((y_fut - mu) ** 2 / (var + 1e-3)).sum(dim=-1)  # [B,T]
        calib_loss = (d2.mean() - 2.0) ** 2
        cov_reg = self.lambda_cov * calib_loss

        # ---- 新增：覆盖率辅助损失（对齐“sample K条”的可视化评估）----
        loss_cov, cov_mean = self._soft_coverage_loss_prior_samples(
            h_enc, stat, last_pos, y_fut, mu_p, logvar_p,
            alpha=0.9,  # 90%椭圆
            tau=0.5,  # 平滑温度，越小越像硬阈值
            eps=1e-6
        )


        loss = nll/200 + kl + lambda_mse + cov_reg + loss_cov

        return {
            "loss": loss,
            "lambda_mse": lambda_mse,
            "loss_cov": loss_cov,
            "cov_reg": cov_reg,
            "nll":  nll.detach()/200,
            "kl":   kl.detach(),
            "mu":   mu.detach(),
            "logvar": logvar.detach(),
            "delta_mu": delta_mu,
        }

    @torch.no_grad()
    def sample(self, x_hist, stat=None, K=10):
        self.eval()
        B = x_hist.size(0)
        if stat is None:
            stat = x_hist.new_zeros(B, 0)

        h_enc = self.enc_hist(x_hist)
        mu_p, logvar_p = self.prior(torch.cat([h_enc, stat], dim=1))
        last_pos = x_hist[:, -1, :2]

        trajs, prior_lp, mus, logvars = [], [], [], []
        for _ in range(K):
            z = reparameterize(mu_p, logvar_p)
            cond = torch.cat([h_enc, z, stat], dim=1)
            # 关键：采样时也把 z 传入解码器
            mu, logvar, _ = self.dec(cond, z, last_pos=last_pos)
            y_sample = mu
            lp = -0.5 * (((z - mu_p) ** 2 / torch.exp(logvar_p)) + logvar_p + LOG_2PI).sum(dim=1)

            trajs.append(y_sample); mus.append(mu); logvars.append(logvar); prior_lp.append(lp)

        trajs = torch.stack(trajs, dim=0)
        mus = torch.stack(mus, dim=0)
        logvars = torch.stack(logvars, dim=0)
        prior_logprob = torch.stack(prior_lp, dim=0)
        return trajs, prior_logprob, mus, logvars


