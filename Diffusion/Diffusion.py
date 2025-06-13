import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 这里是对loss的定义
class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    # 预测值，目标值，权重
    def forward(self, pred, targ, weighted=1.0):
        loss = self._loss(pred, targ)
        WeightedLoss = (loss * weighted).mean()
        return WeightedLoss


# 这里是L1loss
class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {"l1": WeightedL1, "l2": WeightedL2}


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# 完成对位置的编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        # 这段实现的就是位置编码 ， 最终得到的emb，前半部分是sin，后半部分是cos
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# 整个DDPM中所需要的网络
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim=16):
        super(MLP, self).__init__()
        self.t_dim = t_dim
        self.a_dim = action_dim
        self.device = device

        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),  # 在Diffusion中用这个作为激活函数
            nn.Linear(t_dim * 2, t_dim),
        )
        # 中间层
        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        #输出层
        self.final_layer = nn.Linear(hidden_dim, action_dim)

        # 对神经网络的参数进行初始化,有些初始化能够提升模型效果
        self.init_weights()

    # 初始化函数
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, time, state):
        t_emb = self.time_mlp(time)
        x = torch.cat([x, state, t_emb], dim=1)

        x = self.mid_layer(x)
        return self.final_layer(x)


class Diffusion(nn.Module):
    # 参数：采用那种loss，对时间的离散方式是怎眼的（线性或者类似余弦的其他方式）
    def __init__(
        self,
        loss_type,
        #  state_dim,
        beta_schedule="linear",
        clip_denoised=True,
        predict_epsilon=True,
        **kwargs,
    ):
        super(Diffusion, self).__init__()
        self.state_dim = kwargs["obs_dim"]
        self.action_dim = kwargs["act_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.predict_epsilon = predict_epsilon
        # 反传多少步
        self.T = kwargs["T"]
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.device = torch.device(kwargs["device"])
        self.model = MLP(
            self.state_dim, self.action_dim, self.hidden_dim, self.device
        ).to(kwargs["device"])

        # 对时间的离散化
        if beta_schedule == "linear":
            # 这里的betas表示beta[t]，
            # 表示beta在一开始采用多大的值，是等差数列（算数序列），这个在EDM中讨论过
            betas = torch.linspace(
                0.0001, 0.02, self.T, dtype=torch.float32, device=self.device
            )
        alphas = 1.0 - betas

        # 这个cumprod命令表示将a[i] 直接换成前缀积 （eg：[1,3,5]->[1,3,15]
        # 最终希望得到的是公式中的 \bar{alpha_t}
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        # 这个prev表示----?
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=kwargs["device"]), alphas_cumprod[:-1]]
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # 前向过程 根号 alpha x_0 +根号 （1-alpha) eps
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # 反向过程  这里复现就是花大时间讨论导出其 \miu \Sigma的过程,
        # 首先是方差
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

        # 其次是x_0   在已知x_t的情况下，一步求出x_0
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )

        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        # 到此为止前向和反向的过程都已经完成了

        # 接下来是Loss
        self.loss_fn = Losses[loss_type]()

    def q_posterior(self, x_start, x, t):
        # 这里是公式中的 \miu
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(
            self.posterior_log_variance_clipped, t, x.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x, t, pred_noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipm_alphas_cumprod, t, x.shape) * pred_noise
        )

    def p_mean_variance(self, x, t, s):
        # 先拿到模型预测的噪声
        pred_noise = self.model(x, t, s)
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        x_recon.clamp_(-1, 1)

        # 在已知x_t x_0情况下，求出x_{t-1}
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_recon, x, t
        )
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, model_log_variance = self.p_mean_variance(x, t, s)
        # 公式里最后还加了一个噪声，这里的噪声和去噪过程的形状是一样的
        noise = torch.randn_like(x)
        # 最后一个扩散步不需要加noise
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, *args, **kwargs):
        device = self.device
        batch_size = state.shape[0]
        # 初始化噪声
        x = torch.randn(shape, device=device, requires_grad=False)

        # 从T-1开始，一直到0
        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, state)
        return x

    # 接下来是采样过程

    def sample(self, state, *args, **kwargs):
        """
        state:[batch_size,state_dim]
        """
        batch_size = state.shape[0]
        # 初始化噪声，第一位是batch_size,第二位是action_dim
        shape = [batch_size, self.action_dim]

        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-1.0, 1.0)

    # ---------Training--------- 这里复现的就是文章中的Training部分
    """
            Loss的计算过程 预测噪声和标签噪声做差，然后做个一范式或者二范数
            就是论文中grad_\theta ||eps-eps_\theta||_2^2
            其中eps_\theta是模型预测的噪声,参数为 x_0和eps的加权
        """

    # x_start表示x_0
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # 这里是前向过程的汇总步
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        # 这个是sample的噪声 也就是eps
        x_noisy = self.q_sample(x_start, t, noise)
        # 这里是从神经网络预测的噪声 eps_\theta
        x_recon = self.model(x_noisy, t, state)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    # x表示x_0   state表示x_t
    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)


if __name__ == "__main__":
    device = "cpu"
    x = torch.randn(256, 2).to(device)
    state = torch.randn(256, 11).to(device)
    model = Diffusion(
        loss_type="l2", obs_dim=11, act_dim=2, hidden_dim=256, T=100, device=device
    )
    action = model(state)

    loss = model.loss(x, state)

    print(f"action:{action};loss:{loss.item()}")
