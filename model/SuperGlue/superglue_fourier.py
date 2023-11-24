from typing import List, Tuple, Optional
from copy import deepcopy
from typing import List, Tuple
from torch import nn
from loss.descs_loss import *
from loss.val_auc import *
from utils import find_matches


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


# 傅里叶位置编码
class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None,
                 gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ encode position vector """
        # self.Wr.weight.data = self.Wr.weight.to(x.dtype)
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


# 旋转编码
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(
        freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5   # 除以dim的平方根，是为了缩放注意力分数
    prob = torch.nn.functional.softmax(scores, dim=-1)  # 获得注意力权重
    # 通过加权值向量的方式得到加权和，返回加权后的值向量和注意力权重
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    # 多头，增加模型的表达能力
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def _forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, encoding: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_dim = query.size(0)

        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]

        if encoding is not None:
            query = apply_cached_rotary_emb(encoding, query.permute(0, 2, 3, 1))
            key = apply_cached_rotary_emb(encoding, key.permute(0, 2, 3, 1))
            query, key = query.permute(0, 3, 1, 2), key.permute(0, 3, 1, 2)
        x, _ = attention(query, key, value)
        # .contiguous()是一个方法，用于确保张量在内存中的存储顺序是连续的
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

    def forward(self, query, key, value, encoding=None):
        return self._forward(query, key, value, encoding)


# 计算x和source之间的相关性.MLP用于对这些相关性进行非线性变换，得到一个更好的表示.
class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor, encoding: Optional[torch.Tensor] = None) -> torch.Tensor:

        message = self.attn(x, source, source, encoding)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor, encoding0=None, encoding1=None) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                encoding0, encoding1 = None, None
                src0, src1 = desc1, desc0
                delta0, delta1 = layer(desc0, src0, encoding0), layer(desc1, src1, encoding1)
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
                delta0, delta1 = layer(desc0, src0, encoding0), layer(desc1, src1, encoding1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


# 采用对数空间下的操作：为了提高数值计算的稳定性
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)  # 对u和v初始化，和目标分布log_mu和log_nu相同形状
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""

    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N

    return Z


def filter_matches(scores: torch.Tensor, th: float):
    """ obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = scores.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)

    valid0 = mutual0 & (mscores0 > th)

    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

    return indices0, indices1, mscores0, mscores1

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


# 搭建CraquelureNet+Attention网络
class SuperGlue(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 5,
        'num_heads': 4,
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
        'alpha': 1
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        head_dim = self.config['descriptor_dim'] // self.config['num_heads']
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)
        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, data, is_training=True):
        batch_size = len(data['data_pred']['keypoints'])
        all_losses = []
        all_aucs = []
        all_results = []

        for i in range(batch_size):
            refer_desc, query_desc = data['data_pred']['descriptors'][i]
            refer_kpt, query_kpt = data['data_pred']['keypoints'][i]
            refer_score, query_score = data['data_pred']['scores'][i]
            images = data['images'][i].unsqueeze(0)

            refer_desc1, query_desc1 = refer_desc.unsqueeze(0).double(), query_desc.unsqueeze(0).double()
            refer_kpt, query_kpt = refer_kpt.unsqueeze(0).double(), query_kpt.unsqueeze(0).double()
            refer_score, query_score = refer_score.unsqueeze(0).double(), query_score.unsqueeze(0).double()

            if refer_kpt.shape[1] == 0 or query_kpt.shape[1] == 0:  # no keypoints
                return None

            kpts0 = normalize_keypoints(refer_kpt, images.shape)
            kpts1 = normalize_keypoints(query_kpt, images.shape)

            encoding0 = self.posenc(kpts0)
            encoding1 = self.posenc(kpts1)

            # Multi-layer Transformer network.
            refer_desc1, query_desc1 = self.gnn(refer_desc1, query_desc1, encoding0, encoding1)

            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(refer_desc1), self.final_proj(query_desc1)
            scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            scores = scores / self.config['descriptor_dim'] ** .5
            # Run the optimal transport.
            scores = log_optimal_transport(
                scores, self.bin_score,
                iters=self.config['sinkhorn_iterations'])
            # Get the matches with score above "match_threshold".
            indices0, indices1, mscores0, mscores1 = filter_matches(scores, th=self.config['match_threshold'])
            # 训练状态
            if is_training:
                all_matches, all_unmatches = find_matches(data['data_pred']['keypoints'][i], data['homography'][i])
                all_matches, all_unmatches = all_matches.unsqueeze(0), all_unmatches.unsqueeze(0)

                image_name = data['image_prefix_name']
                if len(all_matches[0]) == 0 or len(all_unmatches[0]) == 0:
                    return None
                # 损失
                loss_match = matches_loss(scores, all_matches, all_unmatches)  # 匹配损失
                # AUC
                avg_dist = valiation_data(data['data_pred']['keypoints'][i], matches0=indices0[0], matches1=indices1[0],
                                          img_name=image_name)
                auc = compute_auc(avg_dist)
                all_losses.append(loss_match)
                all_aucs.append(auc)

            all_results.append({
                'matches0': indices0[0],
                'matches1': indices1[0],
                'matching_scores0': mscores0[0],
                'matching_scores1': mscores1[0]
            })

        if is_training:
            loss_matches = sum(all_losses) / len(all_losses)
            auc_matches = np.sum(all_aucs) / len(all_aucs)
            return loss_matches, auc_matches

        return all_results
