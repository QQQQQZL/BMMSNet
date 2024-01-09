import tensorflow as tf
import torch
import torch.nn.functional as F


    

def pearson_torch(y_true, y_pred, axis=1):
    """Pearson correlation function implemented in PyTorch.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth labels. Shape is (batch_size, time_steps, n_features)
    y_pred: torch.Tensor
        Predicted labels. Shape is (batch_size, time_steps, n_features)
    axis: int
        Axis along which to compute the pearson correlation. Default is 1.

    Returns
    -------
    torch.Tensor
        Pearson correlation.
        Shape is (batch_size, 1, n_features) if axis is 1.
    """
    # Compute the mean of the true and predicted values
    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = torch.sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean),
        dim=axis,
        keepdim=True,
    )
    std_true = torch.sum(torch.square(y_true - y_true_mean), dim=axis, keepdim=True) + 1e-8
    std_pred = torch.sum(torch.square(y_pred - y_pred_mean), dim=axis, keepdim=True) + 1e-8
    denominator = torch.sqrt(std_true * std_pred + 1e-8 )

    # Compute the pearson correlation
    return torch.mean(torch.div(numerator, denominator), dim=-1)




def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))


        
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]


    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def temporal_masking(tensor, mask_ratio):
    B, T, F = tensor.size()  # 获取输入张量的维度
    mask_length = int(T * mask_ratio)  # 计算需要mask的时间点数量

    # 生成一个随机的mask索引
    mask_indices = torch.randperm(T)[:mask_length]

    # 创建一个与输入张量相同形状的mask，初始值为1
    mask = torch.ones(B, T, 1).to(tensor.device)

    # 将mask索引对应的时间点置为0
    mask[:, mask_indices, :] = 0

    # 应用mask到输入张量
    masked_tensor = tensor * mask

    return masked_tensor



