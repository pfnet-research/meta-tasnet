import torch
import torch.nn.functional as F


def sdr_objective(estimation, origin, mask=None):
    """
    Scale-invariant signal-to-noise ratio (SI-SNR) loss

    Arguments:
        estimation {torch.tensor} -- separated signal of shape: (B, 4, 1, T)
        origin {torch.tensor} -- ground-truth separated signal of shape (B, 4, 1, T)

    Keyword Arguments:
        mask {torch.tensor, None} -- boolean mask: True when $origin is 0.0; shape (B, 4, 1) (default: {None})

    Returns:
        torch.tensor -- SI-SNR loss of shape: (4)
    """
    origin_power = torch.pow(origin, 2).sum(dim=-1, keepdim=True) + 1e-8  # shape: (B, 4, 1, 1)
    scale = torch.sum(origin*estimation, dim=-1, keepdim=True) / origin_power  # shape: (B, 4, 1, 1)

    est_true = scale * origin  # shape: (B, 4, 1, T)
    est_res = estimation - est_true  # shape: (B, 4, 1, T)

    true_power = torch.pow(est_true, 2).sum(dim=-1).clamp(min=1e-8)  # shape: (B, 4, 1)
    res_power = torch.pow(est_res, 2).sum(dim=-1).clamp(min=1e-8)  # shape: (B, 4, 1)

    sdr = 10*(torch.log10(true_power) - torch.log10(res_power))  # shape: (B, 4, 1)

    if mask is not None:
        sdr = (sdr*mask).sum(dim=(0, -1)) / mask.sum(dim=(0, -1)).clamp(min=1e-8)  # shape: (4)
    else:
        sdr = sdr.mean(dim=(0, -1))  # shape: (4)

    return sdr  # shape: (4)


def dissimilarity_loss(latents, mask):
    """
    Minimize the similarity between the different instrument latent representations

    Arguments:
        latents {torch.tensor} -- latent matrix from the encoder of shape: (B, 1, T', N)
        mask {torch.tensor} -- boolean mask: True when the signal is 0.0; shape (B, 4)

    Returns:
        torch.tensor -- shape: ()
    """
    a_i = (0, 0, 0, 1, 1, 2)
    b_i = (1, 2, 3, 2, 3, 3)

    a = latents[a_i, :, :, :]
    b = latents[b_i, :, :, :]

    count = (mask[:, a_i] * mask[:, b_i]).sum() + 1e-8
    sim = F.cosine_similarity(a.abs(), b.abs(), dim=-1)
    sim = sim.sum(dim=(0, 1)) / count
    return sim.mean()


def similarity_loss(latents, mask):
    """
    Maximize the similarity between the same instrument latent representations

    Arguments:
        latents {torch.tensor} -- latent matrix from the encoder of shape: (B, 1, T', N)
        mask {torch.tensor} -- boolean mask: True when the signal is 0.0; shape (B, 4)

    Returns:
        torch.tensor -- shape: ()
    """
    a = latents
    b = torch.roll(latents, 1, dims=1)

    count = (mask * torch.roll(mask, 1, dims=0)).sum().clamp(min=1e-8)
    sim = F.cosine_similarity(a, b, dim=-1)
    sim = sim.sum(dim=(0, 1)) / count
    return sim.mean()


def calculate_loss(estimated_separation, true_separation, mask, true_latents, estimated_mix, true_mix, args):
    """
    The loss function, the sum of 4 different partial losses

    Arguments:
        estimated_separation {torch.tensor} -- separated signal of shape: (B, 4, 1, T)
        true_separation {torch.tensor} -- ground-truth separated signal of shape (B, 4, 1, T)
        mask {torch.tensor} -- boolean mask: True when $true_separation is 0.0; shape (B, 4, 1)
        true_latents {torch.tensor} -- latent matrix from the encoder of shape: (B, 1, T', N)
        estimated_mix {torch.tensor} -- estimated reconstruction of the mix, shape: (B, 1, T)
        true_mix {torch.tensor} -- ground-truth mixed signal, shape: (B, 1, T)
        args {dict} -- argparse hyperparameters

    Returns:
        (torch.tensor, torch.tensor) -- shape: [(), (7)]
    """
    stats = torch.zeros(7).to(mask.device)

    sdr = sdr_objective(estimated_separation, true_separation, mask)
    stats[:4] = sdr
    total_loss = -sdr.sum()

    reconstruction_sdr = sdr_objective(estimated_mix, true_mix).mean() if args.reconstruction_loss_weight > 0 else 0.0
    stats[4] = reconstruction_sdr
    total_loss += -args.reconstruction_loss_weight * reconstruction_sdr

    if args.similarity_loss_weight > 0.0 or args.dissimilarity_loss_weight > 0.0:
        mask = mask.squeeze(-1)
        true_latents = true_latents * mask.unsqueeze(-1).unsqueeze(-1)
        true_latents = true_latents.transpose(0, 1)

    dissimilarity = dissimilarity_loss(true_latents, mask) if args.dissimilarity_loss_weight > 0.0 else 0.0
    stats[5] = dissimilarity
    total_loss += args.dissimilarity_loss_weight * dissimilarity

    similarity = similarity_loss(true_latents, mask) if args.similarity_loss_weight > 0.0 else 0.0
    stats[6] = similarity
    total_loss += -args.similarity_loss_weight * similarity

    return total_loss, stats
