import torch
import torch.nn as nn
import torch.nn.functional as F


def L1_loss(pred, target, weight=None, reduction='sum'):
    assert reduction == 'sum'

    loss = F.l1_loss(pred, target, reduction='none')
    if weight is not None:
        weight = weight.float()

    # weighted elemen-wise losses
    if weight.dim() != loss.dim():
        weight = weight.unsqueeze(1)
    loss = weight * loss

    # do reduction
    return loss.sum()


def L2_loss(pred, target, weight=None, reduction='sum'):
    assert reduction == 'sum'

    loss = F.mse_loss(pred, target, reduction='none')

    if weight is not None:
        weight = weight.float()

    # weighted elemen-wise losses
    if weight.dim() != loss.dim():
        weight = weight.unsqueeze(1)
    loss = weight * loss

    # do reduction
    return loss.sum()


def mse_loss(pred, target, weight=None, reduction='sum'):
    assert reduction == 'sum'

    loss = F.mse_loss(pred, target, reduction='none')
    if weight is not None:
        weight = weight.float()

    # weighted elemen-wise losses
    if weight.dim() != loss.dim():
        weight = weight.unsqueeze(1)
    loss = weight * loss

    # do reduction
    return loss.sum()


def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)

    return bin_labels, bin_label_weights


def binary_cross_entropy_loss(pred, target, weight=None, reduction='sum'):
    assert reduction == 'sum'
    # no class_weight param, only single classes are actually supported
    if pred.dim() != target.dim():
        label, weight = _expand_onehot_labels(target, weight, pred.size(-1))

    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')

    # weighted elemen-wise losses
    loss = weight * loss

    # do reduction
    return loss.sum()


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.Tensor([target_real_label]))
        self.register_buffer('fake_label', torch.tensor([target_fake_label]))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.criterion = mse_loss
        elif gan_mode == 'vanilla':
            self.criterion = binary_cross_entropy_loss
        elif gan_mode == 'wgangp':
            self.criterion = None
        elif gan_mode == 'wgan':
            self.criterion = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, weight):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.criterion(prediction, target_tensor, weight)
        elif self.gan_mode == 'wgangp' or self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            raise NotImplementedError

        return loss


if __name__ == '__main__':
    criterion_mse = GANLoss(gan_mode='lsgan')
    criterion_bce = GANLoss(gan_mode='vanilla')

    h, w = 64, 64
    pred = torch.randn((2, 1, h, w))
    target = False
    mask = torch.randint(0, 2, (2, h, w))
    weight = mask / (mask.sum() + 1e-6)

    loss = criterion_mse(pred, target, weight)
    print(f'MSE Loss: {loss.item()}')

    bce_loss = criterion_bce(pred.squeeze(1), target, weight)
    print(f'BCE Loss: {bce_loss.item()}')


    # gradient penalty [wgan-gp]
    # use this code when optimizing discriminator, fake predict
    b_size = 8
    real_img = torch.randn((b_size, 3, 480, 640))
    fake_img = torch.randn((b_size, 3, 480, 640))
    discriminator = None  # a discriminator or called critic
    eps = torch.rand(b_size, 1, 1, 1).to(real_img.device)
    x_hat = eps * real_img.data + (1 - eps) * fake_img.data
    x_hat.requires_grad = True
    from torch.autograd import grad
    hat_predict = discriminator(x_hat)
    grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
    grad_penalty = (
            (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
    ).mean()
    grad_penalty = 10 * grad_penalty       # the lambda coefficient equals to 10 in wgan-gp
    grad_penalty.backward()
