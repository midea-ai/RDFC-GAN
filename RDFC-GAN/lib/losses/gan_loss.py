import torch
import torch.nn as nn
import torch.nn.functional as F


def L1_loss(pred, target, weight=None, reduction='sum'):
    assert reduction == 'sum'

    loss = F.l1_loss(pred, target, reduction='none')
    if weight is not None:
        weight = weight.float()
    else:
        weight = torch.ones_like(pred).to(pred.device)
        weight = weight / (weight.sum() + 1e-6)

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
    else:
        weight = torch.ones_like(pred).to(pred.device)
        weight = weight / (weight.sum() + 1e-6)

    # weighted elemen-wise losses
    if weight.dim() != loss.dim():
        weight = weight.unsqueeze(1)
    loss = weight * loss

    # do reduction
    return loss.sum()

def norm_normalize(norm_out):
    norm_x, norm_y, norm_z = torch.split(norm_out, 1, dim=1)
    norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    final_out = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm], dim=1)
    return final_out

def manhattan_loss(normal,semantic,norm_mask,label_wall,label_floor,label_ceiling,weight = None):

    semantic = F.softmax(semantic, dim=1).argmax(1)

    wall_mask = semantic == label_wall
    floor_mask = semantic == label_floor
    ceiling_mask = semantic == label_ceiling
    floor_loss,wall_loss,ceiling_loss = 0.0,0.0,0.0

    gt_floor = torch.zeros_like(normal).to(normal.device)
    gt_floor[:,1,:,:] = 1
    gt_ceiling = torch.zeros_like(normal).to(normal.device)
    gt_ceiling[:,1,:,:] = -1

    #normal = norm_normalize(normal)

    if norm_mask.dim() != normal.dim():
        norm_mask = norm_mask.unsqueeze(1)
    #floor_loss 
    if floor_mask.sum() > 0:
        if normal.dim() != floor_mask.dim():
            floor_mask = torch.unsqueeze(floor_mask,dim = 1)
        surface_normal_floor = normal * floor_mask
        floor_loss = torch.cosine_similarity(surface_normal_floor, gt_floor, dim=1)
        floor_loss = (floor_loss * (-1) + 1) * floor_mask
        floor_loss = floor_loss.sum() / ((floor_loss != 0).sum() + 1e-6)

    #wall_loss 
    if wall_mask.sum() > 0:
        if normal.dim() != wall_mask.dim():
            wall_mask = torch.unsqueeze(wall_mask,dim = 1)
        surface_normal_wall = normal * wall_mask
        
        wall_loss = torch.cosine_similarity(surface_normal_wall, gt_floor, dim=1)
        wall_loss = wall_loss * wall_mask * 2
        wall_loss = (wall_loss.abs().sum()) / ((wall_loss != 0).sum() + 1e-6)

    #ceiling-loss 
    if ceiling_mask.sum() > 0:
        if normal.dim() != ceiling_mask.dim():
            ceiling_mask = torch.unsqueeze(ceiling_mask,dim = 1)
        surface_normal_ceiling = normal * ceiling_mask
        ceiling_loss = torch.cosine_similarity(surface_normal_ceiling, gt_ceiling, dim=1) 
        ceiling_loss = (ceiling_loss * (-1) + 1) * ceiling_mask
        ceiling_loss = (ceiling_loss.sum()) / ((ceiling_loss != 0).sum() + 1e-6)

    if weight:
        return  floor_loss * weight ,wall_loss * weight ,ceiling_loss * weight
    else:
        floor_loss ,wall_loss ,ceiling_loss
    



def mse_loss(pred, target, weight=None, reduction='sum'):
    assert reduction == 'sum'

    loss = F.mse_loss(pred, target, reduction='none')
    if weight is not None:
        weight = weight.float()
    else:
        weight = torch.ones_like(pred).to(pred.device)
        weight = weight / (weight.sum() + 1e-6)

    # weighted elemen-wise losses
    if weight.dim() != loss.dim():
        weight = weight.unsqueeze(1)
    loss = weight * loss

    # do reduction
    return loss.sum()

def nor_loss(pred,target,norm_masks):

    dot = torch.cosine_similarity(pred, target, dim=1)

    if norm_masks.dim() != 4:
        norm_masks = norm_masks.unsqueeze(1)
    valid_mask = norm_masks[:,0,:,:].float() * (dot.detach() < 0.999).float() * (dot.detach() > -0.999).float()
    valid_mask = valid_mask > 0.0
    dot = ((dot * -1) + 1) * valid_mask 
    loss = dot.sum() / ((dot != 0.0).sum() + 1e-6)
    return loss


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

    def __call__(self, prediction, target_is_real, weight=None):
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
