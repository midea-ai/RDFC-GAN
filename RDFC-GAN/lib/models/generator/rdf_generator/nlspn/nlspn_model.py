from .modulated_deform_conv_func import ModulatedDeformConvFunction
import torch
import torch.nn as nn


class NLPSN(nn.Module):
    def __init__(self, channels_g, channels_f, k_g, k_f,
                 prop_time=1, affinity=None, affinity_gamma=0.5, conf_prop=True, preserve_input=False):
        super(NLPSN, self).__init__()

        self.conf_prop = conf_prop
        self.preserve_input = preserve_input

        # Guidance: [B, channels_g, h, w]
        # Feature: [B, channels_f, h, w]

        assert channels_f == 1, 'only tested with channels_f == 1 but {}'.format(channels_g)
        assert (k_g % 2) == 1, 'only odd kernel is supported but k_g = {}'.format(k_g)
        assert (k_f % 2) == 1, 'only odd kernel is supported but k_g = {}'.format(k_f)

        pad_g = int((k_g - 1) / 2)
        pad_f = int((k_f - 1) / 2)

        self.prop_time = prop_time
        self.affinity = affinity

        self.channels_g = channels_g
        self.channels_f = channels_f
        self.k_g = k_g
        self.k_f = k_f
        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2
        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            self.conv_offset_aff = nn.Conv2d(self.channels_g, 3 * self.num, kernel_size=self.k_g, stride=1,
                                             padding=pad_g, bias=True)
            self.conv_offset_aff.weight.data.zero_()
            self.conv_offset_aff.bias.data.zero_()

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.channels_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.channels_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.channels_f
        self.deformable_groups = 1
        self.im2col_step = 64

    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        B, _, H, W = guidance.shape

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity == 'TC':
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Apply confidence
        # import pdb
        # pdb.set_trace()
        if self.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)

            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

            for idx_off in range(0, self.num + 1):
                ww = idx_off % self.k_f
                hh = idx_off // self.k_f

                if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                    continue

                offset_tmp = offset_each[idx_off].detach()

                conf_tmp = ModulatedDeformConvFunction.apply(confidence,
                                                             offset_tmp,
                                                             modulation_dummy,
                                                             self.w_conf, self.b, self.stride, 0, self.dilation,
                                                             self.groups, self.deformable_groups, self.im2col_step)
                list_conf.append(conf_tmp)

            conf_aff = torch.cat(list_conf, dim=1)
            aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(feat, offset, aff, self.w, self.b, self.stride, self.padding,
                                                 self.dilation, self.groups, self.deformable_groups, self.im2col_step)

        return feat

    def forward(self, feat_init, guidance, confidence=None, feat_fix=None, rgb=None):
        assert self.channels_g == guidance.shape[1]
        assert self.channels_f == feat_init.shape[1]

        if self.conf_prop:
            assert confidence is not None
            offset, aff = self._get_offset_affinity(guidance, confidence, rgb)
        else:
            offset, aff = self._get_offset_affinity(guidance, None, rgb)

        # Propagation
        if self.preserve_input:
            assert feat_init.shape == feat_fix.shape
            mask_fix = torch.sum(feat_fix > 0.0, dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(feat_fix)

        feat_result = feat_init

        list_feat = []

        for k in range(1, self.prop_time + 1):
            # Input preservation for each iteration
            if self.preserve_input:
                feat_result = (1.0 - mask_fix) * feat_result + mask_fix * feat_fix

            feat_result = self._propagate_once(feat_result, offset, aff)

            list_feat.append(feat_result)

        return feat_result, list_feat, offset, aff, self.aff_scale_const.data


class NLSPNRefineModule(nn.Module):
    """No encoder-decoder here. """
    def __init__(self, prop_kernel=3, prop_time=18, affinity='TGASS',
                 affinity_gamma=0.5, conf_prop=True, preserve_input=False):
        super(NLSPNRefineModule, self).__init__()

        self.num_neighbors = prop_kernel * prop_kernel - 1
        self.conf_prop = conf_prop

        self.prop_layer = NLPSN(channels_g=self.num_neighbors, channels_f=1, k_g=3, k_f=prop_kernel,
                                prop_time=prop_time, affinity=affinity, affinity_gamma=affinity_gamma,
                                conf_prop=conf_prop, preserve_input=preserve_input)


    def forward(self, init_pred_depth, guide, confidence, origin_depth, origin_rgb=None):
        y, y_inter, offset, aff, aff_const = self.prop_layer(init_pred_depth, guide, confidence, origin_depth, origin_rgb)

        # y = torch.tanh(y)

        return y, confidence
