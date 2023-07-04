from .yacs import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.save_dir = '/'

# common params for network
# cfg.model = CfgNode(new_allowed=True)
# cfg.model.arch = CfgNode(new_allowed=True)
# cfg.model.backbone = CfgNode(new_allowed=True)
# cfg.model.head = CfgNode(new_allowed=True)


# DATASET related params
# cfg.data = CfgNode(new_allowed=True)
# cfg.data.train = CfgNode(new_allowed=True)
# cfg.data.val = CfgNode(new_allowed=True)
# cfg.device = CfgNode(new_allowed=True)
# # train
# cfg.schedule = CfgNode(new_allowed=True)
#
# # logger
# cfg.log = CfgNode()
# cfg.log.interval = 50
#
# # testing
# cfg.test = CfgNode()


def load_config(cfg, args_cfg):
    cfg.defrost()
    if isinstance(args_cfg, str):
        cfg.merge_from_file(args_cfg)
    else:
        cfg.merge_from_dict(args_cfg)
    cfg.freeze()


def dump_config(stream):
    cfg.dump(**{'stream': stream})


