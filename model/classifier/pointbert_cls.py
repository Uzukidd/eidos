import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "_base_":
                with open(new_config["_base_"], "r") as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, "r") as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config


class pointbert_cls(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.emb_dim = 384  # default

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, batch_X: torch.Tensor):
        if batch_X.size(1) == 3 or batch_X.size(1) == 6:
            batch_X = batch_X.permute((0, 2, 1))
        
        if batch_X.size(2) == 3:
            batch_X = F.pad(batch_X, (0, 3))

        embedding = self.backbone(batch_X)
        cls_token = embedding[:, 0, :]
        logit = self.classifier(cls_token)

        return logit

    @staticmethod
    def create_pointbert_cls(config_path: str, ckpt_path: str):
        pointbert = load_pointbert(config_path, ckpt_path, use_color=True)
        pointbert = pointbert_cls(pointbert, 40).cuda()

        return pointbert


def load_pointbert(config_path: str, ckpt_path: str = None, use_color: bool = False):
    from pointbert import PointTransformer

    point_bert_config = cfg_from_yaml_file(config_path)

    if use_color:
        point_bert_config.model.point_dims = 6
    use_max_pool = getattr(
        point_bert_config.model, "use_max_pool", False
    )  # * default is false

    pointbert = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
    print(f"Using {pointbert.point_dims} dim of points.")

    point_backbone_config = {
        "point_cloud_dim": point_bert_config.model.point_dims,
        "backbone_output_dim": (
            point_bert_config.model.trans_dim
            if not use_max_pool
            else point_bert_config.model.trans_dim * 2
        ),
        # "project_output_dim": self.config.hidden_size,
        # * number of output features, with cls token
        "point_token_len": (
            point_bert_config.model.num_group + 1 if not use_max_pool else 1
        ),
        # "mm_use_point_start_end": self.config.mm_use_point_start_end,
        "projection_hidden_layer": point_bert_config.model.get(
            "projection_hidden_layer", 0
        ),
        "use_max_pool": use_max_pool,
    }
    if point_bert_config.model.get("projection_hidden_layer", 0) > 0:
        # a list
        point_backbone_config["projection_hidden_dim"] = (
            point_bert_config.model.projection_hidden_dim
        )

    print(
        f"Use max pool is {use_max_pool}. Number of point token is {point_backbone_config['point_token_len']}."
    )

    if ckpt_path is not None:
        pointbert_prefix = "model.point_backbone."
        ckpt = torch.load(ckpt_path)
        ckpt = {
            k[len(pointbert_prefix) :]: v
            for k, v in ckpt.items()
            if k.startswith(pointbert_prefix)
        }
        pointbert.load_state_dict(ckpt)

    return pointbert
