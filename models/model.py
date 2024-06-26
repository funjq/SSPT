import torch.nn as nn
from .make_model import make_transformer_model
from .model_neck_query import PN
import torch

Transformer_model_list = ["SSPT_base"]

class SiamUAV_Transformer_Model(nn.Module):
    def __init__(self, opt):
        super(SiamUAV_Transformer_Model, self).__init__()
        backbone = opt.backbone
        self.model_uav = make_transformer_model(opt, transformer_name=backbone)
        # self.model_uav = build_sbtv2_base_model()
        if opt.head == "PN":
            self.model_head = PN(opt)
        self.opt = opt

    def forward(self,z,x):
        out = self.model_uav(z,x)#
        x=out["x"]
        z=out["z"]

        cls, loc = self.model_head(z,x, opt=self.opt)
        return cls, loc

    def load_params(self, opt):
        # load pretrain param
        # if opt.backbone == "SSPT_base":
        #     pretrain = opt.pretrain


        if opt.USE_old_model:
            self.load_param_self_backbone(opt.old_model_pretrain)
        else:
            self.model_uav.transformer.load_param_self_backbone(opt.pretrain)



    def load_param_self_backbone(self, pretrained):
        if isinstance(pretrained, str):
            # model_dict2 = self.state_dict()
            state_dict = torch.load(pretrained)
            self.load_state_dict(state_dict, strict=False)




def make_model(opt, pretrain=False):
    # if opt.backbone_satellite in Transformer_model_list:
    if opt.backbone in Transformer_model_list:
        model = SiamUAV_Transformer_Model(opt)
        if pretrain:
            model.load_params(opt)
    return model
