import torch
import torch.nn as nn
from models.SSPTmodel import build_sspt


class build_transformer(nn.Module):
    def __init__(self, opt, transformer_name):
        super(build_transformer, self).__init__()

        print('using Transformer_type: {} as a backbone'.format(transformer_name))

        if transformer_name == "SSPT_base":
            self.transformer =build_sspt()





    def forward(self,z,x):
        features = self.transformer(z,x)
        # features = self.transformer(x)
        return features

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_transformer_model(opt,transformer_name):
    model = build_transformer(opt, transformer_name=transformer_name)
    return model




