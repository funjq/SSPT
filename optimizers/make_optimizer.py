import torch.optim as optim
from torch.optim import lr_scheduler


def make_optimizer(model, opt):
    ignored_params = []
    for i in [model.module.model_uav]:
        ignored_params += list(map(id, i.transformer.parameters()))
    extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
    optimizer_ft = optim.AdamW([
        {'params': base_params, 'lr': opt.lr},
        {'params': extra_params, 'lr': opt.lr * opt.NEK_W}],
        #weight_decay=0.05)
        weight_decay=5e-4)


    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=opt.num_epochs, eta_min=opt.adamw_cos)

    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=4, verbose=True,threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

    return optimizer_ft, exp_lr_scheduler
