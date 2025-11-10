from torch.utils.data import DataLoader
from losses import *
from datasets.vaihingen_dataset import *
from Lite_HyNet.models.Lite_HyNet import Lite_HyNet
from fvcore.nn import flop_count, parameter_count
import copy
from catalyst.contrib.nn import Lookahead
from catalyst import utils


# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

# weights_name = "Eunetmamba35-r18-512-crop-ms-e100"
weights_name = "Lite_HyNet-r18-512-crop-ms-e100"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "Lite_HyNet-r18-512-crop-ms-e100"
# test_weights_name = "Eunetmamba35-r18-512-crop-ms-e100"

log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = Lite_HyNet(num_classes=num_classes)

# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader

train_dataset = VaihingenDataset(data_root='/root/autodl-fs/data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(data_root='/root/autodl-fs/data/vaihingen/val', transform=val_aug, mode='val')
test_dataset = VaihingenDataset(data_root='/root/autodl-fs/data/vaihingen/test', mode='test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=8,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
model = copy.deepcopy(net)
model.cuda().eval()
input = torch.randn((1, 3, 1024, 1024), device=next(model.parameters()).device)
params = parameter_count(model)[""]
Gflops, unsupported = flop_count(model=model, inputs=(input,))
print('GFLOPs: ', sum(Gflops.values()), 'G')
print('Params: ', params/1e6, 'M')
