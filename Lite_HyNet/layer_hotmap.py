# coding: utf-8
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from EUnet_mamba import EUnetmamba, Decoder, regnety, MFKM
from ABCnet import ABCNet
from unetformer import UNetFormer

# --- 配置路径 ---
# resume_path = '/root/autodl-tmp/model_weights/potsdam/abcnet-r18-768crop-ms-e45/abcnet-r18-768crop-ms-e45.ckpt'
# resume_path = '/root/autodl-fs/model_weights/potsdam/unetformer-r18-768crop-ms-e45/unetformer-r18-768crop-ms-e45.ckpt'
# resume_path = '/root/autodl-tmp/model_weights/potsdam/eunetmamba-r18-768crop-ms-e100/eunetmamba-r18-768crop-ms-e100.ckpt'
# resume_path = '/root/autodl-tmp/model_weights/vaihingen/Eunetmamba35-r18-512-crop-ms-e100/Eunetmamba35-r18-512-crop-ms-e100-v6.ckpt'
# resume_path = '/root/autodl-tmp/model_weights/vaihingen/abcnet-r18-512-crop-ms-e100/abcnet-r18-512-crop-ms-e100.ckpt'
# resume_path = '/root/autodl-fs/model_weights/vaihingen/unetformer-r18-512-crop-ms-e105/unetformer-r18-512-crop-ms-e105-v1.ckpt'
resume_path = '/root/autodl-tmp/model_weights/vaihingen/Eunet-r18-512-crop-ms-e100/Eunet-r18-512-crop-ms-e100-v3.ckpt'
# resume_path = '/root/autodl-tmp/model_weights/vaihingen/Eunetmamba35-r18-512-crop-ms-e100/Eunetmamba35-r18-512-crop-ms-e100-v6.ckpt'

# single_img_path = '/root/autodl-fs/data/potsdam/test/images_1024/top_potsdam_5_13_0_19.tif'
# single_img_path = '/root/autodl-fs/data/vaihingen/test/images_1024/top_mosaic_09cm_area2_0_7.tif'
# single_img_path = '/root/autodl-fs/data/potsdam/test/images_1024/top_potsdam_2_13_0_4.tif'
# single_img_path = '/root/autodl-fs/data/potsdam/test/images_1024/top_potsdam_4_15_0_15.tif's
single_img_path = '/root/autodl-fs/data/vaihingen/test/images_1024/top_mosaic_09cm_area16_0_5.tif'
# single_img_path = '/root/autodl-fs/data/potsdam/test/images_1024/top_potsdam_4_14_0_26.tif'
save_path = '/root/autodl-tmp/hotmap/gradcampp_overlay.png'
target_class = 3 # 指定想要可视化的类别索引

# 全局存储钩子捕获的特征及梯度
features_dict = {}
grads_dict = {}

def get_forward_hook(name):
    def forward_hook(module, inp, out):
        features_dict[name] = out
    return forward_hook

def get_backward_hook(name):
    def backward_hook(module, grad_in, grad_out):
        grads_dict[name] = grad_out[0]
    return backward_hook




def compute_gradcampp(feature_tensor, gradient_tensor):
    """
    计算 Grad-CAM++ 权重并生成热力图
    feature_tensor: [1, C, h, w]
    gradient_tensor: [1, C, h, w]
    返回: 热力图 numpy [h, w]
    """
    # 将 batch 维去掉
    fmap = feature_tensor[0]  # [C, h, w]
    grad = gradient_tensor[0]  # [C, h, w]

    # 确保 cam 跟 fmap 在同一设备
    device = fmap.device

    # 计算alpha系数
    grads_power_2 = grad ** 2
    grads_power_3 = grad ** 3
    sum_fmap = fmap.view(fmap.shape[0], -1).sum(dim=1).view(-1, 1, 1)
    eps = 1e-8
    alpha = grads_power_2 / (2 * grads_power_2 + sum_fmap * grads_power_3 + eps)
    weighted = alpha * F.relu(grad)
    # 确保它在内存中连续
    weighted_cont = weighted.contiguous()
    # 然后就能 safely 用 view
    weights = weighted_cont.view(fmap.shape[0], -1).sum(dim=1)

    # 在 fmap 相同设备上初始化 cam
    cam = torch.zeros(fmap.shape[1:], device=device, dtype=torch.float32)

    # 加权和特征图
    for i, w in enumerate(weights):
        cam += w * fmap[i]
    cam = F.relu(cam)

    # 归一化并转到 CPU
    cam -= cam.min()
    cam /= (cam.max() + eps)
    return cam.detach().cpu().numpy()

def make_heatmap_overlay(cam, orig_img_path, save_path, colormap=cv2.COLORMAP_JET):
    """
    将热力图叠加到原图并保存
    cam: [h, w] numpy, 0-1
    """
    img = cv2.imread(orig_img_path)
    H, W = img.shape[:2]
    heat_resized = cv2.resize(cam, (W, H))
    heat_uint8 = np.uint8(255 * heat_resized)
    heat_color = cv2.applyColorMap(heat_uint8, colormap)
    overlay = cv2.addWeighted(img, 0.8, heat_color, 0.5, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)
    print(f"Grad-CAM++ heatmap saved to {save_path}")

if __name__ == '__main__':
    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EUnetmamba().to(device)
    # model = ABCNet(n_classes=6).to(device)
    # model = UNetFormer().to(device)
    checkpoint = torch.load(resume_path, map_location='cpu')
    raw_state = checkpoint['state_dict']
    new_state = {}
    for k, v in raw_state.items():
        # 如果 key 形如 "net.encoder..."，就去掉 "net."
        new_key = k.replace('net.', '', 1)
        new_state[new_key] = v
    # 最后加载
    model.load_state_dict(new_state, strict=False)
    model.eval()
    # print(new_state.keys())

    # 要可视化的多层名称列表
    out_layers = ['encoder.s1', 'encoder.s2', 'encoder.s3', 'encoder.s4',  'decoder']
    # out_layers = ['decoder']
    # out_layers = ['conv_out']
    handles = []
    for lname in out_layers:
        layer = dict(model.named_modules())[lname]
        # 注册 forward/backward 钩子
        handles.append(layer.register_forward_hook(get_forward_hook(lname)))
        handles.append(layer.register_full_backward_hook(get_backward_hook(lname)))

    # 3. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(single_img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    print(input_tensor.shape)

    features_dict.clear()
    grads_dict.clear()

    # 4. 前向推理
    logits = model(input_tensor)  # 对应语义分割 [1, K, h, w]
    print(logits.shape)
    # # 聚合目标类别分数
    # score = logits[0, target_class].max()
    # 找出目标类别 mask
    mask = (logits.argmax(dim=1) == target_class)
    if mask.sum() > 0:
        score = logits[0, target_class].mean()
    else:
        print("未检测到目标类别区域")

    # 5. 反向传播计算梯度
    model.zero_grad()
    score.backward(retain_graph=False)

    # 生成并保存多层 heatmap
    for lname in out_layers:
        feat = features_dict[lname]  # [1, C, h, w]
        grad = grads_dict[lname]  # [1, C, h, w]
        cam = compute_gradcampp(feat, grad)
        # 构造保存文件名
        out_file = os.path.join(
            os.path.dirname(save_path),
            f"gradcampp_{lname}.png"
        )
        make_heatmap_overlay(cam, single_img_path, out_file)



    # 卸载所有钩子
    for h in handles:
        h.remove()
