import os
import numpy as np
from PIL import Image
import lpips
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

loss_fn = lpips.LPIPS(net='alex').cuda()

def compute_metrics(gt_dir: str, enhanced_dir: str, dataset_name: str):
    psnrs, ssims, lpipses = [], [], []
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for fname in gt_files:
        gt_path = os.path.join(gt_dir, fname)
        enh_path = os.path.join(enhanced_dir, fname)
        
        if not os.path.exists(enh_path):
            continue
            
        gt = np.array(Image.open(gt_path).convert('RGB'))
        enh = np.array(Image.open(enh_path).convert('RGB'))
        
        # PSNR & SSIM
        p = psnr(gt, enh, data_range=255)
        s = ssim(gt, enh, channel_axis=2, data_range=255)
        
        # LPIPS
        gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        enh_tensor = torch.from_numpy(enh).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        l = loss_fn(enh_tensor.cuda(), gt_tensor.cuda()).item()
        
        psnrs.append(p)
        ssims.append(s)
        lpipses.append(l)
    
    print(f"\n=== {dataset_name} ===")
    print(f"PSNR↑  : {np.mean(psnrs):.2f}")
    print(f"SSIM↑  : {np.mean(ssims):.3f}")
    print(f"LPIPS↓ : {np.mean(lpipses):.3f}")
    print(f"测试图像数量: {len(psnrs)}")

if __name__ == "__main__":
    # data, results, dataset name
    compute_metrics("datasets/Huawei/normal", "results/Huawei", "Huawei")
    compute_metrics("datasets/Nikon/normal", "results/Nikon", "Nikon")
    # compute_metrics("datasets/FiveK/normal", "results/FiveK", "FiveK")