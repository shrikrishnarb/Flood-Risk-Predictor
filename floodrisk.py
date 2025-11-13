import os
import json
import argparse
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# Optional imports for geospatial risk
try:
    import rasterio
except Exception:
    rasterio = None

# ---------- Simple U-Net ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base*4, base*8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.conv4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.conv1 = DoubleConv(base*2, base)

        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        u4 = self.up4(bn)
        c4 = self.conv4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(c4)
        c3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))
        out = self.outc(c1)
        return out

# ---------- Dataset ----------
class FloodDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: Optional[str] = None, size: int = 256):
        self.image_paths = sorted([os.path.join(image_dir, p) for p in os.listdir(image_dir) if p.lower().endswith((".png",".jpg",".jpeg"))])
        self.mask_dir = mask_dir
        self.size = size
        if mask_dir:
            self.mask_paths = sorted([os.path.join(mask_dir, p) for p in os.listdir(mask_dir) if p.lower().endswith((".png",".jpg",".jpeg"))])
            assert len(self.image_paths) == len(self.mask_paths), "Mismatch images/masks"
        else:
            self.mask_paths = [None]*len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB').resize((self.size, self.size))
        img_t = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.0
        if self.mask_paths[idx] is not None:
            m = Image.open(self.mask_paths[idx]).convert('L').resize((self.size, self.size))
            m = np.array(m)
            m = (m>127).astype(np.float32)
            mask_t = torch.from_numpy(m).unsqueeze(0)
        else:
            mask_t = torch.zeros(1, self.size, self.size)
        return img_t, mask_t

# ---------- Metrics ----------
def iou(pred: np.ndarray, target: np.ndarray, eps: float=1e-7) -> float:
    inter = ((pred>0.5) & (target>0.5)).sum()
    union = ((pred>0.5) | (target>0.5)).sum()
    return float((inter+eps)/(union+eps))

# ---------- Train/Eval ----------
def train(args):
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    ds = FloodDataset(args.images, args.masks, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)
    model = UNet(in_channels=3, base=args.base)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # BCE + Dice
    bce = nn.BCEWithLogitsLoss()
    def dice_loss(logits, targets, eps=1e-7):
        probs = torch.sigmoid(logits)
        num = 2*(probs*targets).sum()+eps
        den = (probs+targets).sum()+eps
        return 1 - (num/den)

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for x,y in tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}"):
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch+1}: loss={np.mean(losses):.4f}")
    torch.save(model.state_dict(), 'models/unet.pt')
    print('Saved weights to models/unet.pt')


def evaluate(args):
    os.makedirs('outputs', exist_ok=True)
    ds = FloodDataset(args.images, args.masks, size=args.size)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    model = UNet(in_channels=3, base=args.base)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device); model.eval()

    ious = []
    thresholds = np.linspace(0.1, 0.9, 17)
    precs = []; recs = []
    with torch.no_grad():
        for x,y in tqdm(dl, desc='eval'):
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0,0]
            targ  = y.numpy()[0,0]
            ious.append(iou(probs, targ))
            # precision/recall sweep
            p_list = []; r_list = []
            for t in thresholds:
                pred = (probs>t).astype(np.float32)
                TP = ((pred==1) & (targ==1)).sum()
                FP = ((pred==1) & (targ==0)).sum()
                FN = ((pred==0) & (targ==1)).sum()
                precision = TP / (TP+FP+1e-7)
                recall    = TP / (TP+FN+1e-7)
                p_list.append(precision); r_list.append(recall)
            precs.append(p_list); recs.append(r_list)
    print(f"Mean IoU: {np.mean(ious):.4f}")
    # Plot PR curve (mean across images)
    mean_p = np.mean(np.array(precs), axis=0)
    mean_r = np.mean(np.array(recs), axis=0)
    plt.figure(figsize=(5,4))
    plt.plot(mean_r, mean_p, marker='o')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Mean PR Curve')
    plt.grid(True)
    plt.savefig('outputs/pr_curve.png', dpi=200)
    plt.close()
    # Save iou histogram
    plt.figure(figsize=(5,4))
    plt.hist(ious, bins=10, color='steelblue')
    plt.xlabel('IoU'); plt.ylabel('Count'); plt.title('IoU distribution')
    plt.savefig('outputs/iou_hist.png', dpi=200)
    plt.close()


def infer(args):
    os.makedirs('outputs', exist_ok=True)
    img = Image.open(args.image).convert('RGB').resize((args.size, args.size))
    x = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.0
    x = x.unsqueeze(0)
    model = UNet(in_channels=3, base=args.base)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0,0].numpy()
    mask = (probs>args.threshold).astype(np.uint8)*255
    Image.fromarray(mask).save('outputs/mask.png')
    # overlay
    img_np = np.array(img)
    overlay = img_np.copy()
    overlay[mask>0] = (overlay[mask>0]*0.3 + np.array([0,0,255])*0.7).astype(np.uint8)
    Image.fromarray(overlay).save('outputs/overlay.png')
    print('Saved outputs/mask.png and outputs/overlay.png')

# ---------- Risk estimation ----------
def load_population(path: Optional[str], shape: Tuple[int,int]) -> np.ndarray:
    if path is None:
        # synthetic gradient population
        h,w = shape
        y = np.linspace(0,1,h).reshape(h,1)
        x = np.linspace(0,1,w).reshape(1,w)
        pop = (x+y)/2
        return pop.astype(np.float32)
    if path.lower().endswith('.csv'):
        import pandas as pd
        arr = pd.read_csv(path, header=None).values.astype(np.float32)
        return arr
    if path.lower().endswith(('.tif','.tiff')) and rasterio is not None:
        with rasterio.open(path) as src:
            arr = src.read(1)
            # normalize
            arr = (arr - np.nanmin(arr)) / (np.nanmax(arr)-np.nanmin(arr)+1e-7)
            return arr.astype(np.float32)
    raise ValueError('Unsupported population file or rasterio missing')


def risk(args):
    # infer mask first
    img = Image.open(args.image).convert('RGB').resize((args.size, args.size))
    x = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.0
    x = x.unsqueeze(0)
    model = UNet(in_channels=3, base=args.base)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0,0].numpy()
    mask = (probs>args.threshold).astype(np.float32)

    # load population grid and resample to mask size if needed
    pop = load_population(args.population, mask.shape)
    # simple resize if shapes mismatch
    if pop.shape != mask.shape:
        from skimage.transform import resize
        pop = resize(pop, mask.shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

    risk_score = float((mask * pop).sum())
    # heatmap
    plt.figure(figsize=(6,5))
    plt.imshow(pop, cmap='YlOrRd', alpha=0.6)
    plt.imshow(mask, cmap='Blues', alpha=0.4)
    plt.title(f'Flood × Population exposure (score={risk_score:.3f})')
    plt.colorbar(label='Population density (normalized)')
    plt.savefig('outputs/risk_heatmap.png', dpi=200)
    plt.close()
    # save report
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/risk_report.json','w') as f:
        json.dump({'risk_score': risk_score}, f, indent=2)
    print('Saved outputs/risk_heatmap.png and outputs/risk_report.json')

# ---------- CLI ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FloodRisk AI (minimal)')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train')
    p_train.add_argument('--images', required=True)
    p_train.add_argument('--masks', required=True)
    p_train.add_argument('--epochs', type=int, default=5)
    p_train.add_argument('--batch', type=int, default=4)
    p_train.add_argument('--lr', type=float, default=1e-3)
    p_train.add_argument('--size', type=int, default=256)
    p_train.add_argument('--base', type=int, default=32)

    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--images', required=True)
    p_eval.add_argument('--masks', required=True)
    p_eval.add_argument('--weights', default='models/unet.pt')
    p_eval.add_argument('--size', type=int, default=256)
    p_eval.add_argument('--base', type=int, default=32)

    p_infer = sub.add_parser('infer')
    p_infer.add_argument('--image', required=True)
    p_infer.add_argument('--weights', default='models/unet.pt')
    p_infer.add_argument('--size', type=int, default=256)
    p_infer.add_argument('--base', type=int, default=32)
    p_infer.add_argument('--threshold', type=float, default=0.5)

    p_risk = sub.add_parser('risk')
    p_risk.add_argument('--image', required=True)
    p_risk.add_argument('--weights', default='models/unet.pt')
    p_risk.add_argument('--population', default=None)
    p_risk.add_argument('--size', type=int, default=256)
    p_risk.add_argument('--base', type=int, default=32)
    p_risk.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()
    if args.cmd == 'train':
        train(args)
    elif args.cmd == 'eval':
        evaluate(args)
    elif args.cmd == 'infer':
        infer(args)
    elif args.cmd == 'risk':
        risk(args)