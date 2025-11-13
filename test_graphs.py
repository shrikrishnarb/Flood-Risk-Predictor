import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs('outputs', exist_ok=True)

# Synthetic image & mask
H,W = 256,256
img = np.zeros((H,W,3), dtype=np.uint8)
# draw a river-like shape
for y in range(H):
    x0 = int(40 + 20*np.sin(y/20))
    img[y, x0:x0+40, 2] = 180  # blueish
mask = np.zeros((H,W), dtype=np.uint8)
mask[:, 40:80] = 1

Image.fromarray(img).save('outputs/synth_image.png')
Image.fromarray((mask*255).astype(np.uint8)).save('outputs/synth_mask.png')

# IoU/PR demo with synthetic noisy prediction
pred_prob = mask.astype(np.float32) * 0.8 + (1-mask.astype(np.float32))*0.2
noise = np.random.normal(0, 0.1, size=pred_prob.shape)
pred_prob = np.clip(pred_prob + noise, 0, 1)

# IoU
inter = ((pred_prob>0.5) & (mask>0)).sum()
union = ((pred_prob>0.5) | (mask>0)).sum()
iou = (inter+1e-7)/(union+1e-7)
print(f'IoU (synthetic): {iou:.3f}')

# PR curve
ths = np.linspace(0.1,0.9,17)
P,R = [],[]
for t in ths:
    pred = (pred_prob>t).astype(np.uint8)
    TP = ((pred==1)&(mask==1)).sum()
    FP = ((pred==1)&(mask==0)).sum()
    FN = ((pred==0)&(mask==1)).sum()
    P.append(TP/(TP+FP+1e-7))
    R.append(TP/(TP+FN+1e-7))

plt.figure(figsize=(5,4))
plt.plot(R,P,marker='o')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Synthetic PR Curve')
plt.grid(True)
plt.savefig('outputs/synth_pr.png', dpi=200)
plt.close()

# Overlay visualization
overlay = img.copy()
overlay[(pred_prob>0.5)] = (overlay[(pred_prob>0.5)]*0.3 + np.array([0,0,255])*0.7).astype(np.uint8)
Image.fromarray(overlay).save('outputs/synth_overlay.png')

# Risk heatmap (synthetic population)
y = np.linspace(0,1,H).reshape(H,1)
x = np.linspace(0,1,W).reshape(1,W)
pop = (x+y)/2
risk = float(((pred_prob>0.5).astype(np.float32) * pop).sum())
plt.figure(figsize=(6,5))
plt.imshow(pop, cmap='YlOrRd', alpha=0.6)
plt.imshow((pred_prob>0.5), cmap='Blues', alpha=0.4)
plt.title(f'Synthetic Risk (score={risk:.3f})')
plt.colorbar(label='Population density (normalized)')
plt.savefig('outputs/synth_risk.png', dpi=200)
plt.close()
print('Saved outputs: synth_image.png, synth_mask.png, synth_pr.png, synth_overlay.png, synth_risk.png')