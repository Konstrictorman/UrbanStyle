import argparse, os, glob, random
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils as vutils

# -------------------------
# Data
# -------------------------
class FolderDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def __init__(self, root, transform):
        self.paths = []
        for ext in self.IMG_EXTS:
            self.paths += glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}.")
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")  # grayscale
        return self.transform(img)

def make_loader(args):
    tfm = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor(),                # [0,1]
        transforms.Normalize([0.5], [0.5])    # -> [-1,1]
    ])
    if args.use_fashion_mnist:
        ds = datasets.FashionMNIST(root="./", train=True, download=True, transform=tfm)
        ds = torch.utils.data.Subset(ds, range(0, len(ds)))  # strip labels via subset
        collate = lambda batch: torch.stack([x[0] for x in batch], dim=0)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=2, pin_memory=True, collate_fn=collate)
    else:
        ds = FolderDataset(args.data, tfm)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=2, pin_memory=True)

# -------------------------
# Models (DCGAN-ish for 28x28)
# -------------------------
def weights_init(m):
    cname = m.__class__.__name__
    if "Conv" in cname or "Linear" in cname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias.data)
    elif "BatchNorm" in cname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)

class Generator(nn.Module):
    def __init__(self, z_dim=100, base=128):
        super().__init__()
        self.base = base
        self.fc = nn.Sequential(
            nn.Linear(z_dim, base*7*7),
            nn.BatchNorm1d(base*7*7),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base, base//2, 4, 2, 1),  # 7->14
            nn.BatchNorm2d(base//2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base//2, 1, 4, 2, 1),     # 14->28
            nn.Tanh(),
        )
        self.apply(weights_init)
    def forward(self, z):
        x = self.fc(z).view(-1, self.base, 7, 7)
        return self.net(x)  # [B,1,28,28]

class Discriminator(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 4, 2, 1),    # 28->14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1),  # 14->7
            nn.BatchNorm2d(base*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(base*2*7*7, 1)  # logits
        )
        self.apply(weights_init)
    def forward(self, x):
        return self.net(x)  # [B,1]

# -------------------------
# Training
# -------------------------
def train(args):
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "ckpts"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dl = make_loader(args)
    G = Generator(args.z_dim, args.g_base).to(device)
    D = Discriminator(args.d_base).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, args.z_dim, device=device)
    for epoch in range(1, args.epochs+1):
        G.train(); D.train()
        for imgs in dl:
            imgs = imgs.to(device)  # [B,1,28,28]
            B = imgs.size(0)
            real_lbl = torch.ones(B, 1, device=device)
            fake_lbl = torch.zeros(B, 1, device=device)

            # --- Update D: maximize log(D(x)) + log(1 - D(G(z))) ---
            z = torch.randn(B, args.z_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            D_real = D(imgs)
            D_fake = D(fake)
            lossD = bce(D_real, real_lbl) + bce(D_fake, fake_lbl)
            optD.zero_grad()
            lossD.backward()
            optD.step()

            # --- Update G: maximize log(D(G(z))) ---
            z = torch.randn(B, args.z_dim, device=device)
            fake = G(z)
            D_fake = D(fake)
            lossG = bce(D_fake, real_lbl)
            optG.zero_grad()
            lossG.backward()
            optG.step()

        # save samples/checkpoints each epoch
        G.eval()
        with torch.no_grad():
            grid = vutils.make_grid(G(fixed_z), nrow=8, normalize=True, value_range=(-1,1))
        vutils.save_image(grid, os.path.join(args.out_dir, "samples", f"epoch_{epoch:03d}.png"))
        torch.save(G.state_dict(), os.path.join(args.out_dir, "ckpts", f"G_epoch_{epoch:03d}.pt"))
        torch.save(D.state_dict(), os.path.join(args.out_dir, "ckpts", f"D_epoch_{epoch:03d}.pt"))
        print(f"[epoch {epoch}/{args.epochs}] saved samples and checkpoints.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="./data", help="Folder with product images")
    p.add_argument("--use_fashion_mnist", action="store_true", help="Use Fashion-MNIST instead of a folder")
    p.add_argument("--out_dir", type=str, default="./out")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--g_base", type=int, default=128)
    p.add_argument("--d_base", type=int, default=64)
    args = p.parse_args()
    random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    train(args)
