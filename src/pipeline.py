import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

class LeafDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        path, label = self.rows[idx]
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        return image, label

class LeafCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, classes))
    def forward(self, x):
        return self.classifier(self.features(x))

def scan(root):
    folders = sorted(x for x in root.iterdir() if x.is_dir()) if root.exists() else []
    classes = [x.name for x in folders]
    if len(classes) < 2:
        raise ValueError("data/plantvillage altında en az iki sınıf klasörü bulunmalı.")
    rows = [(str(f), i) for i, folder in enumerate(folders) for f in sorted(folder.rglob("*")) if f.suffix.lower() in EXTS]
    counts = Counter(y for _, y in rows)
    if any(counts[i] < 10 for i in range(len(classes))):
        raise ValueError("Her sınıfta en az 10 görüntü olmalı; önerilen en az 50.")
    return rows, classes, counts

def plot_eda(rows, classes, out):
    counts = Counter(y for _, y in rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(classes)), [counts[i] for i in range(len(classes))]); ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_title("Sınıf başına görüntü sayısı"); ax.set_ylabel("Görüntü"); fig.tight_layout(); fig.savefig(out / "01_sinif_dagilimi.png", dpi=160); plt.close(fig)
    fig, axs = plt.subplots(1, min(len(classes), 4), figsize=(12, 3))
    axs = np.atleast_1d(axs)
    for i, ax in enumerate(axs):
        path = next(x for x, y in rows if y == i)
        with Image.open(path) as im: ax.imshow(im.convert("RGB"))
        ax.set_title(classes[i], fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "02_ornek_goruntuler.png", dpi=160); plt.close(fig)
    sizes = []
    for path, _ in rows[:min(len(rows), 300)]:
        with Image.open(path) as im: sizes.append(im.size)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter([x for x,y in sizes], [y for x,y in sizes], alpha=.35)
    ax.set(xlabel="Genişlik (piksel)", ylabel="Yükseklik (piksel)", title="Örnek görüntü boyutları (ilk 300)")
    fig.tight_layout(); fig.savefig(out / "03_goruntu_boyutlari.png", dpi=160); plt.close(fig)

def epoch_run(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, n = 0., 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images); loss = loss_fn(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(labels); correct += (logits.argmax(1) == labels).sum().item(); n += len(labels)
    return total_loss/n, correct/n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "plantvillage"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1: parser.error("epochs ve batch-size pozitif olmalı")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    rows, classes, counts = scan(Path(args.data))
    labels = [y for _, y in rows]
    train_rows, hold_rows = train_test_split(rows, test_size=.30, random_state=args.seed, stratify=labels)
    val_rows, test_rows = train_test_split(hold_rows, test_size=.50, random_state=args.seed, stratify=[y for _,y in hold_rows])
    train_t = transforms.Compose([transforms.Resize((128,128)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3)])
    eval_t = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3)])
    train_loader = DataLoader(LeafDataset(train_rows, train_t), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(LeafDataset(val_rows, eval_t), batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(LeafDataset(test_rows, eval_t), batch_size=args.batch_size, num_workers=0)
    plot_eda(rows, classes, out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeafCNN(len(classes)).to(device); loss_fn = nn.CrossEntropyLoss(); opt = torch.optim.Adam(model.parameters(), lr=.001)
    history = {"train_loss":[], "val_loss":[], "train_accuracy":[], "val_accuracy":[]}
    best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        tl, ta = epoch_run(model, train_loader, loss_fn, device, opt)
        vl, va = epoch_run(model, val_loader, loss_fn, device)
        for k,v in zip(history, (tl,vl,ta,va)): history[k].append(v)
        if vl < best:
            best = vl; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"Epoch {epoch+1}/{args.epochs} train_acc={ta:.3f} val_acc={va:.3f} val_loss={vl:.3f}")
    model.load_state_dict(best_state); model.eval()
    true, pred, wrong = [], [], []
    with torch.no_grad():
        for images, targets in test_loader:
            output = model(images.to(device)).argmax(1).cpu().tolist()
            true.extend(targets.tolist()); pred.extend(output)
    report = classification_report(true, pred, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(true, pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(max(7, len(classes)*1.2), max(5, len(classes))))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout(); fig.savefig(out / "05_karmasiklik_matrisi.png", dpi=160); plt.close(fig)
    fig, axs = plt.subplots(1,2,figsize=(11,4))
    axs[0].plot(history["train_accuracy"], label="Eğitim"); axs[0].plot(history["val_accuracy"], label="Doğrulama"); axs[0].set_title("Accuracy"); axs[0].legend()
    axs[1].plot(history["train_loss"], label="Eğitim"); axs[1].plot(history["val_loss"], label="Doğrulama"); axs[1].set_title("Loss"); axs[1].legend()
    fig.tight_layout(); fig.savefig(out / "04_egitim_grafikleri.png", dpi=160); plt.close(fig)
    torch.save({"state_dict": best_state, "classes": classes, "image_size":128}, out / "model.pt")
    metrics = {"dataset_size":len(rows), "class_counts":dict(zip(classes, [counts[i] for i in range(len(classes))])), "split":{"train":len(train_rows),"validation":len(val_rows),"test":len(test_rows)}, "epochs":args.epochs, "batch_size":args.batch_size, "accuracy":accuracy_score(true,pred), "macro_precision":report["macro avg"]["precision"], "macro_recall":report["macro avg"]["recall"], "macro_f1":report["macro avg"]["f1-score"], "classification_report":report, "confusion_matrix":cm.tolist(), "history":history, "dataset_source":"Kullanıcının data/plantvillage içine yerleştirdiği görüntüler; kaynağı raporda ayrıca doğrulayın"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Test accuracy:", round(metrics["accuracy"], 4), "| macro F1:", round(metrics["macro_f1"],4)); print("Sonuçlar outputs/ klasöründe")

if __name__ == "__main__": main()
