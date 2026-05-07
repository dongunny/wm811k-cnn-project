import os
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import re

PKL_PATH = os.path.join(os.path.dirname(__file__), '..', 'LSWMD.pkl')
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), 'processed_data.npz')
TARGET_SIZE = 64
DEFECT_TYPES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']

class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('pandas.indexes'):
            module = module.replace('pandas.indexes', 'pandas.core.indexes')
        return super().find_class(module, name)

def extract_label(x):
    if isinstance(x, np.ndarray):
        flat = x.flat[0] if x.size > 0 else 'unknown'
        return str(flat).strip()
    if isinstance(x, list):
        v = x[0] if len(x) > 0 else 'unknown'
        return str(v).strip()
    s = str(x).strip()
    if s in ('nan', 'None', ''):
        return 'unknown'
    m = re.findall(r"'([^']+)'", s)
    if m: return m[0]
    return s

def prepare_data():
    if os.path.exists(PROCESSED_PATH):
        print("Already preprocessed. Loading cached data.")
        return
    print("[1/3] Loading LSWMD.pkl...")
    with open(PKL_PATH, 'rb') as f:
        df = CompatUnpickler(f, encoding='latin1').load()
    print("[2/3] Extracting labels...")
    df['label'] = df['failureType'].apply(extract_label)
    df_cls = df[df['label'].isin(DEFECT_TYPES)].copy()
    print(f"Defective wafers: {len(df_cls)}")
    print("[3/3] Resizing to 64x64...")
    X_list, y_list = [], []
    for i, (_, row) in enumerate(df_cls.iterrows()):
        if i % 2000 == 0:
            print(f"  Progress: {i}/{len(df_cls)}")
        wmap = np.array(row['waferMap'], dtype=float)
        if wmap.size == 0 or wmap.shape[0] == 0:
            continue
        zy = TARGET_SIZE / wmap.shape[0]
        zx = TARGET_SIZE / wmap.shape[1]
        resized = zoom(wmap, (zy, zx), order=0)
        X_list.append(resized)
        y_list.append(row['label'])
    X = np.array(X_list)
    le = LabelEncoder()
    y = le.fit_transform(y_list)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, stratify=y_temp, random_state=42)
    np.savez_compressed(PROCESSED_PATH, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, classes=le.classes_)
    print(f"Saved to {PROCESSED_PATH}")

class WaferDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y, self.augment = X, y, augment
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = self.X[idx].copy()
        if self.augment:
            if np.random.rand() > 0.5: img = np.fliplr(img)
            if np.random.rand() > 0.5: img = np.flipud(img)
            img = np.rot90(img, np.random.randint(0, 4))
        img = img / 2.0
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0), torch.tensor(self.y[idx], dtype=torch.long)

def get_dataloaders(batch_size=64):
    prepare_data()
    data = np.load(PROCESSED_PATH, allow_pickle=True)
    train_d = WaferDataset(data['X_train'], data['y_train'], augment=True)
    val_d = WaferDataset(data['X_val'], data['y_val'])
    test_d = WaferDataset(data['X_test'], data['y_test'])
    return (DataLoader(train_d, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(val_d, batch_size=batch_size, num_workers=0),
            DataLoader(test_d, batch_size=batch_size, num_workers=0),
            data['classes'])

if __name__ == "__main__":
    prepare_data()
