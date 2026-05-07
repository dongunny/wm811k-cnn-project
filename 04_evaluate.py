import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import importlib

dataset_module = importlib.import_module("01_dataset")
get_dataloaders = dataset_module.get_dataloaders

model_module = importlib.import_module("02_cnn_model")
SimpleCNN = model_module.SimpleCNN

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "best_cnn_model.pth")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_model():
    print("=" * 60)
    print("  WM-811K CNN Evaluation")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, classes = get_dataloaders(batch_size=64)
    num_classes = len(classes)
    model = SimpleCNN(num_classes=num_classes).to(device)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f"[Model Loaded] {MODEL_SAVE_PATH}")
    else:
        print("[Warning] No trained model found. Using random weights.")
    model.eval()
    all_preds, all_labels = [], []
    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n[Results] Balanced Accuracy: {bal_acc*100:.2f}%")
    for cls in classes:
        r = report[cls]
        print(f"  {cls:12s}: F1={r['f1-score']:.3f} | Prec={r['precision']:.3f} | Rec={r['recall']:.3f}")
    fig, ax = plt.subplots(figsize=(11, 9))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"CNN Confusion Matrix (Balanced Acc: {bal_acc*100:.2f}%)")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "fig6_cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cm_path}")
    fig, ax = plt.subplots(figsize=(12, 6))
    f1_scores = [report[c]['f1-score'] for c in classes]
    ax.bar(classes, f1_scores, color="#00d2ff")
    ax.axhline(bal_acc, color="orange", linestyle="--", label=f"Balanced Acc={bal_acc*100:.1f}%")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1-Score")
    ax.set_title("CNN F1-Score per Defect Pattern")
    ax.legend()
    plt.tight_layout()
    f1_path = os.path.join(RESULTS_DIR, "fig7_cnn_f1_scores.png")
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {f1_path}")
    output = {"model": "SimpleCNN", "balanced_acc": float(bal_acc),
               "per_class": {c: {"f1": float(report[c]['f1-score']),
                                  "precision": float(report[c]['precision']),
                                  "recall": float(report[c]['recall'])} for c in classes}}
    json_path = os.path.join(RESULTS_DIR, "cnn_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved: {json_path}")

if __name__ == "__main__":
    evaluate_model()
