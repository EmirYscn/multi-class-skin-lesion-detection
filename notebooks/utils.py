# =============================================================================
# utils.py — Shared utilities for all experiment notebooks
# =============================================================================

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import wandb
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, roc_curve, auc
)
from wandb.integration.keras import WandbMetricsLogger

tf.random.set_seed(42)
np.random.seed(42)


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR    = '/Users/emiryscn/repos/multi-class-skin-lesion-detection'
TRAIN_DIR   = os.path.join(BASE_DIR, 'data/processed/train')
VAL_DIR     = os.path.join(BASE_DIR, 'data/processed/val')
TEST_DIR    = os.path.join(BASE_DIR, 'data/processed/test')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 7
SEED        = 42

# Training hyperparameters
PHASE_A_EPOCHS = 15
PHASE_A_LR     = 1e-3
PHASE_B_EPOCHS = 20
PHASE_B_LR     = 1e-5
PATIENCE       = 5
DROPOUT        = 0.4

CLASS_NAMES = sorted([c for c in os.listdir(TRAIN_DIR) if not c.startswith('.')])

READABLE_NAMES = {
    'akiec': 'Actinic Keratoses', 'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis', 'df': 'Dermatofibroma',
    'mel': 'Melanoma', 'nv': 'Melanocytic Nevi', 'vasc': 'Vascular Lesions'
}


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def compute_class_weights(train_dir=TRAIN_DIR, class_names=CLASS_NAMES):
    """Compute inverse-frequency weights: weight = total / (n_classes * n_class)."""
    counts = {}
    for cls in class_names:
        cls_path = os.path.join(train_dir, cls)
        n = len([f for f in os.listdir(cls_path) if not f.startswith('.')])
        counts[cls] = n
    total = sum(counts.values())
    n_classes = len(class_names)
    class_weight = {i: total / (n_classes * counts[cls]) for i, cls in enumerate(class_names)}
    return class_weight


# =============================================================================
# DATA LOADING
# =============================================================================

def load_base_dataset(directory, shuffle=True):
    """
    Load images from class-folder structure. Returns UNBATCHED tf.data.Dataset
    of (images, one_hot_labels) with pixels in [0, 1].
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        directory, image_size=IMG_SIZE, batch_size=None,
        label_mode='categorical', shuffle=shuffle, seed=SEED, class_names=CLASS_NAMES
    )
    ds = ds.map(lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds


# =============================================================================
# AUGMENTATION
# =============================================================================

# Traditional augmentation layer
traditional_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom((-0.1, 0.1)),
    tf.keras.layers.RandomContrast(0.1),
], name="traditional_augmentation")


def apply_traditional_augmentation(image, label):
    """Apply traditional augmentation to a single image."""
    image = traditional_augmentation(image, training=True)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def mixup_batch(images, labels, alpha=0.2):
    """Apply MixUp to a batch. Blends pairs of images and their labels."""
    batch_size = tf.shape(images)[0]
    lam = tf.random.gamma(shape=[batch_size, 1, 1, 1], alpha=alpha)
    lam = lam / (lam + tf.random.gamma(shape=[batch_size, 1, 1, 1], alpha=alpha))
    indices = tf.random.shuffle(tf.range(batch_size))
    lam_labels = tf.reshape(lam, [batch_size, 1])
    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, indices)
    mixed_labels = lam_labels * labels + (1.0 - lam_labels) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


def cutmix_batch(images, labels, alpha=1.0):
    """Apply CutMix to a batch. Cuts a patch from one image and pastes onto another."""
    batch_size = tf.shape(images)[0]
    img_h, img_w = IMG_SIZE
    lam = tf.random.gamma(shape=[], alpha=alpha)
    lam = lam / (lam + tf.random.gamma(shape=[], alpha=alpha))
    cut_ratio = tf.math.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(img_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(img_w, tf.float32) * cut_ratio, tf.int32)
    cy = tf.random.uniform(shape=[], minval=0, maxval=img_h, dtype=tf.int32)
    cx = tf.random.uniform(shape=[], minval=0, maxval=img_w, dtype=tf.int32)
    y1, y2 = tf.maximum(0, cy - cut_h // 2), tf.minimum(img_h, cy + cut_h // 2)
    x1, x2 = tf.maximum(0, cx - cut_w // 2), tf.minimum(img_w, cx + cut_w // 2)
    cut_region = tf.zeros([y2 - y1, x2 - x1, 1])
    mask = tf.pad(cut_region, [[y1, img_h - y2], [x1, img_w - x2], [0, 0]], constant_values=1.0)
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = images * mask + tf.gather(images, indices) * (1.0 - mask)
    actual_lam = 1.0 - tf.cast((y2 - y1) * (x2 - x1), tf.float32) / tf.cast(img_h * img_w, tf.float32)
    mixed_labels = actual_lam * labels + (1.0 - actual_lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels


# =============================================================================
# DATASET BUILDERS
# =============================================================================

def build_dataset_no_aug(raw_ds, batch_size=BATCH_SIZE, shuffle=True):
    """Exp 0: No augmentation. Also used for val/test in ALL experiments."""
    ds = raw_ds
    if shuffle:
        ds = ds.shuffle(buffer_size=2000, seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_dataset_traditional_aug(raw_ds, batch_size=BATCH_SIZE):
    """Exp 1: Traditional augmentation (flip, rotate, zoom, contrast)."""
    ds = raw_ds.shuffle(buffer_size=2000, seed=SEED)
    ds = ds.map(apply_traditional_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_dataset_mixup_cutmix(raw_ds, batch_size=BATCH_SIZE, mixup_prob=0.5):
    """Exp 2: Traditional aug + MixUp/CutMix (50/50 random per batch)."""
    ds = raw_ds.shuffle(buffer_size=2000, seed=SEED)
    ds = ds.map(apply_traditional_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    def apply_mixing(images, labels):
        choice = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
        return tf.cond(choice < mixup_prob,
                       lambda: mixup_batch(images, labels),
                       lambda: cutmix_batch(images, labels))
    return ds.map(apply_mixing, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# =============================================================================
# MODEL BUILDER: ResNet50
# =============================================================================

def build_resnet50(num_classes=NUM_CLASSES, dropout=DROPOUT):
    """
    ResNet50 + custom head: GlobalAvgPool → Dense(256) → Dropout → Dense(7, softmax).
    Returns (model, base_model) with base FROZEN for Phase A.
    """
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name="ResNet50_SkinLesion")
    print(f"ResNet50 built. Base layers: {len(base_model.layers)}, "
          f"Total params: {model.count_params():,}")
    return model, base_model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def compile_for_phase_a(model, lr=PHASE_A_LR):
    """Compile for Phase A: feature extraction with frozen base."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def unfreeze_for_phase_b(model, base_model, lr=PHASE_B_LR, unfreeze_pct=0.30):
    """Unfreeze top N% of base layers and re-compile with low learning rate."""
    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = int(total_layers * (1.0 - unfreeze_pct))

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"Fine-tuning: unfreezing top {trainable_count}/{total_layers} layers "
          f"({unfreeze_pct*100:.0f}%)")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(experiment_name, phase):
    """Standard callbacks: early stopping, LR reduction, model checkpoint."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=3, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"{experiment_name}_{phase}_best.keras"),
            monitor='val_loss', save_best_only=True, verbose=0
        ),
    ]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, test_ds, experiment_name, class_names=CLASS_NAMES):
    """
    Full evaluation on test set: classification report, confusion matrix,
    per-class AUC-ROC curves. Saves plots and JSON summary to RESULTS_DIR.
    Returns a summary dict.
    """
    # Collect predictions
    y_true, y_pred_probs = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.append(labels.numpy())
        y_pred_probs.append(preds)

    y_true = np.concatenate(y_true, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)

    # 1. Classification Report
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {experiment_name}")
    print(f"{'='*60}\n")
    report = classification_report(
        y_true_idx, y_pred_idx, target_names=class_names, digits=4, output_dict=True
    )
    print(classification_report(y_true_idx, y_pred_idx, target_names=class_names, digits=4))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix — {experiment_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{experiment_name}_confusion_matrix.png"), dpi=150)
    plt.show()

    # 3. Per-Class AUC-ROC
    auc_scores = {}
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(class_names):
        if len(np.unique(y_true[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
            auc_val = auc(fpr, tpr)
            auc_scores[cls] = auc_val
            ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.3f})")
        else:
            auc_scores[cls] = float('nan')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves — {experiment_name}', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{experiment_name}_roc_curves.png"), dpi=150)
    plt.show()

    # 4. Summary
    overall_accuracy = np.mean(y_true_idx == y_pred_idx)
    macro_f1 = f1_score(y_true_idx, y_pred_idx, average='macro')
    macro_precision = precision_score(y_true_idx, y_pred_idx, average='macro')
    macro_recall = recall_score(y_true_idx, y_pred_idx, average='macro')
    mel_recall = report['mel']['recall']
    mel_f1 = report['mel']['f1-score']
    valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
    mean_auc = np.mean(valid_aucs) if valid_aucs else float('nan')

    summary = {
        'experiment': experiment_name,
        'accuracy': overall_accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'mel_recall': mel_recall,
        'mel_f1': mel_f1,
        'mean_auc': mean_auc,
        'per_class_auc': auc_scores,
        'per_class_f1': {cls: report[cls]['f1-score'] for cls in class_names},
        'per_class_recall': {cls: report[cls]['recall'] for cls in class_names},
        'per_class_precision': {cls: report[cls]['precision'] for cls in class_names},
    }

    print(f"\n--- KEY METRICS ---")
    print(f"  Overall Accuracy:  {overall_accuracy:.4f}")
    print(f"  Macro F1-Score:    {macro_f1:.4f}")
    print(f"  Macro Recall:      {macro_recall:.4f}")
    print(f"  Mean AUC-ROC:      {mean_auc:.4f}")
    print(f"  Melanoma Recall:   {mel_recall:.4f}  ← CRITICAL")
    print(f"  Melanoma F1:       {mel_f1:.4f}")

    # Save JSON
    save_summary = {}
    for k, v in summary.items():
        if isinstance(v, (np.floating, np.integer)):
            save_summary[k] = float(v)
        elif isinstance(v, dict):
            save_summary[k] = {k2: float(v2) if isinstance(v2, (np.floating, np.integer)) else v2
                                for k2, v2 in v.items()}
        else:
            save_summary[k] = v

    with open(os.path.join(RESULTS_DIR, f"{experiment_name}_summary.json"), 'w') as f:
        json.dump(save_summary, f, indent=2)

    print(f"\n  Results saved to: {RESULTS_DIR}/{experiment_name}_*")
    return summary


# =============================================================================
# TRAINING HISTORY PLOT
# =============================================================================

def plot_training_history(history_a, history_b, experiment_name):
    """Plot loss and accuracy curves for both training phases."""
    loss = history_a.history['loss'] + history_b.history['loss']
    val_loss = history_a.history['val_loss'] + history_b.history['val_loss']
    acc = history_a.history['accuracy'] + history_b.history['accuracy']
    val_acc = history_a.history['val_accuracy'] + history_b.history['val_accuracy']
    phase_a_end = len(history_a.history['loss'])
    epochs = range(1, len(loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, loss, 'b-', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss')
    ax1.axvline(x=phase_a_end, color='gray', linestyle='--', alpha=0.7, label='Phase B Start')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'{experiment_name} — Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, acc, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_acc, 'r-', label='Val Accuracy')
    ax2.axvline(x=phase_a_end, color='gray', linestyle='--', alpha=0.7, label='Phase B Start')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{experiment_name} — Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{experiment_name}_training_curves.png"), dpi=150)
    plt.show()


# =============================================================================
# MASTER EXPERIMENT RUNNER
# =============================================================================

def run_experiment(experiment_name, train_ds, val_ds, test_ds,
                   build_model_fn, class_weights, use_class_weights=True,
                   architecture_name="ResNet50"):
    """
    Run a complete 2-phase training experiment with W&B logging and full evaluation.

    Args:
        experiment_name: e.g. "exp0_resnet50_no_aug"
        train_ds: training dataset (batched, prefetched)
        val_ds: validation dataset
        test_ds: test dataset
        build_model_fn: function returning (model, base_model)
        class_weights: dict for class_weight parameter
        use_class_weights: True for hard labels, False for soft labels (MixUp/CutMix)
        architecture_name: for W&B config logging

    Returns:
        model: trained model
        summary: evaluation results dict
    """
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"{'#'*60}\n")

    # Initialize W&B
    wandb.init(
        project="ham10000-capstone",
        name=experiment_name,
        config={
            "architecture": architecture_name,
            "phase_a_epochs": PHASE_A_EPOCHS, "phase_a_lr": PHASE_A_LR,
            "phase_b_epochs": PHASE_B_EPOCHS, "phase_b_lr": PHASE_B_LR,
            "batch_size": BATCH_SIZE, "dropout": DROPOUT,
            "patience": PATIENCE, "img_size": IMG_SIZE[0],
            "class_weights": use_class_weights,
        }
    )

    # Build Model
    model, base_model = build_model_fn()

    # PHASE A: Feature Extraction (Frozen Base)
    print("\n--- PHASE A: Feature Extraction (Frozen Base) ---")
    model = compile_for_phase_a(model, lr=PHASE_A_LR)
    cw = class_weights if use_class_weights else None

    history_a = model.fit(
        train_ds, validation_data=val_ds,
        epochs=PHASE_A_EPOCHS, class_weight=cw,
        callbacks=get_callbacks(experiment_name, "phaseA") + [WandbMetricsLogger()],
        verbose=1
    )

    # PHASE B: Fine-Tuning (Top 30% Unfrozen)
    print("\n--- PHASE B: Fine-Tuning (Top 30% Unfrozen) ---")
    model = unfreeze_for_phase_b(model, base_model, lr=PHASE_B_LR, unfreeze_pct=0.30)

    history_b = model.fit(
        train_ds, validation_data=val_ds,
        epochs=PHASE_B_EPOCHS, class_weight=cw,
        callbacks=get_callbacks(experiment_name, "phaseB") + [WandbMetricsLogger()],
        verbose=1
    )

    # Plot Training Curves
    plot_training_history(history_a, history_b, experiment_name)

    # Save Final Model
    model_path = os.path.join(MODELS_DIR, f"{experiment_name}_final.keras")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Full Evaluation on Test Set
    summary = evaluate_model(model, test_ds, experiment_name)

    # Log key metrics to W&B
    wandb.log({
        "test/accuracy": summary['accuracy'],
        "test/macro_f1": summary['macro_f1'],
        "test/macro_recall": summary['macro_recall'],
        "test/mean_auc": summary['mean_auc'],
        "test/mel_recall": summary['mel_recall'],
        "test/mel_f1": summary['mel_f1'],
    })

    wandb.finish()
    print(f"\n{'='*60}")
    print(f"  {experiment_name} COMPLETE")
    print(f"{'='*60}\n")

    return model, summary


# =============================================================================
# STAGE COMPARISON UTILITIES
# =============================================================================

def compare_experiments(summaries, stage_name, class_names=CLASS_NAMES):
    """
    Build comparison table and bar charts from a list of experiment summaries.
    Saves results to RESULTS_DIR.
    """
    # Comparison table
    comparison_data = []
    for s in summaries:
        comparison_data.append({
            'Experiment': s['experiment'],
            'Accuracy': f"{s['accuracy']:.4f}",
            'Macro F1': f"{s['macro_f1']:.4f}",
            'Macro Recall': f"{s['macro_recall']:.4f}",
            'Mean AUC': f"{s['mean_auc']:.4f}",
            'Mel Recall': f"{s['mel_recall']:.4f}",
            'Mel F1': f"{s['mel_f1']:.4f}",
        })
    comparison_df = pd.DataFrame(comparison_data)

    print(f"\n{'='*80}")
    print(f"  {stage_name} RESULTS COMPARISON")
    print(f"{'='*80}")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(os.path.join(RESULTS_DIR, f"{stage_name}_comparison.csv"), index=False)

    # Per-class F1 bar chart
    f1_data = {s['experiment']: s['per_class_f1'] for s in summaries}
    f1_df = pd.DataFrame(f1_data)
    print(f"\n--- Per-Class F1 Scores ---")
    print(f1_df.to_string())

    f1_df.plot(kind='bar', figsize=(12, 6))
    plt.title(f"Per-Class F1 Score — {stage_name}")
    plt.xlabel("Class"); plt.ylabel("F1 Score")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.legend(title="Experiment")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{stage_name}_f1_comparison.png"), dpi=150)
    plt.show()

    # Per-class Recall bar chart
    recall_data = {s['experiment']: s['per_class_recall'] for s in summaries}
    recall_df = pd.DataFrame(recall_data)
    print(f"\n--- Per-Class Recall (Sensitivity) ---")
    print(recall_df.to_string())

    recall_df.plot(kind='bar', figsize=(12, 6), colormap='Set2')
    plt.title(f"Per-Class Recall — {stage_name}")
    plt.xlabel("Class"); plt.ylabel("Recall (Sensitivity)")
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.legend(title="Experiment")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{stage_name}_recall_comparison.png"), dpi=150)
    plt.show()

    return comparison_df


def select_best_experiment(summaries, primary_metric='macro_f1'):
    """Select the best experiment based on primary metric. Returns (index, summary)."""
    values = [s[primary_metric] for s in summaries]
    best_idx = int(np.argmax(values))
    best = summaries[best_idx]

    print(f"\n{'='*60}")
    print(f"  BEST EXPERIMENT (by {primary_metric})")
    print(f"{'='*60}")
    print(f"  Winner:       {best['experiment']}")
    print(f"  Macro F1:     {best['macro_f1']:.4f}")
    print(f"  Mel Recall:   {best['mel_recall']:.4f}")
    print(f"  Mean AUC:     {best['mean_auc']:.4f}")

    return best_idx, best