
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
#  CONFIG  — change these paths / params only
# ─────────────────────────────────────────────
DATASET_PATH = r"D:\3. Internships\iSTUDIO AI internship\Project - Face recognition\dataset\dataset\faces"
IMG_SIZE     = (64, 64)          # every image resized to this
TRAIN_RATIO  = 0.60              # 60 % train, 40 % test (per PDF)
N_IMPOSTERS  = 2                 # last N folders (alphabetically) → imposter set
K_VALUES     = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]  # k sweep for part (a)
THRESHOLD    = 0.50              # confidence below this → "NOT ENROLLED"
RANDOM_STATE = 42
# ─────────────────────────────────────────────


# ==============================================================
# STEP 0 ─ Load images, split enrolled vs imposter persons
# ==============================================================
def load_dataset(dataset_path, img_size):
    """Return (faces_flat, labels, person_names) as numpy arrays."""
    faces, labels = [], []
    persons = sorted(os.listdir(dataset_path))
    person_dirs = [p for p in persons
                   if os.path.isdir(os.path.join(dataset_path, p))]

    label_id = 0
    for person in person_dirs:
        person_path = os.path.join(dataset_path, person)
        loaded = 0
        for fname in sorted(os.listdir(person_path)):
            fpath = os.path.join(person_path, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size).astype(np.float64)
            faces.append(img.flatten())   # flatten → column vector (mn,)
            labels.append(label_id)
            loaded += 1
        if loaded > 0:
            label_id += 1

    return np.array(faces), np.array(labels), person_dirs


print("=" * 60)
print("  FACE RECOGNITION — PCA + ANN")
print("=" * 60)

all_persons = sorted([
    p for p in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, p))
])

enrolled_persons = all_persons[:-N_IMPOSTERS]   # training persons
imposter_persons = all_persons[-N_IMPOSTERS:]   # never-seen persons

print(f"\nEnrolled persons  ({len(enrolled_persons)}): {enrolled_persons}")
print(f"Imposter persons  ({len(N_IMPOSTERS if isinstance(N_IMPOSTERS,list) else [N_IMPOSTERS])}): {imposter_persons}")

# ── Load enrolled faces ──────────────────────────────────────
enrolled_faces, enrolled_labels = [], []
for lbl, person in enumerate(enrolled_persons):
    person_path = os.path.join(DATASET_PATH, person)
    for fname in sorted(os.listdir(person_path)):
        fpath = os.path.join(person_path, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE).astype(np.float64)
        enrolled_faces.append(img.flatten())
        enrolled_labels.append(lbl)

enrolled_faces  = np.array(enrolled_faces)   # shape: (p_total, mn)
enrolled_labels = np.array(enrolled_labels)

print(f"\nTotal enrolled images : {len(enrolled_faces)}")
print(f"Feature dimension (mn): {enrolled_faces.shape[1]}")

# ── Load imposter faces ──────────────────────────────────────
imposter_faces = []
for person in imposter_persons:
    person_path = os.path.join(DATASET_PATH, person)
    for fname in sorted(os.listdir(person_path)):
        fpath = os.path.join(person_path, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE).astype(np.float64)
        imposter_faces.append(img.flatten())

imposter_faces = np.array(imposter_faces)
print(f"Total imposter images : {len(imposter_faces)}")


# ==============================================================
# STEP 1 ─ 60/40 train/test split on ENROLLED data
# ==============================================================
# Stratified manual split (each person contributes to both sets)
np.random.seed(RANDOM_STATE)
train_idx, test_idx = [], []

for lbl in np.unique(enrolled_labels):
    idx = np.where(enrolled_labels == lbl)[0]
    np.random.shuffle(idx)
    n_train = max(1, int(len(idx) * TRAIN_RATIO))
    train_idx.extend(idx[:n_train])
    test_idx.extend(idx[n_train:])

X_train = enrolled_faces[train_idx]   # (p_train, mn)
y_train = enrolled_labels[train_idx]
X_test  = enrolled_faces[test_idx]    # (p_test,  mn)
y_test  = enrolled_labels[test_idx]

print(f"\nTrain set: {len(X_train)} images")
print(f"Test  set: {len(X_test)}  images")


# ==============================================================
# STEP 2 ─ Build Face Database as mn × p matrix
# ==============================================================
Face_Db = X_train.T           # shape: mn × p_train   (PDF notation)
mn, p   = Face_Db.shape
print(f"\nFace_Db shape (mn × p): {Face_Db.shape}")


# ==============================================================
# STEP 3 ─ Mean Face  M (mn × 1)
# ==============================================================
M = np.mean(Face_Db, axis=1, keepdims=True)   # shape: mn × 1
print(f"Mean face shape        : {M.shape}")


# ==============================================================
# STEP 4 ─ Mean-Zero faces  Δ (mn × p)
# ==============================================================
Delta = Face_Db - M           # broadcast: mn × p
print(f"Delta (mean-zero) shape: {Delta.shape}")


# ==============================================================
# STEP 5 ─ Surrogate Covariance  C (p × p)
# ==============================================================
# C = Δ^T · Δ   →   shape p × p   (Turk & Pentland 1991)
C = Delta.T @ Delta
print(f"Surrogate Cov shape    : {C.shape}")


# ==============================================================
# STEP 6 ─ Eigen decomposition of C
# ==============================================================
eigenvalues, eigenvectors = eigh(C)   # eigenvectors: p × p columns

# Sort descending by eigenvalue
sort_idx    = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_idx]
eigenvectors = eigenvectors[:, sort_idx]   # FV candidates: p × p

print(f"Eigenvectors shape     : {eigenvectors.shape}")


# ==============================================================
# Helper: build PCA model for a given k
# ==============================================================
def build_pca(k):
    """
    Returns (eigenfaces, train_signatures) for k components.

    Eigenfaces  Ψ  : k × mn   (PDF Step 7)
    Signatures  S  : k × p    (PDF Step 8)
    """
    # Step 7: Feature vector FV → p × k
    FV = eigenvectors[:, :k]               # p × k

    # Eigenfaces: project mean-zero faces to mn-space
    # Ψ (k × mn) = FV^T (k × p) · Δ^T (p × mn)
    Psi = FV.T @ Delta.T                   # k × mn

    # Normalise each eigenface (row)
    norms = np.linalg.norm(Psi, axis=1, keepdims=True) + 1e-10
    Psi   = Psi / norms                    # k × mn

    # Step 8: Signatures S (k × p) = Ψ (k × mn) · Δ (mn × p)
    S = Psi @ Delta                        # k × p

    return Psi, S


# ==============================================================
# STEP 9a ─ Accuracy vs k  
# ==============================================================
print("\n" + "=" * 60)
print("  PART (a) — Accuracy vs k")
print("=" * 60)

accuracies = []

for k in K_VALUES:
    # Skip if k > p
    if k > p:
        print(f"k={k:4d}  -> skipped (k > p={p})")
        accuracies.append(None)
        continue

    Psi, S_train = build_pca(k)

    # Project test set
    test_delta   = X_test.T - M            # mn × p_test
    S_test       = Psi @ test_delta        # k  × p_test

    # ANN (MLP with backpropagation)
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1
    )
    clf.fit(S_train.T, y_train)            # inputs: (p_train, k)

    preds    = clf.predict(S_test.T)
    accuracy = accuracy_score(y_test, preds)
    accuracies.append(accuracy)
    print(f"k={k:4d}  ->  Accuracy = {accuracy:.4f}  ({accuracy*100:.2f}%)")

# Filter valid results
valid_k   = [k for k, a in zip(K_VALUES, accuracies) if a is not None]
valid_acc = [a for a in accuracies if a is not None]

best_k   = valid_k[int(np.argmax(valid_acc))]
best_acc = max(valid_acc)
print(f"\n*  Best k = {best_k}  with Accuracy = {best_acc:.4f}  ({best_acc*100:.2f}%)")

# Plot accuracy vs k
plt.figure(figsize=(9, 5))
plt.plot(valid_k, [a * 100 for a in valid_acc], marker='o',
         linewidth=2, markersize=7, color='steelblue')
plt.xlabel("k  (number of eigenfaces)", fontsize=13)
plt.ylabel("Classification Accuracy (%)", fontsize=13)
plt.title("PCA + ANN Face Recognition — Accuracy vs k", fontsize=14)
plt.xticks(valid_k)
plt.grid(True, linestyle='--', alpha=0.6)
for kv, av in zip(valid_k, valid_acc):
    plt.annotate(f"{av*100:.1f}%", (kv, av * 100),
                 textcoords="offset points", xytext=(0, 8),
                 ha='center', fontsize=8)
plt.tight_layout()
plt.savefig("accuracy_vs_k.png", dpi=150)
plt.show()
print("Plot saved -> accuracy_vs_k.png")


# ==============================================================
# Final model trained with best k
# ==============================================================
print("\n" + "=" * 60)
print(f"  FINAL MODEL  (k = {best_k})")
print("=" * 60)

Psi_best, S_train_best = build_pca(best_k)

test_delta  = X_test.T - M
S_test_best = Psi_best @ test_delta

final_ann = MLPClassifier(
    hidden_layer_sizes=(512, 256),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.1
)
final_ann.fit(S_train_best.T, y_train)

preds_test   = final_ann.predict(S_test_best.T)
final_acc    = accuracy_score(y_test, preds_test)

print(f"\nFinal Test Accuracy : {final_acc:.4f}  ({final_acc*100:.2f}%)")
print("\nClassification Report (enrolled test set):")
print(classification_report(
    y_test, preds_test,
    target_names=[enrolled_persons[i] for i in np.unique(y_test)]
))


# ==============================================================
# STEP 10 ─ Imposter Detection  
# ==============================================================
print("=" * 60)
print("  PART (b) — Imposter Detection")
print("=" * 60)
print(f"Imposter persons: {imposter_persons}")
print(f"Imposter images : {len(imposter_faces)}")
print(f"Confidence threshold: {THRESHOLD}")

def classify_face(image_flat, M, Psi, ann_model, threshold):
    """
    Returns (label_or_None, confidence).
    label = None means IMPOSTER.
    """
    # Testing Step 2: mean-zero
    I2 = image_flat.reshape(-1, 1) - M         # mn × 1

    # Testing Step 3: project to eigenfaces
    S_test_single = Psi @ I2                   # k × 1

    # Testing Step 4: predict with ANN
    proba = ann_model.predict_proba(S_test_single.T)[0]
    conf  = np.max(proba)
    pred  = np.argmax(proba)

    if conf < threshold:
        return None, conf
    return pred, conf


# ── Evaluate on known test faces ────
print("\n[A] Known enrolled test faces:")
enrolled_correct = 0
for i, face in enumerate(X_test):
    label, conf = classify_face(face, M, Psi_best, final_ann, THRESHOLD)
    if label is not None:
        enrolled_correct += 1

enrolled_accuracy = enrolled_correct / len(X_test)
print(f"    Correctly identified as ENROLLED : {enrolled_correct}/{len(X_test)}  ({enrolled_accuracy*100:.1f}%)")

# ── Evaluate on imposter faces (should all be NOT ENROLLED) ──
print("\n[B] Imposter faces (never seen during training):")
imposter_correct = 0
imposter_results = []

for i, face in enumerate(imposter_faces):
    label, conf = classify_face(face, M, Psi_best, final_ann, THRESHOLD)
    is_imposter = (label is None)
    imposter_correct += int(is_imposter)
    imposter_results.append((is_imposter, conf))

imposter_detection_rate = imposter_correct / max(len(imposter_faces), 1)
print(f"    Correctly rejected as NOT ENROLLED: {imposter_correct}/{len(imposter_faces)}  ({imposter_detection_rate*100:.1f}%)")

# Detailed breakdown per imposter
for idx, (person, (is_imp, conf)) in enumerate(
        zip(imposter_persons * (len(imposter_results) // max(len(imposter_persons),1) + 1),
            imposter_results)):
    status = "ACCEPTED" if not is_imp else "REJECTED"
    note = "NOT ENROLLED" if is_imp else "FALSELY ACCEPTED"
    print(f"    Image {idx+1:3d} | Confidence {conf:.3f} | {status} ({note})")

# ── Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Best k                     : {best_k}")
print(f"  Enrolled test accuracy     : {final_acc*100:.2f}%")
print(f"  Enrolled detection rate    : {enrolled_accuracy*100:.1f}%")
print(f"  Imposter rejection rate    : {imposter_detection_rate*100:.1f}%")
print(f"  Confidence threshold used  : {THRESHOLD}")
print("=" * 60)


# ==============================================================
# Visualise a few eigenfaces 
# ==============================================================
print("\nVisualising top eigenfaces...")
n_show  = min(10, best_k)
fig, axes = plt.subplots(1, n_show, figsize=(2 * n_show, 2.5))
mn_pixels = IMG_SIZE[0] * IMG_SIZE[1]

for i in range(n_show):
    ef = Psi_best[i, :mn_pixels].reshape(IMG_SIZE)
    ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-10)
    axes[i].imshow(ef_norm, cmap='gray')
    axes[i].set_title(f"EF {i+1}", fontsize=8)
    axes[i].axis('off')

plt.suptitle(f"Top {n_show} Eigenfaces  (k={best_k})", fontsize=12)
plt.tight_layout()
plt.savefig("eigenfaces.png", dpi=150)
plt.show()
print("Eigenfaces saved -> eigenfaces.png")
