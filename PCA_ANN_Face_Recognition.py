import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.linalg import eigh

DATASET_PATH = r"D:\3. Internships\iSTUDIO AI internship\Project - Face recognition\dataset\dataset\faces"
IMG_SIZE = (64, 64)
faces = []
labels = []
label_id = 0
print("Loading dataset...")
for person in sorted(os.listdir(DATASET_PATH)):
    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float64)
        faces.append(img.flatten())
        labels.append(label_id)
    label_id += 1
faces = np.array(faces)
labels = np.array(labels)
print("Images:", len(faces))
print("Classes:", len(np.unique(labels)))
X_train, X_test, y_train, y_test = train_test_split(
    faces,
    labels,
    test_size=0.40,
    random_state=42,
    stratify=labels
)
Face_Db = X_train.T
print("\nFace Database Shape:", Face_Db.shape)
mean_face = np.mean(Face_Db, axis=1, keepdims=True)
Delta = Face_Db - mean_face
C = Delta.T @ Delta
print("Covariance Shape:", C.shape)
eigenvalues, eigenvectors = eigh(C)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
k_values = [5,10,15,20,25,30,40,50,75,100]
accuracies = []
best_accuracy = 0
best_k = 0
for k in k_values:
    print(f"\nTesting k = {k}")
    selected_vectors = eigenvectors[:, :k]
    eigenfaces = Delta @ selected_vectors
    eigenfaces = eigenfaces / (
        np.linalg.norm(eigenfaces, axis=0) + 1e-10
    )
    train_features = eigenfaces.T @ Delta
    test_delta = X_test.T - mean_face
    test_features = eigenfaces.T @ test_delta
    clf = MLPClassifier(
        hidden_layer_sizes=(512,256),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
    clf.fit(train_features.T, y_train)
    predictions = clf.predict(test_features.T)
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)
    print("Accuracy =", accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
print("\n===========================")
print("BEST K =", best_k)
print("BEST ACCURACY =", best_accuracy)
print("===========================")
plt.figure(figsize=(8,5))
plt.plot(
    k_values,
    accuracies,
    marker='o'
)
plt.xlabel("k Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.grid(True)
plt.show()
selected_vectors = eigenvectors[:, :best_k]
eigenfaces = Delta @ selected_vectors
eigenfaces = eigenfaces / (
    np.linalg.norm(eigenfaces, axis=0) + 1e-10
)
train_features = eigenfaces.T @ Delta
test_delta = X_test.T - mean_face
test_features = eigenfaces.T @ test_delta
final_ann = MLPClassifier(
    hidden_layer_sizes=(512,256),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    early_stopping=True
)
final_ann.fit(train_features.T, y_train)
predictions = final_ann.predict(test_features.T)
final_accuracy = accuracy_score(
    y_test,
    predictions
)
print("\nFINAL ACCURACY =", final_accuracy)
print("\nTesting Imposter Detection...")
random_noise = np.random.randint(
    0,
    255,
    (64*64,1)
).astype(np.float64)
noise_delta = random_noise - mean_face
noise_feature = eigenfaces.T @ noise_delta
prob = np.max(
    final_ann.predict_proba(
        noise_feature.T
    )
)
if prob < 0.50:
    print("NOT ENROLLED PERSON")
else:
    print("ENROLLED PERSON")