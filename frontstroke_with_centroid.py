# -*- coding: utf-8 -*-
"""
frontstroke_with_centroid_improved_balanced.py
Final Version — Balanced Dataset + Improved Splits + Better Training Stability
Biomechanics, YOLO logic, tracking, and 150-frame processing remain unchanged.
"""

import os
import cv2
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


# =====================================================================
# DEVICE CHECK
# =====================================================================
print("Computing devices available:")
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

for gpu in gpus:
    print(f"  ✓ GPU: {gpu}")
    tf.config.experimental.set_memory_growth(gpu, True)

for cpu in cpus:
    print(f"  ✓ CPU: {cpu}")

if not gpus:
    print("⚠ No GPU found – running on CPU")


# =====================================================================
# PATH SETUP
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "vids")
CSV_PATH = os.path.join(BASE_DIR, "csvPoints.csv")
MODEL_PATH = os.path.join(BASE_DIR, "yolo11n-pose.pt")

print(f"✓ Using local workspace: {BASE_DIR}")


# =====================================================================
# FIND VIDEOS
# =====================================================================
def find_videos_locally():
    if not os.path.exists(VIDEOS_DIR):
        print(f"✖ Videos directory missing: {VIDEOS_DIR}")
        return [], None

    video_files = []
    for fname in os.listdir(VIDEOS_DIR):
        if fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.wmv')):
            path = os.path.join(VIDEOS_DIR, fname)
            size = os.path.getsize(path) / (1024 * 1024)
            video_files.append({"name": fname, "path": path, "size": size})

    print(f"✓ Found {len(video_files)} videos")
    return video_files, VIDEOS_DIR


# =====================================================================
# FILTER TO FRONTSTROKE OR NOT
# =====================================================================
def filter_frontstroke(videos):
    front = []
    other = []

    print("\nAnalyzing video names...")
    for vid in videos:
        name = vid["name"].lower()

        if any(kw in name for kw in ["freestyle", "front", "crawl", "free", "fc"]):
            print(f"  ✓ FRONTSTROKE: {vid['name']}")
            front.append(vid)
        else:
            print(f"  ⊘ OTHER: {vid['name']}")
            other.append(vid)

    print(f"\nTotal frontstroke videos: {len(front)}")
    print(f"Total non-frontstroke videos: {len(other)}")

    return front, other


print("\nSearching for videos...")
videos, _ = find_videos_locally()
frontstroke_videos, non_frontstroke_videos = filter_frontstroke(videos)


# =====================================================================
# STREAM VIDEO INTO MEMORY
# =====================================================================
def stream_video(video_path):
    try:
        with open(video_path, "rb") as f:
            return f.read()
    except:
        return None


def extract_poses(video_bytes, yolo_model, csv_coord=None, max_frames=150):
    """
    Multi-person tracking using CSV coordinates.
    Always selects the person whose centroid is closest to the CSV point.
    Falls back to previous-frame tracking for stability.
    """
    import tempfile
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4', prefix='frontstroke_')
    os.close(temp_fd)

    try:
        with open(temp_path, 'wb') as f:
            f.write(video_bytes)
    except Exception as e:
        print(f"Error writing temp file: {e}")
        return None

    cap = cv2.VideoCapture(temp_path)

    poses = []
    confidences = []
    tracked_idx = None
    prev_centroid = None
    count = 0

    # If CSV exists, use that coordinate; else None
    if csv_coord is not None:
        csv_x, csv_y = csv_coord
    else:
        csv_x, csv_y = None, None

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False)
        if not (results and results[0].keypoints is not None):
            count += 1
            continue

        keypoints = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Keypoint confidence
        if hasattr(results[0].keypoints, 'conf'):
            kp_conf = results[0].keypoints.conf.cpu().numpy()
        else:
            kp_conf = np.ones((len(keypoints), 17))

        # If persons detected
        if len(keypoints) > 0:

            # STEP 1 — match person via CSV coordinate
            if tracked_idx is None and csv_x is not None:
                min_d = 1e9
                best_idx = None

                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    d = np.hypot(cx - csv_x, cy - csv_y)
                    if d < min_d:
                        min_d = d
                        best_idx = i

                tracked_idx = best_idx
                if tracked_idx is not None:
                    x1, y1, x2, y2 = boxes[tracked_idx]
                    prev_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

            # STEP 2 — if no CSV or first detection failed, use index 0
            if tracked_idx is None:
                tracked_idx = 0
                x1, y1, x2, y2 = boxes[0]
                prev_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

            # STEP 3 — maintain tracking across frames
            if prev_centroid is not None:
                px, py = prev_centroid
                min_d = 1e9
                best_idx = tracked_idx

                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    d = np.hypot(cx - px, cy - py)
                    if d < min_d:
                        min_d = d
                        best_idx = i
                        new_centroid = (cx, cy)

                tracked_idx = best_idx
                prev_centroid = new_centroid

            # STEP 4 — append pose
            if tracked_idx < len(keypoints):
                avg_conf = np.mean(kp_conf[tracked_idx])
                if avg_conf > 0.3:  # keep only high confidence
                    poses.append(keypoints[tracked_idx])
                    confidences.append(avg_conf)

        count += 1

    cap.release()
    os.remove(temp_path)

    if not poses:
        return None

    return np.array(poses), np.array(confidences)



# =====================================================================
# FEATURE EXTRACTION (UNCHANGED)
# =====================================================================
def process_sequence(seq, confidences=None):
    try:
        if len(seq) < 2:
            return None

        features = []

        for frame_idx, kpts in enumerate(seq):
            nose = kpts[0]
            Ls, Rs = kpts[5], kpts[6]
            Lh, Rh = kpts[11], kpts[12]
            Lk, Rk = kpts[13], kpts[14]
            La, Ra = kpts[15], kpts[16]
            Lw, Rw = kpts[9], kpts[10]
            Le, Re = kpts[7], kpts[8]

            if np.all(Lh == 0) or np.all(Rh == 0) or np.all(Ls == 0) or np.all(Rs == 0):
                features.append(np.zeros(50))
                continue

            hip_center = (Lh + Rh) / 2
            shoulder_center = (Ls + Rs) / 2
            body_center = (hip_center + shoulder_center) / 2

            f = []

            f.extend([hip_center[0], hip_center[1]])
            f.extend([shoulder_center[0], shoulder_center[1]])

            torso_length = np.linalg.norm(shoulder_center - hip_center)
            shoulder_width = np.linalg.norm(Rs - Ls)
            hip_width = np.linalg.norm(Rh - Lh)

            f.append(torso_length)
            f.append(shoulder_width)
            f.append(hip_width)

            left_arm_ext = np.linalg.norm(Lw - Ls)
            right_arm_ext = np.linalg.norm(Rw - Rs)
            f.append(left_arm_ext)
            f.append(right_arm_ext)
            f.append(left_arm_ext / (right_arm_ext + 1e-6))

            f.append(np.linalg.norm(Le - Ls))
            f.append(np.linalg.norm(Re - Rs))

            f.append(np.linalg.norm(La - Lh))
            f.append(np.linalg.norm(Ra - Rh))
            f.append(abs(La[1] - Ra[1]))
            f.append(np.linalg.norm(La - Ra))

            torso_vec = shoulder_center - hip_center
            torso_angle = np.arctan2(torso_vec[0], torso_vec[1])
            f.append(torso_angle)

            shoulder_vec = Rs - Ls
            shoulder_angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
            f.append(shoulder_angle)

            def angle_between(a, b, c):
                if np.all(b == 0) or np.all(c == 0):
                    return 0
                u = b - a
                v = c - b
                cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6)
                return np.arccos(np.clip(cosang, -1, 1))

            left_elbow_angle = angle_between(Ls, Le, Lw)
            right_elbow_angle = angle_between(Rs, Re, Rw)
            f.append(left_elbow_angle)
            f.append(right_elbow_angle)
            f.append(abs(left_elbow_angle - right_elbow_angle))

            f.append(angle_between(Lh, Lk, La))
            f.append(angle_between(Rh, Rk, Ra))

            f.append(nose[0] - shoulder_center[0])
            f.append(nose[1] - shoulder_center[1])
            f.append(np.linalg.norm(nose - shoulder_center))

            f.append(np.linalg.norm(Lw - Rw))
            f.append(np.linalg.norm(Le - Re))
            f.append(np.linalg.norm(Ls - Rs))

            def col_score(p1, p2, p3):
                v1 = p2 - p1
                v2 = p3 - p2
                return abs(np.cross(v1, v2)) / (np.linalg.norm(v1) *
                                                np.linalg.norm(v2) + 1e-6)

            f.append(col_score(Ls, Lh, La))
            f.append(col_score(Rs, Rh, Ra))
            f.append(col_score(shoulder_center, hip_center, body_center))

            extremities = np.array([Lw, Rw, La, Ra])
            avg_spread = np.mean([np.linalg.norm(e - body_center)
                                  for e in extremities])
            f.append(avg_spread)

            f.append(Lw[1] - shoulder_center[1])
            f.append(Rw[1] - shoulder_center[1])

            f.append(Lw[0] - shoulder_center[0])
            f.append(Rw[0] - shoulder_center[0])

            if confidences is not None and frame_idx < len(confidences):
                f.append(confidences[frame_idx])
            else:
                f.append(1.0)

            while len(f) < 50:
                f.append(0)

            features.append(np.array(f[:50], dtype=np.float32))

        features = np.array(features)

        velocities = np.zeros_like(features)
        for i in range(1, len(features)):
            velocities[i] = features[i] - features[i - 1]

        combined = np.concatenate([features, velocities], axis=1)

        out = np.zeros_like(combined)
        for i in range(combined.shape[1]):
            col = combined[:, i]
            if col.std() > 0:
                out[:, i] = (col - col.mean()) / col.std()
            else:
                out[:, i] = col

        from scipy.ndimage import median_filter
        out = median_filter(out, size=(3, 1))

        return out

    except Exception as e:
        print("⚠ Error in process_sequence:", e)
        return None


# =====================================================================
# LOAD YOLO MODEL + CSV TRACKING
# =====================================================================
print("\nLoading YOLO pose model...")
yolo_model = YOLO(MODEL_PATH)

print("\nLoading tracking CSV...")
csv_tracking = {}

try:
    df = pd.read_csv(CSV_PATH)
    for _, row in df.iterrows():
        video_id = str(row[0]).strip()
        pt = str(row[2])
        x, y = [int(v.strip()) for v in pt.replace("(", "").replace(")", "").split(",")]
        csv_tracking[video_id] = (x, y)
    print(f"✓ CSV tracking loaded for {len(csv_tracking)} videos")
except:
    print("⚠ CSV not found – tracking first person")


# =====================================================================
# EXTRACT POSES FOR ALL VIDEOS
# =====================================================================
all_poses = []
all_labels = []

print("\n================ FRONTSTROKE VIDEOS ================")
for i, vid in enumerate(frontstroke_videos):
    print(f"[{i + 1}/{len(frontstroke_videos)}] {vid['name']} ... ", end="")

    coord = None
    for key in csv_tracking:
        if key in vid["name"]:
            coord = csv_tracking[key]
            print(f"(CSV: {coord}) ", end="")

    data = stream_video(vid["path"])
    if data is None:
        print("✖ Read error")
        continue

    result = extract_poses(data, yolo_model, coord)
    if result is None:
        print("⚠ No poses detected")
        continue

    poses, confidences = result
    proc = process_sequence(poses, confidences)

    if proc is not None:
        all_poses.append(proc)
        all_labels.append(1)
        print(f"✓ {proc.shape}")
    else:
        print("⚠ Processing failed")


print("\n================ NON-FRONTSTROKE VIDEOS ================")
for i, vid in enumerate(non_frontstroke_videos):
    print(f"[{i + 1}/{len(non_frontstroke_videos)}] {vid['name']} ... ", end="")

    coord = None
    for key in csv_tracking:
        if key in vid["name"]:
            coord = csv_tracking[key]

    data = stream_video(vid["path"])
    if data is None:
        print("✖ Read error")
        continue

    result = extract_poses(data, yolo_model, coord)
    if result is None:
        print("⚠ No poses detected")
        continue

    poses, confidences = result
    proc = process_sequence(poses, confidences)

    if proc is not None:
        all_poses.append(proc)
        all_labels.append(0)
        print(f"✓ {proc.shape}")
    else:
        print("⚠ Processing failed")


print(f"\nTotal usable sequences: {len(all_poses)}")


# =====================================================================
# BALANCE DATASET
# =====================================================================
print("\n⚖ Balancing dataset...")

idx_front = [i for i, lbl in enumerate(all_labels) if lbl == 1]
idx_nonfront = [i for i, lbl in enumerate(all_labels) if lbl == 0]

min_count = min(len(idx_front), len(idx_nonfront))

np.random.shuffle(idx_front)
np.random.shuffle(idx_nonfront)

final_idx = idx_front[:min_count] + idx_nonfront[:min_count]
np.random.shuffle(final_idx)

balanced_poses = [all_poses[i] for i in final_idx]
balanced_labels = [all_labels[i] for i in final_idx]

print(f"Final balanced dataset → {len(balanced_labels)} samples")


# =====================================================================
# PAD SEQUENCES (150 × 100)
# =====================================================================
max_len = 150
num_features = 100

X, y = [], []

for seq, label in zip(balanced_poses, balanced_labels):
    if seq.shape[1] != num_features:
        continue

    if len(seq) < max_len:
        seq = np.pad(seq, ((0, max_len - len(seq)), (0, 0)), constant_values=-1)
    else:
        seq = seq[:max_len]

    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Dataset shape → {X.shape}")


# =====================================================================
# SPLIT TRAIN / VAL / TEST
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

print("\nSplits:")
print("Train:", len(X_train))
print("Val:", len(X_val))
print("Test:", len(X_test))


# =====================================================================
# AUGMENT TRAINING DATA (same logic as sitting/standing)
# =====================================================================
X_aug_total = [X_train]
y_aug_total = [y_train]

for aug_idx in range(4):
    X_aug = np.zeros_like(X_train)

    for i, seq in enumerate(X_train):
        aug = seq.copy()

        if aug_idx == 0:
            aug += np.random.normal(0, 0.05, aug.shape)
        elif aug_idx == 1:
            aug += np.random.normal(0, 0.10, aug.shape)
        elif aug_idx == 2:
            idx = np.random.choice(np.arange(len(aug)), size=len(aug), replace=True)
            aug = aug[np.sort(idx)]
        else:
            mask = np.random.choice([0, 1], size=aug.shape, p=[0.1, 0.9])
            aug = aug * mask

        X_aug[i] = np.clip(aug, -10, 10)

    X_aug_total.append(X_aug)
    y_aug_total.append(y_train)

X_train = np.vstack(X_aug_total)
y_train = np.hstack(y_aug_total)

print("\nAugmented training set:", X_train.shape)


# =====================================================================
# MODEL ARCHITECTURE
# =====================================================================
def build_lstm_model(max_len=150, num_features=100, num_classes=2):
    inp = layers.Input(shape=(max_len, num_features))
    x = layers.Masking(mask_value=-1.0)(inp)
    x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(
        layers.LSTM(384, return_sequences=True, dropout=0.3, recurrent_dropout=0.15)
    )(x)
    x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(
        layers.LSTM(192, return_sequences=True, dropout=0.3, recurrent_dropout=0.15)
    )(x)
    x = layers.LayerNormalization()(x)

    x = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    )(x)
    x = layers.LayerNormalization()(x)

    att = layers.Dense(1, activation='tanh')(x)
    att = layers.Flatten()(att)
    att = layers.Softmax()(att)
    att = layers.RepeatVector(192)(att)
    att = layers.Permute((2, 1))(att)

    x = layers.Multiply()([x, att])
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


model = build_lstm_model()
print(model.summary())


# =====================================================================
# CLASS WEIGHTS
# =====================================================================
from sklearn.utils.class_weight import compute_class_weight
cw_vals = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw = {0: cw_vals[0], 1: cw_vals[1]}


# =====================================================================
# TRAINING — (patience=5, min_delta=0.03)
# =====================================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    class_weight=cw,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode='max',
            min_delta=0.03,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_frontstroke.keras",
            save_best_only=True,
            monitor="val_accuracy"
        )
    ]
)


# =====================================================================
# TEST EVALUATION
# =====================================================================
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

pred = np.argmax(model.predict(X_test), axis=1)

print("\nPrediction breakdown:")
print(f"Frontstroke correct: {((pred == 1) & (y_test == 1)).sum()}/{(y_test == 1).sum()}")
print(f"Non-frontstroke correct: {((pred == 0) & (y_test == 0)).sum()}/{(y_test == 0).sum()}")

from sklearn.metrics import confusion_matrix, classification_report

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=['Non-Frontstroke', 'Frontstroke']))


# =====================================================================
# SAVE MODEL
# =====================================================================
model.save("frontstroke_lstm_model_improved.keras")
print("\nModel saved → frontstroke_lstm_model_improved.keras")

