from __future__ import annotations
import os, json, random, shutil, argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

def ensure_dirs():
    for d in ["data/interim", "data/processed", "models", "reports/figures"]:
        os.makedirs(d, exist_ok=True)

def split_raw_to_folders(raw_train_dir: str, out_root: str,
                         val_split: float = 0.2, test_split: float = 0.1):
    out_root = Path(out_root)
    for split in ["train", "val", "test"]:
        for cls in ["cat", "dog"]:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    cat_files = sorted([p for p in Path(raw_train_dir).glob("cat.*.jpg")])
    dog_files = sorted([p for p in Path(raw_train_dir).glob("dog.*.jpg")])

    def split_list(files):
        random.shuffle(files)
        n = len(files)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        test = files[:n_test]
        val = files[n_test:n_test+n_val]
        train = files[n_test+n_val:]
        return train, val, test

    cat_train, cat_val, cat_test = split_list(cat_files)
    dog_train, dog_val, dog_test = split_list(dog_files)

    plan = [
        ("train", cat_train, "cat"),
        ("val",   cat_val,   "cat"),
        ("test",  cat_test,  "cat"),
        ("train", dog_train, "dog"),
        ("val",   dog_val,   "dog"),
        ("test",  dog_test,  "dog"),
    ]

   
    for split, files, cls in plan:
        dst_dir = out_root / split / cls
        if any(dst_dir.iterdir()):
            continue
        for src in tqdm(files, desc=f"Copiando {split}/{cls}"):
            shutil.copy2(src, dst_dir / src.name)

def show_samples(folder: str, cls: str, n: int = 9):
    paths = list(Path(folder).glob(f"{cls}/*.jpg"))
    if len(paths) == 0:
        print(f"No hay imágenes en {folder}/{cls}")
        return

    sel = random.sample(paths, min(n, len(paths)))
    cols = 3
    rows = (len(sel) + cols - 1) // cols
    plt.figure(figsize=(10, 10))
    for i, p in enumerate(sel, 1):
        img = tf.keras.utils.load_img(p)
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(f"{cls} - {p.name}")
        plt.axis("off")
    plt.tight_layout()
    fig_path = Path("reports/figures") / f"samples_{Path(folder).parent.name}_{Path(folder).name}_{cls}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Guardada muestra de {cls}: {fig_path}")

def make_datagens(img_size=(200,200), batch_size=32, base_dir="data/interim/catsdogs"):
    train_dir = Path(base_dir) / "train"
    val_dir   = Path(base_dir) / "val"
    test_dir  = Path(base_dir) / "test"

    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        horizontal_flip=True
    )
    test_aug = ImageDataGenerator(rescale=1./255)

    trdata = train_aug.flow_from_directory(
        directory=str(train_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=SEED
    )
    valdata = test_aug.flow_from_directory(
        directory=str(val_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    tsdata = test_aug.flow_from_directory(
        directory=str(test_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    return trdata, valdata, tsdata

def build_vgg_like(input_shape=(200,200,3)):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train(args):
    ensure_dirs()

    
    split_raw_to_folders(
        raw_train_dir=args.raw_dir,
        out_root=args.work_dir,
        val_split=args.val_split,
        test_split=args.test_split
    )

    
    show_samples(os.path.join(args.work_dir, "train"), "dog", n=9)
    show_samples(os.path.join(args.work_dir, "train"), "cat", n=9)
    show_samples(os.path.join(args.work_dir, "test"), "dog", n=9)
    show_samples(os.path.join(args.work_dir, "test"), "cat", n=9)


    trdata, valdata, tsdata = make_datagens(
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        base_dir=args.work_dir
    )

    #Modelo
    model = build_vgg_like(input_shape=(args.img_size, args.img_size, 3))
    model.summary()


    best_path = "models/best_catsdogs.keras"
    cbs = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True)
    ]


    history = model.fit(
        trdata,
        epochs=args.epochs,
        validation_data=valdata,
        callbacks=cbs
    )

    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend(); plt.title("Loss"); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.savefig("reports/figures/loss.png", dpi=150); plt.close()

    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend(); plt.title("Accuracy"); plt.xlabel("epoch"); plt.ylabel("acc")
    plt.savefig("reports/figures/accuracy.png", dpi=150); plt.close()

    best_model = load_model(best_path)
    test_loss, test_acc = best_model.evaluate(tsdata, verbose=0)
    print(f"Test -> loss: {test_loss:.4f}, acc: {test_acc:.4f}")


    y_true = tsdata.classes
    class_indices = tsdata.class_indices  # p.ej: {'cat':0,'dog':1}
    idx_to_class = {v:k for k,v in class_indices.items()}

    y_prob = best_model.predict(tsdata, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(y_true, y_pred, target_names=[idx_to_class[0], idx_to_class[1]], output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "classification_report": report,
        "confusion_matrix": cm,
        "class_indices": class_indices
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Guardadas métricas en models/metrics.json")
    print(f"Mejor modelo guardado en {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/train", help="Carpeta con cat.* y dog.* juntos")
    parser.add_argument("--work-dir", default="data/interim/catsdogs", help="Dónde crear train/val/test por clase")
    parser.add_argument("--img-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    args = parser.parse_args()
    train(args)