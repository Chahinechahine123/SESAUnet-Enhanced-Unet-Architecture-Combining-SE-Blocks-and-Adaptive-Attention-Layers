import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision



H = 512
W = 512

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


    return x, y
def tf_parse(x, y, augment=True):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)

        if augment:
            # 1. Random rotation
            k = np.random.randint(0, 4)  # 0,1,2,3
            x = np.rot90(x, k)
            y = np.rot90(y, k)

            # 2. Random horizontal flip
            if np.random.rand() < 0.5:
                x = np.fliplr(x)
                y = np.fliplr(y)

            # 3. Random vertical flip
            if np.random.rand() < 0.5:
                x = np.flipud(x)
                y = np.flipud(y)

            # 4. Random contrast adjustment
            factor = 0.8 + np.random.rand() * 0.4  # 0.8 - 1.2
            x = np.clip(x * factor, 0, 1)

        return x.astype(np.float32), y.astype(np.float32)

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(x, y, batch=8, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(lambda xi, yi: tf_parse(xi, yi, augment=augment))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("/content/drive/MyDrive/SESAUnetF2")

    """ Hyperparameters """
    batch_size = 4
    lr = 1e-4
    num_epochs = 100
    model_path = os.path.join("/content/drive/MyDrive/SESAUnetF2", "SESAUnetF.h5")
    csv_path = os.path.join("/content/drive/MyDrive/SESAUnetF2", "data1.csv")

    """ Dataset """
    dataset_path = os.path.join("/content/data")
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "val")

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size, augment=True)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size, augment=False)

    """ Model """
    model = build_unet((H, W, 3))
    from tensorflow.keras.metrics import Recall, Precision

    recall = Recall()
    precision = Precision()
    metrics = [iou,Recall(), Precision()]
    model.compile(loss=dice_loss, optimizer=Adam(learning_rate=lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        shuffle=False
    )