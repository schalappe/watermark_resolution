# -*- coding: utf-8 -*-
"""
Script used to fine-tune a model
"""
from os.path import join

import tensorflow as tf
from rich.progress import Progress

from src.addons.images import random_binary_images
from src.addons.losses import WatermarkLoss
from src.addons.metrics import BER, PSNR
from src.addons.optimizers import GCAdam
from src.config import DIMS_IMAGE, DIMS_MARK, MODEL_PATH, TEST_PATH, TRAIN_PATH
from src.data import prepare_data_from_slice
from src.model import ExtractWaterMark, WaterMark
from src.processors import preprocess_output, random_augmentation
from src.visualizations import print_tables

# ############ WARM UP #############

# Hyper-Parameters
BS = 32  # 100
EPOCHS = 50  # 4000

LR_embedding = 1e-4
LR_extraction = 1e-5

strength = 1.0
strength_embedding_mse = 45.0  # 45.0
strength_embedding_mae = 0.2
strength_extraction_mae = 20  # 20

# print hyper-paremeters
print("\n[INFO] Hyper-parameters for this training ...")
headers = ["Batch", "Epochs", "LR_embedding", "LR_extraction", "s", "λ_1", "λ_2", "λ_3"]
contents = [
    [
        BS,
        EPOCHS,
        LR_embedding,
        LR_extraction,
        strength,
        strength_embedding_mse,
        strength_embedding_mae,
        strength_extraction_mae,
    ]
]
print_tables(title="Hyper-parameters", headers=headers, contents=contents)

# load dataset
print("\n[INFO]: Load dataset ...")
train_set, train_size = prepare_data_from_slice(
    inputs_path=TRAIN_PATH, extension="png", batch_size=BS
)
test_set, test_size = prepare_data_from_slice(
    inputs_path=TEST_PATH, extension="png", batch_size=BS, training=False
)

# construct our model
print("\n[INFO]: Create model ...")
embedding_model = WaterMark.build(
    image_dims=DIMS_IMAGE, mark_dims=DIMS_MARK, strength=strength
)

extraction_model = ExtractWaterMark.build(mark_dims=DIMS_MARK)

# Instantiate optimizers.
print("\n[INFO]: Instantiate optimizers ...")
# embedding_optimizer = tf.keras.optimizers.Adam(
#     learning_rate=LR_embedding, decay=0.01/EPOCHS
# )
# extraction_optimizer = tf.keras.optimizers.Adam(
#     learning_rate=LR_extraction, decay=0.01/EPOCHS
# )
embedding_optimizer = GCAdam(learning_rate=LR_embedding, decay=0.01 / EPOCHS)
extraction_optimizer = GCAdam(learning_rate=LR_extraction, decay=0.01 / EPOCHS)

# Instantiate a loss function.
print("\n[INFO]: Instantiate a loss function ...")
loss_fn = WatermarkLoss(
    strength_embedding_mse=strength_embedding_mse,
    strength_embedding_mae=strength_embedding_mae,
    strength_extraction_mae=strength_extraction_mae,
)

# Instantiate a metric function.
print("\n[INFO]: Instantiate a metric function ...")
psnr_train = PSNR()
psnr_test = PSNR()

ber_train = BER()
ber_test = BER()


# ############ FUNCTION #############


@tf.function
def train_step(images: tf.Tensor):
    # Random binary images
    batch_size = tf.shape(images)[0]
    marks = random_binary_images(batch_size=batch_size, shape=DIMS_MARK[0])

    with tf.GradientTape() as embedding_tape, tf.GradientTape() as extraction_tape:
        # embedding and marks
        embeddings = embedding_model([images, marks], training=True)
        extracted_marks = extraction_model(
            tf.numpy_function(
                random_augmentation,
                [preprocess_output(embeddings, mode="tf")],
                tf.float32,
            ),
            training=True,
        )

        # calculates loss
        loss_embedding, loss_extraction = loss_fn(
            images, marks, embeddings, extracted_marks
        )

    # computes gradient
    grads_embedding = embedding_tape.gradient(
        loss_embedding, embedding_model.trainable_weights
    )
    grads_extraction = extraction_tape.gradient(
        loss_extraction, extraction_model.trainable_weights
    )

    # train extraction model
    embedding_optimizer.apply_gradients(
        zip(grads_embedding, embedding_model.trainable_weights)
    )
    extraction_optimizer.apply_gradients(
        zip(grads_extraction, extraction_model.trainable_weights)
    )

    # update metrics
    psnr_train.update_state(
        preprocess_output(images, mode="tf"), preprocess_output(embeddings, mode="tf")
    )
    ber_train.update_state(marks, extracted_marks)
    return loss_embedding, loss_extraction


@tf.function
def test_step(images: tf.Tensor):
    # Random binary images
    batch_size = tf.shape(images)[0]
    marks = random_binary_images(batch_size=batch_size, shape=DIMS_MARK[0])

    # embedding and marks
    embeddings = embedding_model([images, marks], training=False)
    extracted_marks = extraction_model(
        preprocess_output(embeddings, mode="tf"), training=False
    )

    # update metrics
    psnr_test.update_state(
        preprocess_output(images, mode="tf"), preprocess_output(embeddings, mode="tf")
    )
    ber_test.update_state(marks, extracted_marks)


# ############ TRAIN #############


print("\n[INFO] training model ...")
for epoch in range(EPOCHS):
    # Reset training metrics at the end of each epoch
    psnr_train.reset_states()
    ber_train.reset_states()

    # Reset testing metrics at the end of each epoch
    psnr_test.reset_states()
    ber_test.reset_states()

    with Progress() as progress:
        train_bar = progress.add_task(
            f"Epoch {epoch+1} / {EPOCHS}", total=train_size // BS
        )
        for step, real_images in enumerate(train_set):
            # train embedding and extraction model
            loss_embedding, loss_extraction = train_step(real_images)

            # Logging
            if (step + 1) % 10 == 0:
                progress.console.print(
                    f"Step {step+1} -> Embedding loss: {loss_embedding:.2f} - Extraction loss: {loss_extraction:.2f}"
                )
            progress.advance(train_bar)

    # Display metrics at the end of each epoch.
    print(
        f"Training -> PSNR: {float(psnr_train.result()):.4f} - BER: {float(ber_train.result()):.4f}"
    )

    # Run a validation loop at the end of each epoch.
    for real_images in test_set:
        test_step(real_images)

    # Display metrics.
    print(
        f"Testing -> PSNR: {float(psnr_test.result()):.4f} - BER: {float(ber_test.result()):.4f}"
    )


print("\n[INFO] Save model ...")
embedding_model.compile(
    optimizer=embedding_optimizer, loss=loss_fn, metrics=[PSNR(), BER()]
)

embedding_model.save(join(MODEL_PATH, "embedding_model_Attacked.h5"))

extraction_model.compile(
    optimizer=extraction_optimizer, loss=loss_fn, metrics=[PSNR(), BER()]
)
extraction_model.save(join(MODEL_PATH, "extraction_model_Attacked.h5"))
