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
from src.config import MODEL_PATH, TEST_PATH, TRAIN_PATH
from src.data import prepare_data_from_slice
from src.model import ExtractWaterMark, WaterMark
from src.processors import preprocess_output, random_augmentation
from src.visualizations import print_tables


def train(params: dict) -> None:
    # ############ WARM UP #############

    # load dataset
    print("\n[INFO]: Load dataset ...")
    train_set, train_size = prepare_data_from_slice(
        inputs_path=TRAIN_PATH,
        extension=params["image_format"],
        image_dims=params["dimensions"]["images"],
        batch_size=params["hyperparemeters"]["batch_size"],
    )
    test_set, test_size = prepare_data_from_slice(
        inputs_path=TEST_PATH,
        extension=params["image_format"],
        image_dims=params["dimensions"]["images"],
        batch_size=params["hyperparemeters"]["batch_size"],
        training=False,
    )

    # construct our model
    print("\n[INFO]: Create model ...")
    embedding_model = WaterMark.build(
        image_dims=params["dimensions"]["images"],
        mark_dims=params["dimensions"]["mark"],
        strength=params["strength"],
    )
    extraction_model = ExtractWaterMark.build(mark_dims=params["dimensions"]["mark"])

    # Instantiate optimizers.
    print("\n[INFO]: Instantiate optimizers ...")
    embedding_optimizer = tf.keras.optimizers.Adam(
        learning_rate=params["hyperparemeters"]["learning_rate"]["embedding"],
        decay=0.01 / params["hyperparemeters"]["epochs"],
    )
    extraction_optimizer = tf.keras.optimizers.Adam(
        learning_rate=params["hyperparemeters"]["learning_rate"]["extraction"],
        decay=0.01 / params["hyperparemeters"]["epochs"],
    )

    # Instantiate a loss function.
    print("\n[INFO]: Instantiate a loss function ...")
    loss_fn = WatermarkLoss(
        strength_embedding_mse=params["loss_parameters"]["embedding_mse"],
        strength_embedding_mae=params["loss_parameters"]["embedding_mae"],
        strength_extraction_mae=params["loss_parameters"]["extraction_mae"],
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
        marks = random_binary_images(
            batch_size=batch_size, shape=params["dimensions"]["mark"][0]
        )

        with tf.GradientTape() as embedding_tape, tf.GradientTape() as extraction_tape:
            # embedding and marks
            embeddings = embedding_model([images, marks], training=True)
            extracted_marks = extraction_model(
                tf.numpy_function(
                    random_augmentation,
                    [
                        preprocess_output(embeddings, mode="tf"),
                        params["attack"]["type"],
                        params["attack"].get("percentage", 0),
                    ],
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
            preprocess_output(images, mode="tf"),
            preprocess_output(embeddings, mode="tf"),
        )
        ber_train.update_state(marks, extracted_marks)
        return loss_embedding, loss_extraction

    @tf.function
    def test_step(images: tf.Tensor):
        # Random binary images
        batch_size = tf.shape(images)[0]
        marks = random_binary_images(
            batch_size=batch_size, shape=params["dimensions"]["mark"][0]
        )

        # embedding and marks
        embeddings = embedding_model([images, marks], training=False)
        extracted_marks = extraction_model(
            preprocess_output(embeddings, mode="tf"), training=False
        )

        # update metrics
        psnr_test.update_state(
            preprocess_output(images, mode="tf"),
            preprocess_output(embeddings, mode="tf"),
        )
        ber_test.update_state(marks, extracted_marks)

    # ############ TRAIN #############

    print("\n[INFO] training model ...")
    for epoch in range(params["hyperparemeters"]["epochs"]):
        # Reset training metrics at the end of each epoch
        psnr_train.reset_states()
        ber_train.reset_states()

        # Reset testing metrics at the end of each epoch
        psnr_test.reset_states()
        ber_test.reset_states()

        with Progress() as progress:
            train_bar = progress.add_task(
                f"Epoch {epoch + 1} / {params['hyperparemeters']['epochs']}",
                total=train_size // params["hyperparemeters"]["batch_size"],
            )
            for step, real_images in enumerate(train_set):
                # train embedding and extraction model
                loss_embedding, loss_extraction = train_step(real_images)

                # Logging
                if (step + 1) % 10 == 0:
                    progress.console.print(
                        f"Step {step + 1} ->:"
                        f" Embedding loss: {loss_embedding:.2f} - Extraction loss: {loss_extraction:.2f}"
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

    # ############ STORE #############

    if params.get("storage_name", False):
        print("\n[INFO] Save model ...")
        embedding_model.compile(
            optimizer=embedding_optimizer, loss=loss_fn, metrics=[PSNR(), BER()]
        )

        embedding_model.save(
            join(MODEL_PATH, f"embedding_model_{params['storage_name']}.h5")
        )

        extraction_model.compile(
            optimizer=extraction_optimizer, loss=loss_fn, metrics=[PSNR(), BER()]
        )
        extraction_model.save(
            join(MODEL_PATH, f"extraction_model_{params['storage_name']}.h5")
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from json import load as json_load

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--config", help="JSON configuration file path", type=str, required=True
    )
    parser.add_argument("--name", help="Model name to store", type=str)
    args = parser.parse_args()

    # load configuration
    params = json_load(open(file=args.config, mode="r", encoding="utf-8"))
    if args.name:
        params["storage_name"] = args.name

    # print hyperparameters
    print("\n[INFO] Hyper-parameters for this training ...")
    headers = [
        "Batch",
        "Epochs",
        "LR_embedding",
        "LR_extraction",
        "s",
        "λ_1",
        "λ_2",
        "λ_3",
    ]
    contents = [
        [
            params["hyperparemeters"]["batch_size"],
            params["hyperparemeters"]["epochs"],
            params["hyperparemeters"]["learning_rate"]["embedding"],
            params["hyperparemeters"]["learning_rate"]["extraction"],
            params["strength"],
            params["loss_parameters"]["embedding_mse"],
            params["loss_parameters"]["embedding_mae"],
            params["loss_parameters"]["extraction_mae"],
        ]
    ]
    print_tables(title="Hyper-parameters", headers=headers, contents=contents)

    # train model and store it
    train(params)
