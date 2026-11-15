"""Research segmentation models. TensorFlow is imported only when needed."""
from __future__ import annotations


def build_model(name: str, image_size: int = 256, imagenet_weights: bool = True):
    import tensorflow as tf
    from tensorflow.keras import layers as L

    if image_size < 64 or image_size % 32:
        raise ValueError("Image size must be >=64 and divisible by 32")
    shape = (image_size, image_size, 3)

    def conv_block(x, channels):
        for _ in range(2):
            x = L.Conv2D(channels, 3, activation="relu", padding="same")(x)
        return x

    if name == "unet":
        inputs = tf.keras.Input(shape=shape)
        c1 = conv_block(inputs, 32)
        c2 = conv_block(L.MaxPooling2D()(c1), 64)
        c3 = conv_block(L.MaxPooling2D()(c2), 128)
        bridge = conv_block(L.MaxPooling2D()(c3), 256)
        x = L.Concatenate()([L.UpSampling2D()(bridge), c3])
        x = conv_block(x, 128)
        x = L.Concatenate()([L.UpSampling2D()(x), c2])
        x = conv_block(x, 64)
        x = L.Concatenate()([L.UpSampling2D()(x), c1])
        x = conv_block(x, 32)
        return tf.keras.Model(inputs, L.Conv2D(1, 1, activation="sigmoid")(x), name="compact_unet")

    weights = "imagenet" if imagenet_weights else None
    if name == "deeplabv3plus":
        backbone = tf.keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights=weights)
        backbone.trainable = False
        high = backbone.get_layer("block_13_expand_relu").output
        low = backbone.get_layer("block_2_expand_relu").output
        high_h, high_w = int(high.shape[1]), int(high.shape[2])
        low_h, low_w = int(low.shape[1]), int(low.shape[2])

        def bn_relu(x):
            return L.Activation("relu")(L.BatchNormalization()(x))

        paths = [bn_relu(L.Conv2D(128, 1, padding="same", use_bias=False)(high))]
        for rate in (6, 12, 18):
            paths.append(bn_relu(L.Conv2D(128, 3, dilation_rate=rate, padding="same", use_bias=False)(high)))
        pooled = L.GlobalAveragePooling2D(keepdims=True)(high)
        pooled = bn_relu(L.Conv2D(128, 1, padding="same", use_bias=False)(pooled))
        paths.append(L.Resizing(high_h, high_w, interpolation="bilinear")(pooled))
        x = bn_relu(L.Conv2D(128, 1, padding="same", use_bias=False)(L.Concatenate()(paths)))
        x = L.Resizing(low_h, low_w, interpolation="bilinear")(x)
        low = bn_relu(L.Conv2D(48, 1, padding="same", use_bias=False)(low))
        x = L.Concatenate()([x, low])
        for _ in range(2):
            x = bn_relu(L.Conv2D(128, 3, padding="same", use_bias=False)(x))
        x = L.Resizing(image_size, image_size, interpolation="bilinear")(x)
        return tf.keras.Model(backbone.input, L.Conv2D(1, 1, activation="sigmoid")(x), name="deeplabv3plus_mobilenetv2")

    if name == "resnet50_unet":
        backbone = tf.keras.applications.ResNet50(input_shape=shape, include_top=False, weights=weights)
        backbone.trainable = False
        skips = [backbone.get_layer(layer).output for layer in (
            "conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out")]
        x = skips[-1]
        for skip, filters in zip(reversed(skips[:-1]), (256, 128, 64, 32)):
            x = L.Conv2DTranspose(filters, 3, strides=2, padding="same")(x)
            x = L.Concatenate()([x, skip])
            x = conv_block(x, filters)
        x = conv_block(L.Conv2DTranspose(16, 3, strides=2, padding="same")(x), 16)
        return tf.keras.Model(backbone.input, L.Conv2D(1, 1, activation="sigmoid")(x), name="resnet50_unet")
    raise ValueError(f"Unknown model {name!r}; use unet, deeplabv3plus, or resnet50_unet")


def preprocess(images, name: str):
    """Use the correct input convention for each new training model."""
    import numpy as np
    images = np.asarray(images, dtype=np.float32)
    if name == "unet":
        return images / 255.0
    if name == "deeplabv3plus":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input(images.copy())
    if name == "resnet50_unet":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(images.copy())
    raise ValueError(f"Unknown model: {name}")
