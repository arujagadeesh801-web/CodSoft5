import tensorflow as tf


def build_model(vocab_size, sequence_length=80):
    """
    Build a character-level LSTM text-generation model.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(sequence_length,),
            dtype=tf.int32
        ),

        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=64
        ),

        tf.keras.layers.LSTM(
            128,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(vocab_size)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=["sparse_categorical_accuracy"]
    )

    return model