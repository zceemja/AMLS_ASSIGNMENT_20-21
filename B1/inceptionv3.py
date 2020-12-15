import tensorflow as tf
import tensorflow_hub as hub

def prepare_model():
    m = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
                       trainable=False),  # Can be True, see below.
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    m.build([None, 299, 299, 3])  # Batch input shape.

# def inceptionv3_model_fn(features, labels, mode):
#     # Load Inception-v3 model.
#     module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
#     input_layer = adjust_image(features["x"])
#     outputs = module(input_layer)
#
#     logits = tf.keras.layers.Dense(inputs=outputs, units=10)
#
#     predictions = {
#         # Generate predictions (for PREDICT and EVAL mode)
#         "classes": tf.argmax(input=logits, axis=1),
#         # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
#         # `logging_hook`.
#         "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
#     }
#
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#     # Calculate Loss (for both TRAIN and EVAL modes)
#     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
#
#     # Configure the Training Op (for TRAIN mode)
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#         train_op = optimizer.minimize(
#             loss=loss,
#             global_step=tf.train.get_global_step())
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
#
#     # Add evaluation metrics (for EVAL mode)
#     eval_metric_ops = {
#         "accuracy": tf.metrics.accuracy(
#             labels=labels, predictions=predictions["classes"])}
#     return tf.estimator.EstimatorSpec(
#         mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
#
#
# def adjust_image(data):
#     # Reshape to [batch, height, width, channels].
#     imgs = tf.reshape(data, [-1, 28, 28, 1])
#     # Adjust image size to Inception-v3 input.
#     imgs = tf.image.resize_images(imgs, (299, 299))
#     # Convert to RGB image.
#     imgs = tf.image.grayscale_to_rgb(imgs)
#     return imgs