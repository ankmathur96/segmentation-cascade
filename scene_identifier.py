import tensorflow as tf
from video_utils import get_all_images_for_scene
scenes = [1,3,4,5]
LEARNING_RATE = 1e-4
MODEL_DIR = 'scene-recog-model/'
BATCH_SIZE = 4
TRAIN_EPOCHS = 100
scene_to_label = {1:0, 3:1, 4:2, 5:3}
# Input Specification
def parse_fn(f, l):
    image_string = tf.read_file(f)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, (299,299))
    image = tf.cast(image_resized, tf.float32)
    label = tf.one_hot(l, depth=len(scenes))
    print(label)
    return image, label
images, labels = [], []
for scene_idx in scenes:
    images_from_scene = get_all_images_for_scene(scene_idx)
    # images_from_scene = map(parse_fn, images_from_scene)
    images.extend(images_from_scene)
    labels.extend([scene_to_label[scene_idx] for _ in range(len(images_from_scene))])
    # sess.run([images])
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.map(parse_fn)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
# Model Specification

def create_model(inputs):
    # inputs = tf.placeholder(tf.float32, (None, 299, 299, 3))
    conv_outs = tf.layers.conv2d(inputs, 16, kernel_size=5)
    flattened = tf.contrib.layers.flatten(conv_outs)
    fc_outs = tf.layers.dense(flattened, 1024)
    fc_final = tf.layers.dense(fc_outs, 4)
    return fc_final

def model_fn(features, labels, mode, params):
    print(features)
    images = features
    model = create_model(images)
    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model
        predictions = {
            'classes': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,predictions=predictions)
    elif mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        logits = model
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        print(tf.argmax(logits, axis=1))
        print(labels)
        # accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))
        tf.identity(LEARNING_RATE, 'learning_rate')
        tf.identity(loss, 'cross_entropy')
        tf.summary.scalar('xentropy_loss', loss)
        # tf.identity(accuracy[1], name='train_accuracy')
        # tf.summary.scalar('train_accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
    elif mode == tf.estimator.ModeKeys.EVAL:
        logits = model
        loss = tf.losses.softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops={'accuracy' : tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))})

def train_detector():
    scene_classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=MODEL_DIR)

    for i in range(TRAIN_EPOCHS):
        print(i)
        scene_classifier.train(input_fn=lambda: train_input_fn(images, labels, BATCH_SIZE))
        # eval_results = scene_classifier.evaluate(input_fn=eval_input_fn)
        print('\nEvaluation results:\n\t%s\n' % eval_results)
train_detector()
    
# load images