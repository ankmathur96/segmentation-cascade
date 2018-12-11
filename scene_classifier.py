import tensorflow as tf
from video_utils import get_all_images_for_scene


class SceneClassifier():

	def __init__(self):
		self.lr = 1e-4
		self.scene_to_label = {1:0, 3:1, 4:2, 5:3}
		self.scene_indices = [1,3,4,5]
		self.num_classes = 4
		self.print_every = 5
		self.num_train_steps = 1000
		self.batch_size = 8
		self.model_dir = "./scene_classification_model"

		self.image_paths, self.labels = self.load_data()
		self.dataset_size = len(self.image_paths)

		self.logits, self.loss = self.forward_prop()
		print("set up model architecture ")
		self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		self.sess = tf.InteractiveSession()
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())


	def load_data(self):
		print("Loading data ... ")
		image_paths, labels = [], []
		for scene_idx in self.scene_indices:
			images_from_scene = get_all_images_for_scene(scene_idx)
			image_paths.extend(images_from_scene)
			labels.extend([self.scene_to_label[scene_idx] for _ in range(len(images_from_scene))])
		print("finished.")
		return image_paths, labels


	def forward_prop(self):
	    inputs = tf.placeholder(tf.float32, (None, 299, 299, 3))
	    labels = tf.placeholder(tf.int32, (None, self.num_classes))

	    conv_outs = tf.layers.conv2d(inputs, 16, kernel_size=5)
	    flattened = tf.contrib.layers.flatten(conv_outs)
	    fc_outs = tf.layers.dense(flattened, 1024)
	    fc_final = tf.layers.dense(fc_outs, 4)

	    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=fc_final)
	    return fc_final, loss 


	def sample_batch(self):
		random_indices = np.random.random_integers(self.dataset_size-1, size(self.batch_size,))
		sampled_image_paths = [self.images[idx] for idx in random_indices]
		sampled_labels = [self.labels[idx] for idx in random_labels]

		label_batch = tf.one_hot(sampled_labels, depth=self.num_classes)  # batch_size x num_classes
		image_batch = tf.zeros(shape=(self.batch_size, 299, 299, 3))

		for idx, image_path in enumerate(sampled_image_paths):
			image_string = tf.read_file(image_path)
			image_decoded = tf.image.decode_jpeg(image_string, channels=3)
			image_resized = tf.image.resize_images(image_decoded, (299,299))
			image = tf.cast(image_resized, tf.float32)
			image_batch[idx, :, :, :] = image 

		return image_batch, label_batch

	def train_model(self):
		print("Beginning model training ")
		for i in range(self.num_train_steps):
			if (i % self.print_every == 0):
				print("Beginning batch ", i)

			image_batch, label_batch = self.sample_batch()
			self.train_step.run(feed_dict={inputs: image_batch, labels: label_batch})

			if (i % self.print_every == 0):
				print("Loss: ", self.loss.eval(feed_dict={inputs: image_batch, labels: label_batch}))


if __name__=="__main__":
	model = SceneClassifier()
	model.train_model()
