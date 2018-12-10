import tensorflow as tf

labels # 0-4 based on scene in a flattened array
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
input = tf.Placeholder((-1, 299, 299, 3))
conv_outs = tf.layers.conv2d(inputs, 16, kernel_size=5)
fc_outs = tf.layers.dense(conv_outs, 1024)
softmax = tf.nn.softmax(tf.layers.dense(fc_outs, 5))
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)