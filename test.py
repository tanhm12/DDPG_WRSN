import numpy as np
import tensorflow as tf

# model = tf.keras.Sequential()
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

state = tf.keras.layers.Input(shape=(3,))
state1 = tf.keras.layers.Dense(32, activation="relu")(state)
action = tf.keras.layers.Input(shape=(2,))
action1 = tf.keras.layers.Dense(32, activation="relu")(action)

out = tf.keras.layers.Add()([state1, action1])
out = tf.keras.layers.Dense(32, activation="linear")(out)
out = tf.keras.layers.Dense(1, activation="tanh")(out)
out = tf.multiply(out, np.array([2]))
model = tf.keras.models.Model(inputs=[state, action], outputs=out)
model.summary()

s = np.array([[1., 2., 3.], [3., 4., 5.]])
a = np.array([[4., 5.], [2., 3.]])
# print(s.shape)
print(tens)
# print(s.shape, tf.reduce_mean(tens, axis=0, keepdims=True))

with tf.GradientTape() as tape:
    pred = model([s, a])
    print(pred)
    print(tf.reduce_mean(pred, axis=1, keepdims=True))
print(tape.gradient(pred, model.trainable_variables))
# grad = tf.keras.backend.gradients(model.outputs, model.inputs[1])
# grads = tf.keras.backend.function(model.inputs, [grad])
# # print(model.inputs, model.outputs)
# print(grads([s, a])[0])
# print(model.predict([s, a]))
