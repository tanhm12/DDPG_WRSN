import numpy as np
import tensorflow as tf

df = 0.99
lr = 22e-4
Tau = 0.1
#
tf.compat.v1.disable_eager_execution()
init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=22)


class Actor:
    """Actor directly maps states to actions."""

    def __init__(self, input_shape=None,
                 output_len=None,
                 output_scale=None,
                 learning_rate=22e-4,
                 tau=0.1):
        self.input_shape = input_shape
        self.output_len = output_len
        self.output_scale = output_scale
        self.learning_rate = learning_rate
        self.tau = tau  # for update the target net
        self.out = None
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.optimizers = tf.keras.optimizers.Adam()

    def _build_model(self):
        inp = tf.keras.layers.Input(shape=self.input_shape)
        model = tf.keras.layers.Dense(int(0.8*self.input_shape[0]), activation="relu", kernel_initializer=init)(inp)
        model = tf.keras.layers.Dense(int(0.5*self.input_shape[0]), activation="linear", kernel_initializer=init)(model)
        model = tf.keras.layers.Dense(int(0.2*self.input_shape[0]), activation="linear", kernel_initializer=init)(model)
        out = tf.keras.layers.Dense(self.output_len, activation="sigmoid")(model)
        # out = tf.math.multiply(out, self.output_scale)
        self.out = out
        return tf.keras.models.Model(inputs=inp, outputs=out)

    def predict(self, state):
        return self.model.predict(np.array([state]))
        # return np.array([pred[i] * self.output_scale[i] for i in range(self.output_len)])

    def target_predict(self, state):
        return self.target_model.predict(np.array([state]))
        # return np.array([pred[i] * self.output_scale[i] for i in range(self.output_len)])

    def get_grads(self, critic_grads, samples):
        # critic_grads = np.array([[critic_grads[0][i]/self.output_scale[i] for i in range(self.output_len)]])
        critic_grads = np.array(critic_grads[0]).reshape((-1, self.output_len))
        states = np.array([sample[0] for sample in samples])
        # print(critic_grads)
        grad = tf.gradients(self.model.outputs, self.model.trainable_variables, -critic_grads)
        get_grad = tf.keras.backend.function(self.model.inputs, [grad])
        grads = get_grad(states)
        # print(critic_grads)
        # print(grads[0])
        # with tf.GradientTape() as tape:
        #     pred = self.model(states)
        # print(pred, "\n", critic_grads)
        # grads = tape.gradient(pred, self.model.trainable_variables, -critic_grads)
        # grads = np.array(grads)
        # print(len(grads[0][0]))
        return grads

    def update_target_net(self):
        # network_params = self.model.trainable_variables
        # target_network_params = self.target_model.trainable_variables
        # update_target_network_params = \
        #     [target_network_params[i].assign(tf.multiply(network_params[i], self.tau) +
        #                                      tf.multiply(target_network_params[i], 1. - self.tau))
        #      for i in range(len(target_network_params))]
        weights, target_weights = self.model.get_weights(), self.target_model.get_weights()

        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]

        self.target_model.set_weights(target_weights)

    def network_copy(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, path):
        self.model.save_weights(path+'/actor.h5')


class Critic:
    """Critic calculates Q-values"""

    def __init__(self, state_shape=None,
                 action_len=None,
                 output_len=None,
                 learning_rate=22e-4,
                 discount_factor=0.9,
                 tau=0.1):
        self.state_shape = state_shape
        self.action_len = action_len
        self.output_len = output_len
        # self.action_tens = action_tens
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.tau = tau  # for update the target net
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        # self.out = None

    def _build_model(self):
        state = tf.keras.layers.Input(shape=self.state_shape)
        state1 = tf.keras.layers.Dense(int(0.7*self.state_shape[0]), activation="relu", kernel_initializer=init)(state)
        action = tf.keras.layers.Input(shape=(self.action_len,))
        # action1 = tf.keras.layers.Dense(24, activation="relu")(action)

        out = tf.keras.layers.concatenate([state1, action])
        out = tf.keras.layers.Dense(int(0.5*self.state_shape[0]), activation="linear", kernel_initializer=init)(out)
        out = tf.keras.layers.Dense(int(0.3*self.state_shape[0]), activation="linear", kernel_initializer=init)(out)
        out = tf.keras.layers.Dense(1, activation="linear")(out)
        model = tf.keras.models.Model(inputs=[state, action], outputs=out)
        # self.out = out
        model.compile(optimizer="adam", loss="MSE")

        return model

    def predict(self, state, action):
        return self.model.predict([[state], [action]])[0]

    def target_predict(self, state, action):
        # print(self.target_model.predict([[state], [action]])[0])
        return self.target_model.predict([[state], [action]])[0]

    def get_grads(self, samples):
        grad = tf.keras.backend.gradients(self.model.outputs, [self.model.inputs[1]])
        get_grad = tf.keras.backend.function(self.model.inputs, [grad])

        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        grads = get_grad([states, actions])
        return grads[0]

    def update_target_net(self):
        # network_params = self.model.trainable_variables
        # target_network_params = self.target_model.trainable_variables
        # update_target_network_params = \
        #     [target_network_params[i].assign(tf.multiply(network_params[i], self.tau) +
        #                                      tf.multiply(target_network_params[i], 1. - self.tau))
        #      for i in range(len(target_network_params))]
        weights, target_weights = self.model.get_weights(), self.target_model.get_weights()

        for i in range(len(weights)):
            target_weights[i] = self.tau*weights[i] + (1-self.tau)*target_weights[i]

        self.target_model.set_weights(target_weights)

    def network_copy(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, path):
        self.model.save_weights(path+'/critic.h5')


class Agent:
    def __init__(self, state_shape=None, action_len=None, action_scale=None):
        self.state_shape = state_shape
        self.action_len = action_len
        self.action_scale = action_scale
        self.actor = Actor(state_shape, action_len, action_scale)
        self.critic = Critic(state_shape, action_len, output_len=1)

    def act(self, state):
        return self.actor.predict(state)[0]

    def summary(self):
        print("Actor network:")
        self.actor.model.summary()
        print("\nCritic network:")
        self.critic.model.summary()

    def train_critic(self, samples):
        inp = [[sample[0] for sample in samples], [sample[1] for sample in samples]]
        expected_output = []
        for state, action, reward, next_state, done in samples:
            if done:
                y = reward
            else:
                next_action = self.actor.target_predict(state)[0]
                y = reward + self.critic.gamma * self.critic.target_predict(next_state, next_action)[0]
            # print(y)
            expected_output.append(np.array([y]))

        # print(inp)
        # print(expected_output)
        # print(self.critic.model.predict(inp))
        self.critic.model.fit(inp, [np.array(expected_output)], verbose=1)

    def train_actor(self, critic_grads, samples):
        # states = np.array([sample[0] for sample in samples])
        # actions = np.array([sample[1] for sample in samples])
        # with tf.GradientTape() as tape:
        #     pred = tf.reduce_mean(self.critic.model([states, actions]), axis=0, keepdims=True)
        #     loss = -tf.reduce_mean(pred)
        # print(grads)
        # grads = tape.gradient(loss, self.actor.model.trainable_variables)
        grads = self.actor.get_grads(critic_grads, samples)[0]
        tf.keras.optimizers.Adam(learning_rate=lr).apply_gradients(zip(grads, self.actor.model.trainable_weights))

    def train(self, samples):
        self.train_critic(samples)
        grads = self.critic.get_grads(samples)
        self.train_actor(grads, samples)

    def update_target_net(self):
        self.actor.update_target_net()
        self.critic.update_target_net()

    def network_copy(self):
        self.actor.network_copy()
        self.critic.network_copy()

    def save(self, path):
        self.actor.save(path)
        self.critic.save(path)
