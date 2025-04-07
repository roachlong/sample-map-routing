import tensorflow as tf
import numpy as np
import random
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.steps = 0
        self.state_size = state_size  # e.g. 4: [ax, ay, gx, gy]
        self.action_size = action_size  # number of available actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        ####### 
        # Neural Net for Deep-Q learning Model
        # Input: state (4 values)
        # Output: Q-values for each action (action_size)
        #
        # Compile with Adam optimizer and mean squared error loss
        # Note: The architecture and hyperparameters can be tuned for better performance
        # The current architecture is a simple feedforward neural network
        # with 3 hidden layers and ReLU activation functions.
        # The output layer uses linear activation to predict Q-values.
        #
        # You can make changes to the model architecture,
        # learning rate, and other hyperparameters
        # to improve the performance of the agent.
        # See the init method for parmeters that can be changed.
        #######
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)
            target_f[0][action] = target
            self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)
        self.steps += 1
        self.epsilon = max(self.epsilon_min, self._epsilon_schedule())

    def load(self, name):
        self.model.load_weights(name)

    def _epsilon_schedule(self):
        # Custom schedule: slower decay at first, faster later
        if self.steps < 500:
            return self.epsilon * 0.999
        elif self.steps < 2000:
            return self.epsilon * 0.995
        else:
            return self.epsilon * 0.99

    def save(self, name):
        self.model.save_weights(name)
