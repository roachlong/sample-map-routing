import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, sequence_length=5):
        self.steps = 0
        self.state_size = state_size  # e.g. 4: [ax, ay, gx, gy]
        self.sequence_length = sequence_length  # lookback window
        self.action_size = action_size  # number of available actions
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.00
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
            tf.keras.layers.Input(shape=(self.sequence_length, self.state_size)),
            tf.keras.layers.GRU(256, return_sequences=True, dropout=0.2),
            tf.keras.layers.GRU(128, dropout=0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state_seq, action, reward, next_state_seq, done):
        self.memory.append((state_seq, action, reward, next_state_seq, done))

    def act(self, state_seq):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Ensure state_seq has the correct shape
        if len(state_seq.shape) == 2:  # (sequence_length, state_size)
            state_seq = state_seq[np.newaxis, :, :]  # Add batch dimension
        q_values = self.model.predict(state_seq, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for state_seq, action, reward, next_state_seq, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state_seq[np.newaxis, :, :], verbose=0)[0])
            target_f = self.model.predict(state_seq[np.newaxis, :, :], verbose=0)
            target_f[0][action] = target
            self.model.fit(state_seq[np.newaxis, :, :], target_f, epochs=1, verbose=0)
        self.steps += 1
        self.epsilon = max(self.epsilon_min, self._epsilon_schedule())

    def load(self, name):
        self.model.load_weights(name)

    def _epsilon_schedule(self):
        # Custom schedule: slower decay at first, faster later
        if self.steps < 2000:
            return self.epsilon * 0.9995  # Slower decay
        elif self.steps < 10000:
            return self.epsilon * 0.995
        else:
            return self.epsilon * 0.99

    def save(self, name):
        self.model.save_weights(name)
