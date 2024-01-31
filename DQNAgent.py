import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 #было 0.995
        self.learning_rate = 0.01 # было 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) #ценность действий| влево - ценность 0.5, вправо - ценность 0.3. Поэтому выбираем лево
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size) #сейчас 33 действия batch, мы выбираем произвольные 32
        for state, action, reward, next_state, done in minibatch:
            target = reward #целевая награда
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0])) # дисконт следующей награды
            target_f = self.model.predict(state) # спрашиваем у модели какая будет награда при текущем состоянии
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def predict(self, state):
        act_values = self.model.predict(state) #ценность действий| влево - ценность 0.5, вправо - ценность 0.3. Поэтому выбираем лево
        return act_values