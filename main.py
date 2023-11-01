import os
import tensorflow as tf
import Numex
import DQNAgent
import matplotlib.pyplot as plt
#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

EPISODES = 10


def train_model():
    all_rewards = []
    actions = []
    strategy = []
    actions_memory =[]
    state_size = Numex.state_size
    action_size = Numex.action_size
    agent = DQNAgent.DQNAgent(state_size, action_size)
    file_path = "C:/Users/Raul_MSI/Python_Notebooks/GPU_tests/saved_model/Numex_model_test"+str(state_size)+".h5"
    if os.path.exists(file_path) == True:
        agent.load(file_path)
    done = False
    batch_size = 59 #размер выборки

    for e in range(EPISODES):
        state = Numex.reset()
        for time in range(500):
            # env.render()
            actions = []
            action = agent.act(state)
            actions.append(action)
            next_state, reward, done, actions_memory, info = Numex.run_Numex(state,action, actions_memory)
            # observation, reward, terminated, truncated, info
            agent.memorize(state, action, reward, next_state, done) # запоминаем состояния, действия и награду
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size: #набрали ли мы достаточное кол-во опыта
                agent.replay(batch_size)
        strategy.append(actions_memory)
        all_rewards.append(reward)
        print('Все награды: ', all_rewards)
        agent.save(name=file_path)
    plt.plot(all_rewards)
    plt.title("Зависимость NPV от числа эпизодов")
    plt.xlabel('Эпизоды')
    plt.ylabel('NPV [млн. руб.]')
    plt.show()
    print("Total Reward: ", all_rewards)
    print("Epsilon : ", agent.epsilon)
    print("Все стратегии: ", strategy)

def predict_model():
    state_size = Numex.state_size
    action_size = Numex.action_size
    agent = DQNAgent.DQNAgent(state_size, action_size)
    file_path = "C:/Users/Raul_MSI/Python_Notebooks/GPU_tests/saved_model/Numex_model_test"+str(state_size)+".h5"
    if os.path.exists(file_path) == True:
        agent.load(file_path)
    else:
        print("File is not exist!")
        return
    state = Numex.reset()
    result = agent.predict(state)
    print(result)
    return

if __name__ == "__main__":
    train_model()