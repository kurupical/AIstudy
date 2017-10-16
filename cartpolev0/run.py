# common lib
import gym
import json

# original lib
from dqn import *
from agent import *
from organizer import *

# Configファイルの取得
f = open("config/config.json", "r")
config = json.load(f)

# ゲームの作成
env = gym.make('CartPole-v0')
state = env.reset()
n_in = state.size
n_out = 2 # 取れるactionの種類

# 各種パラメータの取得
## network param
n_l1hidden = config['n_l1hidden']
n_l2hidden = config['n_l2hidden']
learning_rate_nw = config['learning_rate_nw']
## agent param
gamma = config['gamma']
epsilon = config['epsilon']
annealing_rate = config['annealing_rate']
learning_rate_ag = config['learning_rate_ag']

# 環境構築
network = DQN(n_l1hidden=n_l1hidden,
              n_l2hidden=n_l2hidden,
              n_in=n_in,
              n_out=n_out,
              learning_rate=learning_rate_nw)

agent = Agent(network=network,
              gamma=gamma,
              epsilon=epsilon,
              annealing_rate=annealing_rate,
              learning_rate=learning_rate_ag)

organizer = Organizer(agent=agent,
                      env=env)

for i in range(1000):
    val_loss = organizer.step(isTrain=True, batch_size=100)
    # print("i={}, val_loss={}".format(i, val_loss))
    organizer.step(isTrain=False, batch_size=20)
print("end")
