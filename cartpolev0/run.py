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
n_out = 1 # 取れるactionの種類

n_l1hidden = config['n_l1hidden']
n_l2hidden = config['n_l2hidden']

network = DQN(n_l1hidden=n_l1hidden,
              n_l2hidden=n_l2hidden,
              n_in=n_in,
              n_out=n_out)

agent = Agent(network=network)

organizer = Organizer(agent=agent,
                      env=env)

for i in range(1000):
    organizer.step(isTrain=True, batch_size=1000)
    organizer.step(isTrain=False, batch_size=10)
print("end")
