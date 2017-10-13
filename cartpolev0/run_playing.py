# common lib
import gym
import json
import datetime

# original lib
from dqn import *
from agent import *
from organizer import *


# Configファイルの取得
config_path = "model/config.json"
f = open(config_path, "r")
config = json.load(f)
# モデル読み込み先ディレクトリ
load_path = "model/model.ckpt"
# モデル保存先ディレクトリ
result_path = "result/" + datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S") + "_" + str(config['title']) + "/"

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
network.load(load_path)

agent = Agent(network=network,
              gamma=gamma,
              epsilon=epsilon,
              annealing_rate=annealing_rate,
              learning_rate=learning_rate_ag)

organizer = Organizer(agent=agent,
                      env=env)

for i in range(100):
    val_loss = organizer.step(isTrain=True, batch_size=100)
    print("i={}, val_loss={}".format(i, val_loss))
    organizer.step(isTrain=False, batch_size=10,isVisualize=False)

agent.save(result_path, config_path)
print("end")
