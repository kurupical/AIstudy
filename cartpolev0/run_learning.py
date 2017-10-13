# common lib
import gym
import json
import datetime
from matplotlib import pyplot as plt

# original lib
from dqn import *
from agent import *
from organizer import *


# Configファイルの取得
config_path = "config/config.json"
f = open(config_path, "r")
config = json.load(f)
# モデル保存先ディレクトリの設定
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

agent = Agent(network=network,
              gamma=gamma,
              epsilon=epsilon,
              annealing_rate=annealing_rate,
              learning_rate=learning_rate_ag)

organizer = Organizer(agent=agent,
                      env=env)

# 学習と推論
val_loss_history = []
for i in range(5):
    val_loss = organizer.step(isTrain=True, batch_size=100)
    print("i={}, val_loss={}".format(i, val_loss))
    organizer.step(isTrain=False, batch_size=10,isVisualize=False)
    val_loss_history.append(val_loss)

history_count = len(organizer.result_history)

#
# グラフ表示
#  - x軸を共有して、val_lossとtimestepのグラフを同時に表示させる
#

fig, ax1 = plt.subplots()

ax1.set_xlabel("epochs")
ax1.set_ylabel("timestep")
p1, = ax1.plot(organizer.result_history, color="r", label="timestep")

ax2 = ax1.twinx()
ax2.set_ylabel("val_loss")
p2, = ax2.plot(val_loss_history, color="b", label="val_loss")
plt.xticks(np.linspace(0, history_count, 5, endpoint=False))
plt.legend([p1, p2], ["timestep", "val_loss"])
plt.title("cartpole_play")
plt.show()

agent.save(result_path, config_path)
print("end")
