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
# モデル読み込み先ディレクトリ
load_path = "model/model.ckpt"

# ゲームの作成
env = gym.make('CartPole-v0')
state = env.reset()
n_in = state.size
n_out = env.action_space.n # 取れるactionの種類

# 各種パラメータの取得
## network param
n_l1hidden = config['n_l1hidden']
n_l2hidden = config['n_l2hidden']
learning_rate_nw = config['learning_rate_nw']
isLoadModel = config['isLoadModel']
## agent param
gamma = config['gamma']
epsilon = config['epsilon']
annealing_rate = config['annealing_rate']
min_epsilon = config['min_epsilon']

# 環境構築
network = DQN(n_l1hidden=n_l1hidden,
              n_l2hidden=n_l2hidden,
              n_in=n_in,
              n_out=n_out,
              learning_rate=learning_rate_nw)

if isLoadModel:
    print("\n\n\nLoad_Model!\n\n\n")
    network.load(load_path)

agent = Agent(network=network,
              gamma=gamma,
              epsilon=epsilon,
              annealing_rate=annealing_rate,
              min_epsilon=min_epsilon)

organizer = Organizer(agent=agent,
                      env=env)

# 学習と推論
val_loss_history = []
for i in range(1000):
    train_ave, val_loss = organizer.step(isTrain=True, batch_size=32)
    test_ave = organizer.step(isTrain=False, batch_size=10, isVisualize=False)
    print("i={}, val_loss={:.6f}, average_timestep=[train:{:.2f}, test:{:.2f}]".format(i, val_loss, train_ave, test_ave))
    if test_ave >= 195:
        break
    val_loss_history.append(val_loss)

history_count_train = len(organizer.result_history_train)
history_count_test = len(organizer.result_history_test)

#
# グラフ表示
#  - x軸を共有して、val_lossとtimestepのグラフを同時に表示させる
#

fig, ax1 = plt.subplots()

ax1.set_xlabel("epochs")
ax1.set_ylabel("timestep")
p1, = ax1.plot(organizer.result_history_train, color="g", label="timestep_train")
p2, = ax1.plot(organizer.result_history_test, color="r", label="timestep_test")

ax2 = ax1.twinx()
ax2.set_ylabel("val_loss")
p3, = ax2.plot(val_loss_history, color="b", linestyle="dashed", label="val_loss")
plt.xticks(np.linspace(0, history_count_train, 5, endpoint=False))
plt.legend([p1, p2, p3], ["timestep_train","timestep_test", "val_loss"])
plt.title("cartpole_play")
plt.show()

agent.save(result_path, config_path)
plt.savefig(result_path + "graph.png")
print("end")
