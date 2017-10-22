# common lib
import random
import numpy as np
from tqdm import tqdm

class Organizer:
    '''
    ゲームを進めるためのクラス
    '''
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.state = env.reset()
        self.reward = 0
        self.result_ary = []

        self.result_history_train = []
        self.result_history_test = []

        # 前回の平均timestep
        self.before_ave = 0

    def step(self, isTrain, batch_size, isVisualize=True):
        '''
        (1) batch_sizeだけゲームを試行し、(state,act,reward,forward_s,isEndRecord)を蓄積する
        (2) 1で蓄積したデータをシャッフルし、Q-Tableを更新する

        isTrain : 訓練モードであるかどうか
        batch_size : 1stepあたりに試行する回数
        isVisualize : 訓練モード時に、ゲームを表示するか
        '''

        timestep = 0
        total_timestep = 0
        insert_count = 0 # debug用 self.result_aryにappendされた件数
        act_count = 0
        act_choose1_count = 0
        #
        # (1) batch_sizeだけゲームを試行
        #
        for i in range(batch_size):
            w_result_ary = []
            done = False
            while not done:
                act = self.agent.policy(state=self.state, env=self.env, isTrain=isTrain)
                forward_state, reward, done, info = self.env.step(act)

                total_timestep += 1
                timestep += 1
                act_count += 1
                act_choose1_count += act

                if done:
                    if timestep is not 200:
#                        print("rew=0/timestep:{}, isTrain={}".format(timestep, isTrain))
                        forward_reward = -1
                    else: # timestep=200の場合、強制的にゲーム終了となる。この場合のrewardは0とする
#                        print("rew=1/timestep:{}, isTrain={}".format(timestep, isTrain))
                        forward_reward = 0.005
                else:
                    forward_reward = 0.005
#                forward_reward = reward

                if isTrain:
                    # ゲーム経過を記録
                    w_result_ary.append([self.state, act, self.reward, forward_state, False])
                # 現在の行動を保持する
                self.state = forward_state
                self.reward = forward_reward

                if not isTrain and isVisualize:
                    self.env.render()

            # ゲーム終了時の状態を記録。(isEndrecord=True)
            w_result_ary.append([self.state, act, self.reward, forward_state, True])

            # 成績が前回平均以上のデータは記録する。
            # それ以外のデータは、一定の確率で記録する。
            if isTrain:
                if self.before_ave < timestep or random.random() < 0.:
            #    if self.before_ave < timestep:
                    for ary in w_result_ary:
                        self.result_ary.append(ary)
                        insert_count += 1

            self.state = self.env.reset()
            self.reward = 0
            timestep = 0

        average_timestep = total_timestep/batch_size
        print("isTrain={}, act_total={}, choose_1={}".format(isTrain, act_count, act_choose1_count))
        # average_timestepの記録
        if isTrain:
            self.result_history_train.append(average_timestep)
        else:
            self.result_history_test.append(average_timestep)
        if isTrain:
            if self.before_ave < average_timestep:
                self.before_ave = average_timestep

        print("insert_count={}, result_ary_length={}, average={}".format(insert_count, len(self.result_ary), self.before_ave))
        #
        # (2) 1で蓄積したデータをシャッフルし、Q-Tableを更新する
        #

        if isTrain:
            random.shuffle(self.result_ary)

            # [batch_sizeの100倍]のデータを保持する
            if len(self.result_ary) < 3000:
                result_ary = self.result_ary[:300]
            else:
                self.result_ary = self.result_ary[:3000]
                result_ary = self.result_ary[:300]
            val_loss = self.agent.learn(result_ary)
            return average_timestep, val_loss
        else:
            return average_timestep
