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
        self.result_history = []

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
        #
        # (1) batch_sizeだけゲームを試行
        #
        for _ in range(batch_size):
            done = False
            while not done:
                act = self.agent.policy(state=self.state, env=self.env, isTrain=isTrain)
                forward_state, reward, done, info = self.env.step(act)

                total_timestep += 1
                timestep += 1

                if done:
                    if timestep is not 200:
#                        print("rew=0/timestep:{}, isTrain={}".format(timestep, isTrain))
                        forward_reward = -1
                    else: # timestep=200の場合、強制的にゲーム終了となる。この場合のrewardは0とする
#                        print("rew=1/timestep:{}, isTrain={}".format(timestep, isTrain))
                        forward_reward = 0
                else:
                    forward_reward = 0
#                forward_reward = reward

                # ゲーム経過を記録
                self.result_ary.append([self.state, act, self.reward, forward_state, False])

                # 現在の行動を保持する
                self.state = forward_state
                self.reward = forward_reward
                if not isTrain and isVisualize:
                    self.env.render()

            # ゲーム終了時の状態を記録。(isEndrecord=True)
            self.result_ary.append([self.state, act, self.reward, forward_state, True])
            self.state = self.env.reset()
            self.reward = 0
            timestep = 0

        if not isTrain:
            average_timestep = total_timestep/batch_size
            print("試行回数:{0}, 平均:{1}".format(batch_size, average_timestep))
            self.result_history.append(average_timestep)

        #
        # (2) 1で蓄積したデータをシャッフルし、Q-Tableを更新する
        #

        if isTrain:
            '''
            print("**********\nlearning_mode\n**********")
            pbar = tqdm(total=len(result_ary))
            for ary in result_ary:
                val_loss = self.agent.learn(state       = ary[0],
                                            act         = ary[1],
                                            reward      = ary[2],
                                            forward_s   = ary[3],
                                            isEndRecord = ary[4])
                pbar.update(1)
            '''
            random.shuffle(self.result_ary)
            result_ary = self.result_ary[:batch_size]
            val_loss = self.agent.learn(result_ary)
            return val_loss
