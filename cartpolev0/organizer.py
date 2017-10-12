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


    def step(self, isTrain, batch_size):
        '''
        (1) batch_sizeだけゲームを試行し、(state,act,reward,forward_s,isEndRecord)を蓄積する
        (2) 1で蓄積したデータをシャッフルし、Q-Tableを更新する
        '''

        timestep = 0
        total_timestep = 0
        state_ary = np.array([[]])
        act_ary = np.array([[]])
        reward_ary = np.array([[]])
        forward_state_ary = np.array([[]])
        isEndRecord_ary = np.array([[]])

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
                        print("rew=0/timestep:{}, isTrain={}".format(timestep, isTrain))
                        forward_reward = -1
                    else: # timestep=200の場合、強制的にゲーム終了となる。この場合のrewardは0とする
                        print("rew=1/timestep:{}, isTrain={}".format(timestep, isTrain))
                        forward_reward = 0
                else:
                    forward_reward = 0
#                forward_reward = reward

                # ゲーム経過を記録
                if state_ary.size == 0:
                    state_ary = np.array(self.state)
                    act_ary = np.array(act)
                    reward_ary = np.array(self.reward)
                    forward_state_ary = np.array(forward_state)
                    isEndRecord_ary = np.array(False)
                else:
                    state_ary = np.append(state_ary, self.state, axis=0)
                    print(act_ary)
                    print(act)
                    act_ary = np.append(act_ary, act, axis=0)
                    reward_ary = np.append(reward_ary, self.reward, axis=0)
                    forward_state_ary = np.append(forward_state_ary, forward_state, axis=0)
                    isEndRecord_ary = np.append(isEndRecord_ary, False, axis=0)

                # 現在の行動を保持する
                self.state = forward_state
                self.reward = forward_reward
                if not isTrain:
                    self.env.render()

            # ゲーム終了時の状態を記録。(isEndrecord=True)
            state_ary = np.append(state_ary, state, axis=0)
            act_ary = np.append(act_ary, act, axis=0)
            reward_ary = np.append(reward_ary, reward, axis=0)
            forward_state_ary = np.append(forward_state_ary, forward_state, axis=0)
            isEndRecord_ary = np.append(isEndRecord_ary, True, axis=0)

            self.state = self.env.reset()
            self.reward = 0
            timestep = 0

        if not isTrain:
            print("試行回数:{0}, 平均:{1}".format(batch_size, total_timestep/batch_size))

        #
        # (2) 1で蓄積したデータをシャッフルし、Q-Tableを更新する
        #

        if isTrain:
            random.shuffle(state_ary, act_ary, reward_ary, forward_state_ary, isEndRecord_ary)
            print("**********\nlearning_mode\n**********")
            '''
            pbar = tqdm(total=len(result_ary))
            for ary in result_ary:
                val_loss = self.agent.learn(state       = ary[0],
                                            act         = ary[1],
                                            reward      = ary[2],
                                            forward_s   = ary[3],
                                            isEndRecord = ary[4])
                pbar.update(1)
            '''
            val_loss = self.agent.learn(result_ary)
            return val_loss
