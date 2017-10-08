import random

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

        result_ary = []
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

                '''
                if done:
                    if reward == 1:
                        print("rew=0/timestep:{}".format(timestep))
                        forward_reward = -1
                    else:
                        print("rew=1/timestep:{}".format(timestep))
                        forward_reward = 0
                else:
                    forward_reward = 0
                '''
                forward_reward = reward

                # ゲーム経過を記録
                result_ary.append([self.state, act, self.reward, forward_state, False])

                # 現在の行動を保持する
                self.state = forward_state
                self.reward = forward_reward
                if not isTrain:
                    self.env.render()

            # ゲーム終了時の状態を記録。(isEndrecord=True)
            result_ary.append([self.state, act, self.reward, forward_state, True])

            self.state = self.env.reset()
            self.reward = 0
            timestep = 0

        if not isTrain:
            print("試行回数:{0}, 平均:{1}".format(batch_size, total_timestep/batch_size))

        #
        # (2) 1で蓄積したデータをシャッフルし、Q-Tableを更新する
        #

        if isTrain:
            random.shuffle(result_ary)

            for ary in result_ary:
                val_loss = self.agent.learn(state       = ary[0],
                                            act         = ary[1],
                                            reward      = ary[2],
                                            forward_s   = ary[3],
                                            isEndRecord = ary[4])
            return val_loss
