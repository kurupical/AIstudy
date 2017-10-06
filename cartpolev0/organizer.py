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
        #
        # (1) batch_sizeだけゲームを試行
        #
        for _ in range(batch_size):
            done = False
            while not done:
                act = self.agent.policy(state=self.state, env=self.env, isTrain=isTrain)
                forward_state, _, done, info = self.env.step(act)

                if done:
                    forward_reward = -1
                else:
                    forward_reward = 0

                # ゲーム経過を記録
                result_ary.append([self.state, act, self.reward, forward_state, False])

                # 現在の行動を保持する
                self.now_state = forward_state
                self.now_reward = forward_reward
                if not isTrain:
                    timestep += 1
                    self.env.render()

            # ゲーム終了時の状態を記録。(isEndrecord=True)
            result_ary.append([self.state, act, self.reward, forward_state, True])

            self.now_state = self.env.reset()
            self.now_reward = 0

        if not isTrain:
            print("試行回数:{0}, 平均:{1}".format(batch_size, timestep/batch_size))

        #
        # (2) 1で蓄積したデータをシャッフルし、Q-Tableを更新する
        #

        if isTrain:
            random.shuffle(result_ary)

            for ary in result_ary:
                self.agent.learn(state       = ary[0],
                                 act         = ary[1],
                                 reward      = ary[2],
                                 forward_s   = ary[3],
                                 isEndRecord = ary[4])
