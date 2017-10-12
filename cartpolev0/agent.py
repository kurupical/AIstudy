# common lib
import random
import numpy as np
from copy import copy

class Agent:
    '''
    ゲームプレイヤー
    '''
    def __init__(self, network, gamma=0.95, epsilon=0.2, annealing_rate=0.9995, learning_rate=0.001):
        # gamma: 時間割引率
        self.gamma = gamma
        # network: DQNのニューラルネットワーク
        self.network = network
        # epsilon: 探索率ε
        self.epsilon = epsilon
        # annealing_Rate: εの低減率(探索ごとに探索率epsilonを低減させる(焼きなまし))
        self.annealing_rate = annealing_rate
        # learning_rate: 学習率。1回の学習でQ-Tableの更新をどれだけ行うか？
        self.learning_rate = learning_rate

    def policy(self, state, env, isTrain):
        '''
        Agentの行動ポリシー(探索/活用)を定義。
        '''
        if isTrain:
            if random.random() < self.epsilon:
                # ランダム
                act = env.action_space.sample()
            else:
                act = self.predict(state)
        else:
            act = self.predict(state)

        # epsilon(探索率)を減らす
        self.epsilon = self.epsilon * self.annealing_rate
        return act

#    def learn(self, state, act, reward, forward_s, isEndRecord):
    def learn(self, result_ary):
        '''
        行動結果を学習させる
          maxev_Q : 状態forward_sで、取れるactionのうちQ値が最大となるactionを選択したときのQ値
                    (maximum Excepted Value of Q)
        '''
        
        state = result_ary[:,0]
        forward_s = result_ary[:,3]

        # Q(s,a)の計算
        Q = self.network.y.eval(session=self.network.sess, feed_dict={
                self.network.x: forward_s
            })

        # ev_Qの計算
        delta_Q = np.array([[]])
        for q in Q:
            if isEndRecord:
                # 最終レコードの場合は、その状態から先に行かない
                maxev_q = 0
                maxev_q_idx = [0,1]
            else:
                # 状態forward_sに対して、すべてのactionのQ値を計算する
                pred_q_table = self.network.y.eval(session=self.network.sess, feed_dict={
                                    self.network.x: forward_s
                                })
                # forward_stateで取れるactionのうち、最大となるQ値
                maxev_q = np.max(pred_q_table)
                maxev_q_idx = np.argmax(pred_q_table)

                # Excepted Value(期待値) = 現在の報酬(reward) +
                #                         時間割引率(gamma) * 状態forward_stateでmaxのQ値(maxev_Q)
                ev = reward + self.gamma * maxev_q

                # Todo:ミニバッチ処理対応.(1個ずつじゃなく、いっきに学習できるようにする)
                q = np.array([q]).reshape(-1, 2)

                delta_q = q
                delta_q[0, maxev_q_idx] = ev
                delta_q = np.array([delta_q]).reshape(-1, 2)
                if delta_Q.size == 0:
                    delta_Q = copy(delta_q)
                else:
                    delta_Q = np.append(delta_Q, delta_q, axis=0)

        self.network.sess.run(self.network.train_step, feed_dict={
            self.network.x: state,
            self.network.y: Q,
            self.network.t: delta_Q
        })

        val_loss = self.network.loss.eval(session=self.network.sess, feed_dict={
            self.network.x: state,
            self.network.y: Q,
            self.network.t: delta_Q
        })

        return val_loss

    def predict(self, state):
        '''
        状態stateを入力とする。
        ニューラルネットワーク化されたQ-Tableから、Q値が最大となるactionを返す。
        '''
        state = state.reshape(-1,4)
        pred_Q_table = self.network.y.eval(session=self.network.sess, feed_dict={
                            self.network.x: state
                       })

        # Q値が最大となるactionを返す
        # print("state={}, pred_Q_table={}".format(state, pred_Q_table))
        return np.argmax(pred_Q_table)
