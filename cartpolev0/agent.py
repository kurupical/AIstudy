# common lib
import random
import numpy as np
import shutil
import os
from copy import copy

class Agent:
    '''
    ゲームプレイヤー
    '''
    def __init__(self, network, gamma=0.95, epsilon=0.2, annealing_rate=0.9995, learning_rate=0.001, min_epsilon=0.1):
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
        # min_epsilon: 探索率の最小値ε
        self.min_epsilon = min_epsilon

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

        return act

#    def learn(self, state, act, reward, forward_s, isEndRecord):
    def learn(self, result_ary):
        '''
        行動結果を学習させる
          maxev_Q : 状態forward_sで、取れるactionのうちQ値が最大となるactionを選択したときのQ値
                    (maximum Excepted Value of Q)
        '''

        state_ary = np.array([[]])
        Q_ary = np.array([[]])
        delta_Q_ary = np.array([[]])
        # test
        ix = 0
        ix_total = 0
        # test
        for state, act, reward, forward_s, isEndRecord in result_ary:
            state = state.reshape(-1,4)
            forward_s = forward_s.reshape(-1,4)

            # Q(s,a)の計算
            Q = self.network.y.eval(session=self.network.sess, feed_dict={
                    self.network.x: state
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
                    ix_total += 1
                    ix += np.argmax(pred_q_table)
                ev = reward + self.gamma * maxev_q
                # print("forward_s={}, pred_q={}, ev={},end?={}".format(forward_s,pred_q_table, ev, isEndRecord))
                # Todo:ミニバッチ処理対応.(1個ずつじゃなく、いっきに学習できるようにする)
                q = np.array([q]).reshape(-1, 2)

                delta_q = q
                delta_q[0, maxev_q_idx] = ev
                delta_q = np.array([delta_q]).reshape(-1, 2)
                if delta_Q.size == 0:
                    delta_Q = copy(delta_q)
                else:
                    delta_Q = np.append(delta_Q, delta_q, axis=0)

                if state_ary.size == 0:
                    state_ary = np.array(state)
                    Q_ary = np.array(Q)
                    delta_Q_ary = np.array(delta_Q)
                else:
                    state_ary = np.append(state_ary, state, axis=0)
                    Q_ary = np.append(Q_ary, Q, axis=0)
                    delta_Q_ary = np.append(delta_Q_ary, delta_Q, axis=0)

        print("test: total={}, choose_1={}".format(ix_total,ix))
        self.network.sess.run(self.network.train_step, feed_dict={
            self.network.x: state_ary,
            self.network.y: Q_ary,
            self.network.t: delta_Q_ary
        })

        val_loss = self.network.loss.eval(session=self.network.sess, feed_dict={
            self.network.x: state_ary,
            self.network.y: Q_ary,
            self.network.t: delta_Q_ary
        })

        # epsilon(探索率)を減らす
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon - self.annealing_rate

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
        # print("state={}, pred_Q_table={}, max_index={}".format(state, pred_Q_table, np.argmax(pred_Q_table[0])))
        return np.argmax(pred_Q_table[0])

    def save(self, result_path, config_path):
        os.mkdir(result_path)
        # モデルの保存
        self.network.save(result_path + "model.ckpt")
        # コンフィグファイルの保存
        result_config_path = result_path + "config.json"
        shutil.copy2(config_path, result_config_path)
