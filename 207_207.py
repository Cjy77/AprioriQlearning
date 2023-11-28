import pandas as pd
import time,random,pickle,pathlib
import os

import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
# import numpy

import warnings
# 去除掉FutureWarning的警告，与import warnings 模块相关
warnings.simplefilter(action='ignore', category=FutureWarning)
# 去除掉DeprecationWarning的警告，与import warnings 模块相关
# warnings.filterwarnings("ignore", category=DeprecationWarning)
import time

# 记录开始时间
start_time = time.time()

class Maze(tk.Tk):
    '''环境类（GUI）'''
    UNIT = 70  # pixels
    MAZE_H = 3  # grid height
    MAZE_W = 9  # grid width

    def __init__(self):
        '''初始化'''
        super().__init__()

        self.title('找矿')
        h = self.MAZE_H * self.UNIT
        w = self.MAZE_W * self.UNIT
        self.geometry('{0}x{1}'.format(h * 3, w))  # 窗口大小
        self.withdraw()  # 隐藏窗口
        self.canvas = tk.Canvas(self, bg='white', height=h, width=w)
        # 画网格
        for c in range(0, w, self.UNIT):
            self.canvas.create_line(c, 0, c, h)
        for r in range(0, h, self.UNIT):
            self.canvas.create_line(0, r, w, r)

        self._draw_rect(0, 1, 'blue')

        #画奖励
        self._draw_rect(8, 1, 'yellow')
        # 画玩家(保存!!)
        self.rect = self._draw_rect(0, 1, 'red')
        self.canvas.pack()  # 显示画作！

    def _draw_rect(self, x, y, color):
        '''画矩形，  x,y表示横,竖第几个格子'''
        padding = 5  # 内边距5px，参见CSS
        coor = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x + 1) - padding,
                self.UNIT * (y + 1) - padding]
        return self.canvas.create_rectangle(*coor, fill=color)

    def move_to(self, state, delay=0.001):
        '''玩家移动到新位置，根据传入的状态'''
        coor_old = self.canvas.coords(self.rect)  # 形如[5.0, 5.0, 35.0, 35.0]（第一个格子左上、右下坐标）
        x, y = state % 9,state // 9    # 横竖第几个格子
        padding = 5  # 内边距5px，参见CSS
        coor_new = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x + 1) - padding,
                    self.UNIT * (y + 1) - padding]
        dx_pixels, dy_pixels = coor_new[0] - coor_old[0], coor_new[1] - coor_old[1]  # 左上角顶点坐标之差
        self.canvas.move(self.rect, dx_pixels, dy_pixels)
        self.update()  # tkinter内置的update!
        time.sleep(delay)

class Agent(object):
    '''个体类'''

    def __init__(self, alpha=0.1, gamma=0.9):

        '''初始化'''
        self.states = range(27)  # 状态集。0~35 共36个状态
        # self.actions = list('udlrwxyz')  # 动作集。上下左右  4个动作,w右上，x右下，y左上，z左下
        self.actions = list('udrwx')  # 动作集。上下左右  4个动作,w右上，x右下，y左上，z左下


        #手动输入初始奖励值
        # self.rewards = [-10, 0, 0, -1, 0, 0, -2,  -4,  -10,
        #                  0,  0, -1, 0, -4, -2, 0,  0,   10,
        #                 -10, -1, -1, -2,0,-4, -4, -10, -10]

        # self.rewards = [-10, 0, 0, -1, 0, 0, -2,  -4,  -10,
        #                  0,  0, -1, 0, -4, -2, 0,  0,   10,
        #                 -10, -1, -1, -2,0,-4, -4, -10, -10]

        #w右上，x右下，y左上，z左下
        #[0,  1,  2,  3,  4,  5,  6 , 7,  8
        #9,  10, 11, 12, 13, 14, 15, 16, 17
        #18, 19, 20, 21, 22, 23, 24, 25, 26]



        # self.hell_states = [0, 1,2,3,4,5,6,7,8, 18,19,20,21,22,23,24,25,26]  # 陷阱位置
        # self.hell_states = [0, 8, 18, 25, 26]  # 陷阱位置
        self.rewards = every_reward
        self.alpha = alpha
        self.gamma = gamma
        #定义初始q_table表
        self.q_table = pd.DataFrame(data=[[0 for _ in self.actions] for _ in self.states],
                                        index=self.states,
                                        columns=self.actions)  # 定义Q-table，初始值都是0
        # self.q_table = pd.read_csv("./export-result/q_table.csv")  # 定义指定的q_table
        # self.q_table = pd.read_csv('./export-result/'+ str(Num_QTable) + '-' + "q_table.csv")  # 指定的Qtable时的参考

    def choose_action(self, state, epsilon=0.95):
        '''选择相应的动作。根据当前状态，随机或贪婪，按照参数epsilon'''
        # if (random.uniform(0,1) > epsilon) or ((self.q_table.ix[state] == 0).all()):  # 探索
        if random.uniform(0, 1) > epsilon:  # 探索
            action = random.choice(self.get_valid_actions(state))
            randomTrue = 9
        else:
            # action = self.q_table.iloc[state].idxmax() # 利用 当有多个最大值时，会锁死第一个！
            # action = self.q_table.ix[state].filter(items=self.get_valid_actions(state)).idxmax() # 重大改进！然而与上面一样
            s = self.q_table.loc[state].filter(items=self.get_valid_actions(state))
            action = random.choice(s[s == s.max()].index)  # 从可能有多个的最大值里面随机选择一个！
            randomTrue = 0
        return action,randomTrue

    def get_next_state(self, state, action):
        '''对状态执行动作后，得到下一状态，修行列变化的时候需要改变'''
        # u,d,l,r,n = -6,+6,-1,+1,0
        if state % 9 != 8 and action == 'r':  # 除最后一列，皆可向右(+1)
            next_state = state + 1
        elif state % 9 != 0 and action == 'l':  # 除最前一列，皆可向左(-1)
            next_state = state - 1
        elif state // 9 != 2 and action == 'd':  # 除最后一行，皆可向下(+2)
            next_state = state + 9
        elif state // 9 != 0 and action == 'u':  # 除最前一行，皆可向上(-2)
            next_state = state - 9

        elif state // 9 != 0 and state % 9 != 8 and action == 'w':  # 除第一行，除最后一列，皆可向右上
            next_state = state - 9 + 1

        elif state // 9 != 2 and state % 9 != 8 and action == 'x':  # 除最后一行，除最后一列，皆可向右下
            next_state = state + 9 + 1

        # elif state // 9 != 0 and state % 9 != 0 and action == 'y':  # 除第一行，除第一列，皆可向左上
        #     next_state = state - 9 - 1
        #
        # elif state // 9 != 2 and state % 9 != 0 and action == 'z':  # 除最后一行，除第一列，皆可向左下
        #     next_state = state + 9 - 1

        else:
            next_state = state
        # print(next_state)
        return next_state

    def get_q_values(self, state):
        '''取给定状态state的所有Q value'''
        q_values = self.q_table.loc[state, self.get_valid_actions(state)]
        # print(q_values)
        return q_values
    # 学些率alpha为0.1
    def update_q_value(self, state, action, next_state_reward, next_state_q_values):
        '''更新Q value，根据贝尔曼方程'''
        self.q_table.loc[state, action] += self.alpha * (
                next_state_reward + self.gamma * next_state_q_values.max() - self.q_table.loc[state, action])

    # 学些率alpha为0.05
    def update_q_value_0(self, state, action, next_state_reward, next_state_q_values):
        '''更新Q value，根据贝尔曼方程'''
        self.q_table.loc[state, action] += 0.05 * (
                next_state_reward + self.gamma * next_state_q_values.max() - self.q_table.loc[state, action])
    # 学些率alpha为0.2
    def update_q_value_2(self, state, action, next_state_reward, next_state_q_values):
        '''更新Q value，根据贝尔曼方程'''
        self.q_table.loc[state, action] += 2 * self.alpha * (
                next_state_reward + self.gamma * next_state_q_values.max() - self.q_table.loc[state, action])
    def get_valid_actions(self, state):
        '''
        取当前状态下的合法动作集合
        global reward
        valid_actions = reward.ix[state, reward.ix[state]!=0].index
        return valid_actions
        行列变化的时候需要改变，注意%-取余和//的使用，可以用程序验证一下
        '''
        valid_actions = set(self.actions)
        # if state % 9 == 8 :  # 最后一列，则
        #     valid_actions -= set(['r','w','x'])  # 无向右的动作
        # if state % 9 == 0:  # 最前一列，则
        #     valid_actions -= set(['l','y','z'])  # 去掉向左的动作
        # if state // 9 == 2:  # 最后一行，则
        #     valid_actions -= set(['d','x','z'])  # 无向下
        # if state // 9 == 0:  # 最前一行，则
        #     valid_actions -= set(['u','w','y'])  # 无向上

        if state % 9 == 8 :  # 最后一列，则
            valid_actions -= set(['r','w','x'])  # 无向右的动作
        # if state % 9 == 0:  # 最前一列，则
        #     valid_actions -= set(['l','y','z'])  # 去掉向左的动作
        if state // 9 == 2:  # 最后一行，则
            valid_actions -= set(['d','x'])  # 无向下
        if state // 9 == 0:  # 最前一行，则
            valid_actions -= set(['u','w'])  # 无向上
        return list(valid_actions)

    #为learn方法设置形参episode=100000, epsilon=0.95，plot 和 save变量的名称，简称p_s，用于保存图和csv文件
    def learn(self, env=None, train_onesample_num=0,episode=100000, epsilon=0.95, p_s=0):
        '''q-learning算法'''
        print('Agent is learning...')
        # print(f'本次学习所用的初始奖励值：\n{self.rewards}')
        reward_list = []  # 记录每一个episode的奖励总值
        # count_list = []
        all_agent_loc = []

        epo_avg = []
        epo_std = []
        every_epo_mean = []
        add_every_epo_reward = []
        #
        # self.q_table = pd.DataFrame(data=[[0 for _ in self.actions] for _ in self.states],
        #                                 index=self.states,
        #                                 columns=self.actions)  # 定义Q-table初始值都是0
        for i in range(episode):
            """从最左边的位置开始，起始位置需要修改"""
            current_state = self.states[9]

            if env is not None:  # 若提供了环境，则重置之！
                env.move_to(current_state)
            every_epo_reward = []
            agent_loc = []

            reward_sum = 0
            j = 0
            # print('q值表：\n', self.q_table)
            next_state_sets=[] # 下一个状态的集合
            # next_state_reward_sets = [] #下一个奖励值的集合
            while current_state != self.states[17]:  # 从当前的合法动作中，随机（或贪婪）的选一个作为 当前动作

                ''' -------------------------调用choose_action__________________________'''
                current_action ,randomTrue = self.choose_action(current_state, epsilon)  # 按一定概率，随机或贪婪地选择

                ''' -------------------------调用choose_action__________________________'''

                # print('当前动作选择为：', current_action)
                '''---------------调用get_next_state   执行当前动作，得到下一个状态（位置）------------'''
                next_state = self.get_next_state(current_state, current_action)
                '''---------------调用get_next_state   执行当前动作，得到下一个状态（位置）------------'''
                for current_action_x in ('u','d','r','w','x'):
                    next_state_one = self.get_next_state(current_state, current_action_x)
                    next_state_sets.append(next_state_one)
                '''---------------从numpy的ndarray(多维数组)得到下一个状态（位置）------------'''
                next_state_reward = self.rewards[next_state]

                # # 通过索引列表获取ndarry中的多个指定位置的多个值，其中列表是一个状态可能去往任意一个状态的集合
                next_state_reward_sets = self.rewards[next_state_sets]
                '''___________________________________________________________________'''
                '''___________________________________________________________________'''
                '''___________________________________________________________________'''
                #获取ndarry中也即下一状态所有奖励值中第k大的值
                k=2
                '''___________________________________________________________________'''
                '''___________________________________________________________________'''
                '''___________________________________________________________________'''
                next_state_reward_sets_k = np.partition(next_state_reward_sets, -k)[-k]

                # print('下一个位置的奖励：', next_state_reward)
                '''--------------调用get_q_values  取下一个状态所有的Q-value，待取其最大值--------------------'''
                next_state_q_values = self.get_q_values(next_state)
                '''--------------调用get_q_values  取下一个状态所有的Q-value，待取其最大值--------------------'''


                if (randomTrue==9) and (next_state_reward > next_state_reward_sets_k):
                    '''------调用update_q_value  根据贝尔曼方程更新Q-table中当前状态-动作对应的Q-value-------------'''
                    self.update_q_value_2(current_state, current_action, next_state_reward, next_state_q_values)
                    '''------调用update_q_value  根据贝尔曼方程更新Q-table中当前状态-动作对应的Q-value-------------'''
                elif (randomTrue==9) and (next_state_reward == next_state_reward_sets_k):
                    self.update_q_value(current_state, current_action, next_state_reward, next_state_q_values)
                else:
                    self.update_q_value_0(current_state, current_action, next_state_reward, next_state_q_values)


                agent_loc.append(current_state)
                # print(agent_loc)
                current_state = next_state
                # print('当前动作选择为：', current_action,'下一个位置的奖励：', next_state_reward,'下一个状态位置：',next_state)
                # print('q值表：',self.q_table)
                if env is not None:  # 若提供了环境，则更新之！
                    env.move_to(current_state)

#记录每一次到达终点获取得reward的平均值,每回合累加奖励值，走了多少回合，平均值，一共走了多少步，记录最终路径
                '''  -----记录行动后的累加奖励值的和，是一个数-----'''
                reward_sum += next_state_reward
                '''  -----记录行动后的累加奖励值的和，是一个数-----'''

                # print('reward_sum:',reward_sum)
                '''------记录每一个reward_sum并形成一个序列----------'''
                reward_list.append(reward_sum)
                '''------记录每一个reward_sum并形成一个序列----------'''
                # print('reward_list:', reward_list)
                '''------与reward_list相同----------'''
                every_epo_reward.append(reward_sum)
                '''------与reward_list相同----------'''
                # print('every_epo_reward:',every_epo_reward)
                # agent_loc.append(current_state)
                j += 1
                # count_list.append(j)
            # print('------------------------到达终点的次数：', i)

            '''  ---all_agent_loc----与while相对应，表示记录一回合的位置--------  '''
            all_agent_loc.append(agent_loc)
            '''  ---all_agent_loc----与while相对应，表示记录一回合的位置--------  '''
            # print('every_epo_reward', every_epo_reward)
            add_every_epo_reward.append(every_epo_reward)
            # print('add_every_epo_reward:',add_every_epo_reward)
            # print('运行到第几组数据:',iii)
            # pd.DataFrame(add_every_epo_reward).to_csv(save_file + '/' + str(iii) +'-'+ 'add_every_epo_reward.csv')

            # pd.DataFrame(all_agent_loc).to_csv(save_file + '/' + str(iii) +'-'+ 'agent_location.csv')

            every_epo_mean.append(mean(every_epo_reward))
            # print('max:',np.array(every_epo_reward).max())
            # print('min:',np.array(every_epo_reward).min())
            # print('epo_avg上面的reward_list:', reward_list)
            # print('mean(reward_list):' , mean(reward_list))
            epo_avg.append(mean(reward_list))
            # print('epo_avg:', epo_avg)
            # print('reward_list',reward_list)
            epo_std.append(std(reward_list))
            # avg_list = []
            # for i in range(len(reward_list)):
            #     avg_list.append(reward_list[i] / count_list[i])

        # plt.plot(avg_list)
        #此处为画图
        plt.plot(epo_avg,label='epo_avg')
        #plt.plot(every_epo_mean,label='every_epo_mean')
        # 添加图例
        # plt.legend()
        plt.savefig(save_file + '/' + str(iii) + '-' +str(train_onesample_num)+'-' +  str(p_s)+ '.png' , dpi = 350)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Reward')
        # plt.show()
        # print('avg_list',len(avg_list))

        pd.DataFrame(every_epo_mean).to_csv(save_file + '/' + str(iii) +
                                            '-'+str(train_onesample_num)+'-'+ str(p_s)+'-'+'every_epo_mean.csv')

        # pd.DataFrame(epo_avg,epo_std).to_csv(save_file + '/' + str(iii) +'-'+'epo_avg_std.csv')

        # pd.DataFrame(reward_list).to_csv(save_file + '/' + str(iii) +'-'+'reward_list.csv')

        self.q_table.to_csv(save_file + '/' + str(iii) +
                            '-'+str(train_onesample_num)+ '-'+str(p_s) + '-' + 'q_table.csv')

    def save_policy(self):
        '''保存Q table'''
        with open('q_table.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.q_table, f, pickle.HIGHEST_PROTOCOL)

    def load_policy(self):
        '''导入Q table'''
        with open('q_table.pickle', 'rb') as f:
            self.q_table = pickle.load(f)

    def test(self):
        '''测试agent是否已具有智能'''
        count = 0
        current_state = self.states[9]
        while current_state != self.states[17]:
            current_action = self.choose_action(current_state, 1.)  # 1., 贪婪
            next_state = self.get_next_state(current_state, current_action)
            current_state = next_state
            count += 1

            if count > 27:  # 没有在36步之内走出迷宫，则
                return False  # 无智能

        return True  # 有智能
    # 添加end_action标识最后行走的路径用于保存
    def play(self, env=None, train_onesample_num=0,delay=0.001, end_action=0,):
        '''玩游戏，使用策略'''
        assert env != None, 'Env must be not None!'

        if not self.test():  # 若尚无智能，则
            if pathlib.Path("q_table.pickle").exists():
                self.load_policy()
            else:
                print("I need to learn before playing this game.")
                self.learn(env, episode=1000000000, epsilon=0.9)
                self.save_policy()

        print('Agent is playing...')
        # print(f"查看目前使用的Qtable\n:{self.q_table}")
        play_agent_loc = [] #记录完全最大奖励值的路径
        q_values = []  # 用于存储最大奖励值路径所对应的Q值
        ini_rewards = []  # 用于存储最大奖励值路径所对应的初始奖励值
        current_state = self.states[9]
        env.move_to(current_state, delay)
        while current_state != self.states[17]:
            current_action ,randomTrue = self.choose_action(current_state, 1.)  # 1., 贪婪
            next_state = self.get_next_state(current_state, current_action)
            current_q_value = self.q_table.loc[current_state,current_action]  # 获取当前状态和动作对应的Q值
            q_values.append(current_q_value)  # 将Q值添加到列表中

            # 获取当前状态对应的奖励值
            ini_reward_nums = every_reward[current_state]
            ini_rewards.append(ini_reward_nums)  # 将奖励值添加到列表中

            every_reward[current_state] = 0
            every_reward[9] = 0
            update_ini_reward = every_reward
            current_state = next_state
            play_agent_loc.append(current_state)
            env.move_to(current_state, delay)
        ini_rewards = ini_rewards[1:]  # 删除第一个位置，即9，对应的奖励值 同时到达终点也即矿点的奖励值本身就不在其中
        pd.DataFrame(play_agent_loc).to_csv(save_file + '/' + str(iii) + '-'
                                            + str(train_onesample_num) + '-'
                                            + str(end_action) + '-'
                                            + '训练后完全按照Qtable最大值没有随机的形成的路径.csv')

        # pd.DataFrame(update_ini_reward).to_csv('替换后的初始奖励值.csv')

        # print(f"完全按照Q-Table行走的路径：'{play_agent_loc}'")
        # print(f"路径对应的Q值：{q_values}")  # 输出路径对应的Q值
        # print(f"路径对应的初始奖励值：{ini_rewards}")  # 输出路径对应的奖励值
        print('\nCongratulations, Agent got it!')

        return ini_rewards ,update_ini_reward,q_values  # 返回Q值列表

    # def forward(self,endnum):
    #     # print(f'查看输入的初始的数值是多少： {endnum}')
    #     # 设置随机种子
    #     torch.manual_seed(0)
    #     input_size = len(endnum)
    #     # 定义隐藏层到输出层的线性变换，输入维度是50，输出维度是1（单个数值）
    #     fc1 = nn.Linear(input_size, 1 , bias=False)
    #
    #     endnum = fc1(endnum)  # 通过第二层，无激活函数
    #     # print(f"输出一个数的结果：{endnum}")
    #     return endnum



if __name__ == '__main__':
    # 设置到达终点的次数，也为训练的回合 epoch
    episode_num = 1500
    # 设置随机行动的概率 1-epsilon_num
    epsilon_num = 0.95
    # 设置训练一个单一样本的次数
    train_onesample_nums = 15
    ''''_____________________________________________________________________________________________'''
    ''''_____________________________________________________________________________________________'''
    # 设置保存文件夹名称
    save_file = 'Qresult'
    ''''_____________________________________________________________________________________________'''
    ''''_____________________________________________________________________________________________'''

    data_log = "运行次数保存.txt"

    if os.path.exists(save_file + '/' + data_log):
        os.remove(save_file + '/' + data_log)
        print(f"文件 '{data_log}' 已成功删除。")

    if os.path.exists('one.csv'):
        os.remove('one.csv')
        print(f"文件 '{'one.csv'}' 已成功删除。")
    # 创建文件夹（如果不存在）
    if not os.path.exists(save_file):
        os.mkdir(save_file)
        print(f"文件夹 {save_file} 创建成功")

    ''''_____________________________________________________________________________________________'''
    ''''_____________________________________________________________________________________________'''
    data_all = pd.read_csv("207Fault.csv")

    data_all = data_all[195:207]
    '''________________________________________________________________________________________________'''
    '''________________________________________________________________________________________________'''

    TenMean_Q_sets = []   #保存单个样品的循环训练后的  均值，初代循环次数为10
    TenStd_Q_sets = []    #保存单个样品的循环训练后的  标准差，初代循环次数为10
    TenMax_Q_sets = []      #保存最大值
    TenMin_Q_sets = []      #保存最小值

    for iii in range(data_all.shape[0]):      #data_all.shape[0]代表行，有几个样本也就是有几行则循环多少次
        ini_rewards_mean_sets = []  # 保存初始奖励值  均值  的集合
        ini_rewards_sum_sets = []  # 保存初始奖励值  和  的集合
        Q_mean_sets = []  # 保存初始奖励值  均值  的集合
        Q_sum_sets = []
        every_reward = np.array(data_all.iloc[iii, :])
        for train_onesample_num in range(train_onesample_nums):   #一个样本循环train_onesample_nums次
            # 只是指示for循环里面的内容，不包括与for缩进一致的内容，
            # for里面的循环是将一个样品连续更新两次 Qtable，其中包括替换初始值
            every_reward = np.array(data_all.iloc[iii, :])
            for simple_1 in range(2):
                if simple_1 == 0:
                    env = Maze()  # 环境
                    agent = Agent()  # 个体（智能体）

                    agent.learn(env, train_onesample_num, episode=episode_num, epsilon=epsilon_num, p_s=0)  # 先学习
                    agent.save_policy()
                    agent.load_policy()
                    ini_rewards_1, update_ini_reward_1,q_values_1 =agent.play(env,train_onesample_num,end_action=0)  # 再玩耍

                else:
                    # print(f'更新后的初次奖励值，用于第二次训练：\n {every_reward}')
                    env = Maze()  # 环境
                    agent = Agent()  # 个体（智能体）
                    agent.learn(env, train_onesample_num, episode=episode_num, epsilon=epsilon_num, p_s=1)  # 先学习
                    agent.save_policy()
                    agent.load_policy()
                    ini_rewards_2, update_ini_reward_2,q_values_2 = agent.play(env,train_onesample_num, end_action=1) # 再玩耍
                    print(f"单样品第{train_onesample_num}次循环结束")
            # 将第一次和第二次提取的初始奖励值进行合并

            ini_rewards_merge = ini_rewards_1+ini_rewards_2

            #将第一次和第二次提取的 Q 值进行合并
            ini_Q_merge = q_values_1 + q_values_2

            # 去除掉初始奖励值为-100的情况
            # ini_rewards_merge = [xxx for xxx in ini_rewards_merge if xxx != -100]

            # 查看更新后的初始奖励值
            # update_ini_reward_merge = update_ini_reward_2
            # print(f"最初筛选的初值之合并：{ini_rewards_merge}")
            ini_rewards_merge = np.partition(ini_rewards_merge, -10)[-10:]  # 获取前numpy初始奖励值列表 10 个值
            # print(f"取出的最大前10个数{ini_rewards_merge}")

            ini_Q_merge = np.partition(ini_Q_merge, -10)[-10:]  # 获取numpy,Q 列表中前 10 个值

            # 两次Qtable更新后  计算两次提取的初始值的  均值  以及 和
            ini_rewards_mean = mean(ini_rewards_merge)
            ini_rewards_sum = sum(ini_rewards_merge)

            #一个样品两次QTable更新后（调整初始值）计算合并后的Q值均值及求和
            Q_merge_mean = mean(ini_Q_merge)
            Q_merge_sum = sum(ini_Q_merge)

            #用于多个数据，逐个保存，目前在这个循环里，一个样品经历10次循环，每次循环经历两次Qtable连续更新
            # 所以保存的是一个样品的10结果  是关于 初始  reward  值  的
            ini_rewards_mean_sets.append(ini_rewards_mean)
            ini_rewards_sum_sets.append(ini_rewards_sum)

            #用于多个数据，逐个保存，目前在这个循环里，一个样品经历10次循环，每次循环经历两次Qtable连续更新
            # 所以保存的是一个样品的10结果  是关于 初始  Q  值  的
            Q_mean_sets.append(Q_merge_mean)
            Q_sum_sets.append(Q_merge_sum)


            # print(f'经过两次筛选最终确定的初始值之均值：{ini_rewards_mean}')
            # print(f'经过两次筛选最终确定的初始值之和：{ini_rewards_sum}')

            # print(f'经过两次筛选最终确定的Q值之和：{Q_merge_sum}')
            print(f'经过两次筛选最终确定的Q值之均值：{Q_merge_mean}')

        pd.DataFrame({'Rewardmean':ini_rewards_mean_sets,
                  'Rewardsum':ini_rewards_sum_sets,
                  'Qmean':Q_mean_sets,
                  'Qsum':Q_sum_sets}).to_csv(save_file + '/' + str(iii) + '-' + str(train_onesample_nums) +'次训练的结果.csv')
        # to_csv(save_file + '/' + str(iii) + '-' + str(
        #     end_action) + '-' + '训练后完全按照Qtable最大值没有随机的形成的路径.csv')

        sorted_Q_mean_sets = np.sort(Q_mean_sets)  #对其排序 从 小 到 大
        Q_mean_sets1 = sorted_Q_mean_sets[3:-3]  #去除第一个和最后一个
        Q_max_sets1 = max(sorted_Q_mean_sets)       #获取列表的最大值
        Q_min_sets1 = min(sorted_Q_mean_sets)

        TenMean_Q = mean(Q_mean_sets1)  # 对一个样本训练10次再取均值
        TenStd_Q = std(Q_mean_sets1)    # 对一个样本训练10次再取标准差

        TenMean_Q_sets.append(TenMean_Q)
        TenStd_Q_sets.append(TenStd_Q)
        TenMax_Q_sets.append(Q_max_sets1)
        TenMin_Q_sets.append(Q_min_sets1)
        '''_________________________________________________________________________________________-'''
        '''_________________________________________________________________________________________-'''
        pd.DataFrame({'TenMean': TenMean_Q_sets
                      ,'TenStd':TenStd_Q_sets
                      ,'TenMax':TenMax_Q_sets
                      ,'TenMin':TenMin_Q_sets}).to_csv(str(iii)+'Q_Pre_Result.csv')
        '''_________________________________________________________________________________________-'''
        '''_________________________________________________________________________________________-'''
        # ——————————————————————————————————————————————————————————————————————————————
        #  两线之间表示用神经网络初始奖励值变成一个值

        # only_one_for_one_sample = agent.forward(torch.Tensor(ini_rewards_merge)).item()
        # only_one_sets = []
        # only_one_sets.append(only_one_for_one_sample)
        # print(f"输出一个样本经过Q，然后经过线性神经网络后形成的一个数：{only_one_sets}")
        # ———————————————————————————————————————————————————————————————————————————————————


        # print(f"挑选出的初始奖励值：\n {ini_rewards_merge}")
        # print(f"挑选出初始值后将其替换为-100后，进而完成更新并显示更新后的初始奖励值：\n {update_ini_reward_merge}")
        # 可以通过这个方式调用最优路径所对应的Q值
        # q_values, = agent.play(env)
        # rewards = agent.play(env)
        # agent_location, rewards = agent.play(env)       return play_agent_loc, rewards

# 计算程序运行时间
end_time = time.time()
total_seconds = end_time - start_time
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)
print("Total Running Time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))