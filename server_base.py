import argparse
from collections import defaultdict
from typing import final

import torch
import pickle
from client.client import Client
import numpy as np
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.data_utils import split_fl_data,dirichlet_split_noniid
from dataset.meld import MELD
import server.NSGAII as NSGAII
from server.NSGAII import Individual,loss_rmse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
# from utils.data_utils import



class Server(object):
    def __init__(self, args):
        # 初始化服务器参数
        self.data_path = None
        self.args = args
        self.num_clients = args.num_clients
        self.alpha = args.alpha
        self.top_k = 1

        # 设置随机种子
        np.random.seed(42)
        # 初始化并分配数据集
        self.total_data = self.initialize_dataset()
        self.clients = self.set_clients()
        self.syn_select_modal = []
        self.run()



    
    def processpool_client_synthetic_data(self, client):
        modal_list = client.run()
        return modal_list

    def processpool_client_eval_data(self, client):
        client_acc = client.eval_data(self.syn_select_modal)
        return client_acc
    def run(self):
        # 训练并聚合模型
        # 初始化并分配数据
        # self.clients[0].run()
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=40) as executor:
            result = list(executor.map(self.processpool_client_synthetic_data, self.clients))
        end_time1 = time.time()
        client_time = end_time1 - start_time
        print(f"Client time: {client_time} seconds") 

        transformed_modal_list = {0: [], 1: []}
        for i in range(len(result)):
            transformed_modal_list[0].append(result[i][0])
            transformed_modal_list[1].append(result[i][1])
        for i in range(len(transformed_modal_list)):

            solution,top_k_solution = self.NSGAII(transformed_modal_list[i])
            solution_transposed = np.array(solution).transpose(1, 0)
            self.syn_select_modal.append(solution_transposed)

        
        np.save('××××××××××××/seed/syn_select_modal_data.npy',self.syn_select_modal)
        end_time2 = time.time()
        server_time = end_time2 - end_time1
        print(f"Server time: {server_time} seconds") 
        
        with ProcessPoolExecutor(max_workers=40) as executor:
            final_result = list(executor.map(self.processpool_client_eval_data, self.clients))
        
        np.save('××××××××××××/seed/final_result.npy',self.syn_select_modal)
        acc_first = np.zeros((self.num_clients, 2, 3))
        acc_second = np.zeros((self.num_clients, 2, 3))
        acc_first_label = np.zeros((self.num_clients,2,3,5))
        acc_second_label = np.zeros((self.num_clients,2,3,5))
        for client  in range(len(final_result)):
            for modal in range(len(final_result[client])): 
                for solution in range(len(final_result[client][modal])):
                    acc_first[client][modal][solution] = final_result[client][modal][solution]["first_acc"]
                    acc_second[client][modal][solution] = final_result[client][modal][solution]["second_acc"]
                    acc_first_label[client][modal][solution] = final_result[client][modal][solution]["first_acc_label"]
                    acc_second_label[client][modal][solution] = final_result[client][modal][solution]["second_acc_label"]
        
        mean_first_acc = np.mean(acc_first, axis=0)  # 对clients维度进行平均
        mean_second_acc = np.mean(acc_second, axis=0)
        mean_first_label_acc = np.mean(acc_first_label, axis=0)
        mean_second_label_acc = np.mean(acc_second_label, axis=0)

        # 打印结果
        for modal in range(len(mean_first_acc)):
            print(f"Modal {modal}:")
            for solution in range(len(mean_first_acc[modal])):
                print(f"  Solution {solution}:")
                print(f"  First acc: {mean_first_acc[modal][solution]}")
                print(f"  Second acc: {mean_second_acc[modal][solution]}")
                print(f"  First label acc: {mean_first_label_acc[modal][solution]}")
                print(f"  Second label acc: {mean_second_label_acc[modal][solution]}")
        

        
            
            



    def get_client_data(self,path):
        with open(path, 'r') as file:
            audio_data = json.load(file)  # 加载 JSON 文件内容
        # print(data[10])
        label = []
        for audio_path in audio_data:
            category = audio_path.split('/')[-2]  # 获取倒数第二个元素作为类别
            # print(category)
            label.append(int(category))
        client_data = split_fl_data(audio_data, label, alpha=self.alpha, num_clients=self.num_clients )

        return client_data


    def set_clients(self):
        client_list = []
        if self.args.dataset == 'meld':
            for i in range(self.num_clients):
                data  = {
                    'total_data': self.total_data[i]['data'],
                    'total_label': self.total_data[i]['labels'],
                }
                device = 'cuda:{}'.format(self.args.devices[i % len(self.args.devices)])
                client=Client(self.args, data, i, device)
                client_list.append(client)
        elif self.args.dataset == 'seed':
            for i in range(self.num_clients):
                device = 'cuda:{}'.format(self.args.devices[i % len(self.args.devices)])
                client = Client(self.args, self.total_data[i], i, device)
                client_list.append(client)

        return client_list
    
    def initialize_dataset(self,):
        # 初始化数据集
        total_data = []
        combined_index = None
        combined_label = None
        if self.args.dataset is not None:
            if self.args.dataset == 'meld':
                self.args.num_classes = 7
                self.data_path = '××××××××××××/audio_data/audio_list.json'  # 音频数据路径
                train_index = MELD(split='train').getdata()
                train_label = [train_index[i][3] for i in range(len(train_index)-1)]
                test_index = MELD(split='test').getdata()
                test_label = [test_index[i][3] for i in range(len(test_index)-1)]
                combined_index = train_index + test_index
                combined_label = train_label + test_label
                total_data = split_fl_data(combined_index, combined_label, alpha=self.alpha,
                                           num_clients=self.num_clients)
            elif self.args.dataset == "seed":
                self.args.num_classes = 5
                self.data_path = "××××××××××××/seed/seed_label_data.npy"
                seed_label  = np.load(self.data_path)
                total_data = dirichlet_split_noniid(seed_label,alpha=self.alpha,num_clients=self.num_clients)
        else:
            assert False, 'Please specify the dataset'
        return total_data
    
    def NSGAII(self,result):

        loaded_results = result
        #按照类别进行选择
        solution = []
        top_k_solution = []
        for label in range(self.args.num_classes):
            #统计上传label的客户端
            client_index = []
            client_data = {}
            for i in range(self.num_clients):
                 if label in loaded_results[i][1].keys():
                     client_index.append(loaded_results[i][0])
                     client_data[loaded_results[i][0]] = loaded_results[i][1][label]

            #对客户端数据进行NSGAII操作
            #初始化种群
            popnum = 10  # 种群大小
            solution_dim = len(client_index)
            if solution_dim == 0:
                raise ValueError("No client has data for label {}".format(label))
            solution_way = "no_binary"
            bound_min = 0.2
            bound_max = 1
            eta = 1  # 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
            objective_fun = loss_rmse
            P = []
            for i in range(popnum):
                P.append(Individual(client_index,client_data,label))
                if solution_way == "binary":
                    P[i].solution = np.random.randint(0, 2, size=solution_dim)
                else:
                    step_size = 0.1
                    values = np.arange(bound_min, 1 + step_size, step_size)  # [0, 0.1, 0.2, ..., 1]
                    P[i].solution = np.random.choice(values, size=solution_dim)
                P[i].calculate_objective(objective_fun)

            NSGAII.fast_non_dominated_sort(P)
            Q = NSGAII.make_new_pop(P, eta, bound_min, bound_max, objective_fun,client_index,client_data,label,solution_way)

            P_t = P
            Q_t = Q
            for gen_cur in range(100):
                R_t = P_t + Q_t
                F = NSGAII.fast_non_dominated_sort(R_t)
                P_n = []  # 即为P_t+1,表示下一届的父代
                i = 1
                while len(P_n) + len(F[i]) < popnum:  # until the parent population is filled
                    NSGAII.crowding_distance_assignment(F[i])  # calculate crowding-distance in F_i
                    P_n = P_n + F[i]  # 将第 i 层的个体加入下一代父代种群。
                    i = i + 1  # 切换到下一层
                F[i].sort(key=lambda x: x.distance)  # 数量将要达到popnum时,对下一层的个体按照拥挤度距离排序
                P_n = P_n + F[i][:popnum - len(P_n)]  # 将排序后的个体加入下一代父代种群，以保证种群大小不超过 popnum
                Q_n = NSGAII.make_new_pop(P_n, eta, bound_min, bound_max,
                                   objective_fun,client_index,client_data,label,solution_way)
                P_t = P_n
                Q_t = Q_n
                plt.clf()
                plt.title('current generation_P_t:' + str(gen_cur + 1))  # 绘制当前循环选出来的父代的图
                self.plot_P(P_t)
                plt.pause(0.1)

            top_3_solutions = sorted(P_t, key=lambda ind: ind.distance, reverse=True)[:3]
            solution.insert(label, NSGAII.feature_from_solution(top_3_solutions))
            top_k_solution.insert(label, top_3_solutions)
        return solution,top_k_solution

    def transform_solution(self,solution):
        fusion_syn_select = []

        for k in range(self.top_k):
            # 创建一个新的内层列表来存放每个类（i）对应的值
            inner_list = []
            for i in range(self.args.num_classes):
                inner_list.append(solution[i][k])  # 将数据按位置填入列表
            fusion_syn_select.append(inner_list)  # 将内层列表添加到外层列表中

        return fusion_syn_select

    def plot_P(P):
        """
        假设目标就俩,给个种群绘图
        :param P:
        :return:
        """
        X = []
        Y = []
        for ind in P:
            X.append(ind.objective[1])
            Y.append(ind.objective[2])

        plt.xlabel('F1')
        plt.ylabel('F2')
        plt.scatter(X, Y)






