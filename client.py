import argparse
import os
import pickle
import random
from collections import defaultdict

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torchnet
from torch import optim
from torch.utils.data import dataloader, DataLoader, TensorDataset
from scipy.stats import beta
from client import client_utils
import warnings
from model.meld_model.meld_model import AttentionFusion, MultiModalClassifier
from torch.utils.data import WeightedRandomSampler
from model.seed_model import seed_feature_model,seed_classifier_model
# import client_utils.
# 屏蔽 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
import wandb
from torch.optim import lr_scheduler
from tqdm import tqdm
import copy
class Client(object):
    def __init__(self,args,data,client_id,device):
        # 初始化客户端参数
        self.args=args
        self.data = data
        self.client_id=client_id
        self.device = device
        self.loss = nn.CrossEntropyLoss() # 声明属性
        self.contrastive = client_utils.contrastive_loss
        self.train_dataloader, self.test_dataloader, self.train_dataset, self.test_dataset = None, None, None, None
        self.model, self.model_oral = None, None

        # # 根据类别划分数据
        # self.train_data = []
        # self.train_label = []
        # self.test_data = []
        # self.test_label = []
        #
        # unique_labels = set(self.total_label)
        # for label in unique_labels:
        #     label_indices = [i for i, l in enumerate(self.total_label) if l == label]
        #     label_indices = np.random.permutation(label_indices).tolist()
        #     split_idx = int(0.8 * len(label_indices))
        #     train_indices = label_indices[:split_idx]
        #     test_indices = label_indices[split_idx:]
        #
        #     self.train_data.extend([self.total_data[i] for i in train_indices])
        #     self.train_label.extend([self.total_label[i] for i in train_indices])
        #     self.test_data.extend([self.total_data[i] for i in test_indices])
        #     self.test_label.extend([self.total_label[i] for i in test_indices])
        # #声明变量
        self.acc_label_mean_first, self.acc_label_mean_second = None, None
        self.acc_mean_first, self.acc_mean_second = None, None


    # def contrastive_loss(self, fusion_features, audio_features, image_features, margin=1.0):
    #     distance_audio = F.pairwise_distance(fusion_features, audio_features)
    #     distance_image = F.pairwise_distance(fusion_features, image_features)
    #     label = torch.ones(audio_features.size(0)).to(fusion_features.device)
    #
    #     loss_audio = label * torch.pow(distance_audio, 2) + (1 - label) * torch.pow(torch.clamp(margin - distance_audio, min=0.0), 2)
    #     loss_image = label * torch.pow(distance_image, 2) + (1 - label) * torch.pow(torch.clamp(margin - distance_image, min=0.0), 2)
    #     # 总损失为两者之和
    #     total_loss = loss_audio.mean() + loss_image.mean()
    #     return total_loss



    # def contrastive_loss(self,fusion_feature, image_features, audio_features,temperature=0.07):   #融合后的特征也是（b,c,h,w） (b,32,14,14)
    #     b, c, h, w = image_features.shape
    #     image_features = image_features.view(b, -1) # 平均池化，形状 (b, c)
    #     audio_features = audio_features.view(b, -1) # 平均池化，形状 (b, c)
    #     fusion_feature = fusion_feature.view(b, -1) # 平均池化，形状 (b, c)
    #
    #     labels = torch.arange(b, device=image_features.device)
    #     # 对特征进行 L2 归一化
    #     cosine_sim_image = torch.matmul(image_features, fusion_feature.T) / temperature  # (b, b)
    #     cosine_sim_audio = torch.matmul(audio_features, fusion_feature.T)  / temperature # (b, b)
    #     loss1 = F.cross_entropy(cosine_sim_image, labels)
    #     loss2= F.cross_entropy(cosine_sim_audio, labels)
    #
    #     return loss1 + loss2

    def train(self,model,train_dataloader,test_dataloader,args):
        align_train_optimizer = optim.Adam(
            list(model[0]["feature"].parameters()) + list(model[1]["feature"].parameters()),
            lr=args.learning_rate*0.01)  # 用于对齐训练的优化器
        optimizer_0 = optim.Adam(
            list(model[0]["feature"].parameters()) + list(model[0]["classifier"].parameters()),
            lr=args.learning_rate)  # 用于第一个模型的优化器
        optimizer_1 = optim.Adam(
            list(model[1]["feature"].parameters()) + list(model[1]["classifier"].parameters()),
            lr=args.learning_rate)  # 用于第二个模型的优化器
        beat_accuracy  = [0]*len(model)
        optimizer = [optimizer_0, optimizer_1]
        for epoch in range(args.client_train_epochs):
            for i, each in enumerate(train_dataloader):
                if each[0].shape[0] <= 1:
                    continue
                align_train_optimizer.zero_grad()
                model[0]["feature"].to(self.device).train()
                model[1]["feature"].to(self.device).train()
                feature_0 = model[0]["feature"](each[0].to(self.device))
                feature_1 = model[1]["feature"](each[1].to(self.device))

                loss = self.contrastive(feature_0,feature_1)
                loss.backward()
                align_train_optimizer.step()
            for j in range(len(model)):
                running_loss = 0.0
                correct = 0
                total = 0
                for _, each in enumerate(train_dataloader):
                    if each[0].shape[0] <= 1:
                        continue
                    optimizer[j].zero_grad()  # 清空梯度
                    model[j]["feature"].to(self.device).train()
                    model[j]["classifier"].to(self.device).train()
                    data = model[j]["feature"](each[j].to(self.device))
                    labels  = each[len(each)-1].to(self.device).long()
                    outputs = model[j]["classifier"](data)
                    loss = self.loss(outputs, labels)
                    loss.backward()
                    optimizer[j].step()  # 更新参数
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)  # outputs(bath_size,num_classes),torch.max 函数用于返回输入张量中指定维度的最大值及其索引
                    total += labels.size(0)  # 总样本数
                    correct += (predicted == labels).sum().item()  # 计算正确预测的数量
                # epoch_loss = running_loss / total if total > 0 else 0
                # epoch_accuracy = 100 * correct / total if total > 0 else 0  # 防止除以零
                # print(
                #     f'Epoch [{epoch + 1}/{args.client_train_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

            if (epoch+1) % 10 == 0:
                epoch_accuracy = self.test(model,test_dataloader,args)
            if (epoch+1)  == args.client_train_epochs:
                for j in range(len(model)):
                    self.save(model[j],args,j)




    def test(self, model,test_dataloader,args):
        epoch_accuracy = []
        with (torch.no_grad()):  # 禁用梯度计算
            for j in range(len(model)):
                running_loss = 0.0
                correct = 0
                total = 0
                for _, each in enumerate(test_dataloader):
                    if each[0].shape[0] <= 1:
                        continue
                    model[j]["feature"].to(self.device).eval()
                    model[j]["classifier"].to(self.device).eval()
                    data = model[j]["feature"](each[j].to(self.device))
                    labels = each[len(each) - 1].to(self.device).long()
                    outputs = model[j]["classifier"](data)
                    loss = self.loss(outputs, labels)
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)  # outputs(bath_size,num_classes),torch.max 函数用于返回输入张量中指定维度的最大值及其索引
                    total += labels.size(0)  # 总样本数
                    correct += (predicted == labels).sum().item()  # 计算正确预测的数量
                epoch_accuracy.append(100 * correct / total)
        return epoch_accuracy

            # if isinstance(model, Sound) or isinstance(model, SimpleCNN)or isinstance(model, MultiModalClassifier):
            #     loss_total = running_loss / len(test_dataloader)
            #     acc_total= correct / total if total > 0 else 0  # 防止除以零
            #     # print(f'Test Results - Loss: {loss_total:.4f}, Accuracy: {acc_total:.4f}')
            #     if not self.args.debug:
            #         wandb.log({
            #             model.__class__.__name__+"_test_loss": loss_total,
            #             model.__class__.__name__+"_test_accuracy": acc_total,
            #             "epoch": epoch
            #         })
            # else:
            #     loss_total = running_loss / len(test_dataloader)
            #     # print(f'Test Results - Loss: {loss_total:.4f}')
            #     if not self.args.debug:
            #         wandb.log({
            #             model.__class__.__name__+"_test_loss": loss_total,
            #             "epoch": epoch
            #         })


    def get_feature_data(self,model,dataloader,temp):
        model.eval()
        with torch.no_grad():
            for i, each in enumerate(dataloader):
                data = each[temp].to(self.device)
                labels  = each[len(each)-1].to(self.device).long()
                model = model.to(self.device)
                outputs = model(data)
                if i == 0:
                    all_features = outputs
                    all_labels = labels
                else:
                    all_features = torch.cat([all_features, outputs], dim=0)
                    all_labels = torch.cat([all_labels, labels], dim=0)
            dataset = TensorDataset(all_features, all_labels)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

        return all_features,all_labels,loader


    def save(self, model, args,j, save_path="××××××××××××"):

        path = os.path.join(save_path , args.dataset,f"client_id_{self.client_id}")
        if not os.path.exists(path):
            os.makedirs(path)
        if args.dataset == 'seed':
                if j == 0:
                    name = "eeg"
                else:   
                    name = "eye"    
                model_file_path = f"{path}/{model['feature'].__class__.__name__}_{name}_best.pt"
                torch.save(model['feature'].state_dict(), model_file_path)
                model_file_path = f"{path}/{model['classifier'].__class__.__name__}_{name}_best.pt"
                torch.save(model['classifier'].state_dict(), model_file_path)

    def load_model(self, model, epoch, server_epoch,save_path="××××××××××××"):
        model_file_path = f"{save_path}/server_epoch_{server_epoch}/client_id_{self.client_id}/{model.__class__.__name__}best.pt"

        # 检查文件是否存在
        if not os.path.exists(model_file_path):
            print(f"Error:  {model_file_path}  not have")
        model.load_state_dict(torch.load(model_file_path))
        return model


    def get_network(self,model):
        # 使用 ConvNet_GBN 进行合成数据训练
        if model == 'ConvNet_GBN':
            net = ConvNet_GBN(channel=32, num_classes=10, net_width=64, net_depth=2, net_act='relu', net_norm='layernorm', net_pooling='maxpooling', im_size=(14,14))
        else:
            exit(f"Unknown model for synthetic training: {model}")
        return net
    def eval_data(self, syn_select):

        self.model,self.model_oral = client_utils.initialize_model(self.args,self.client_id, pretrained = True)
        self.train_dataloader , self.test_dataloader, self.train_dataset, self.test_dataset = client_utils.initialize_dataset(self.data,self.args)
        modal_record = defaultdict(dict)
        for modal in range(len(syn_select)):
            train_all_features,train_all_labels,train_loader = self.get_feature_data(self.model[modal]["feature"],self.train_dataloader,modal)
            train_all_features, train_all_labels  = train_all_features.cpu().numpy(),train_all_labels.cpu().numpy()
            _,_,test_loader = self.get_feature_data(self.model[modal]["feature"],self.test_dataloader,modal)
            syn_features_select = None
            syn_label_select = None
            topk_acc_record = []
            for i in range(len(syn_select[modal])):
                for label in range(len(syn_select[modal][i])):
                    if syn_features_select is None:
                        syn_features_select = syn_select[modal][i][label]
                        syn_label_select = np.full(syn_features_select.shape[0], label)
                    else:
                        syn_features_select = np.concatenate([syn_features_select, syn_select[modal][i][label]],axis=0)
                        syn_label_select = np.concatenate([syn_label_select, np.full( syn_select[modal][i][label].shape[0], label)],axis=0)
                
                syn_features_second = torch.tensor(np.concatenate([train_all_features, syn_features_select], axis=0))
                syn_label_second = torch.tensor(np.concatenate([train_all_labels, syn_label_select],axis=0))
                syn_features_first = torch.tensor(train_all_features)
                syn_label_first = torch.tensor(train_all_labels)

                acc_list = []
                acc_label_list = []
                loss_list = []
                for it_eval in range(self.args.num_eval):
                    model_eval = self.model_oral(self.args.num_classes)
                    acc, class_acc, loss_array = client_utils.evaluate_synset(model_eval, syn_features_first, syn_label_first,
                                                                            test_loader,self.device, self.args)
                    acc_list.append(acc)
                    acc_label_list.append(class_acc)
                    loss_list.append(loss_array)

                self.acc_mean_first = np.mean(np.stack(acc_list), axis=0)
                self.acc_label_mean_first = np.mean(np.array(acc_label_list), axis=0)

                for it_eval in range(self.args.num_eval):
                    model_eval = self.model_oral(self.args.num_classes)
                    acc, class_acc, loss_array = client_utils.evaluate_synset(model_eval, syn_features_second, syn_label_second,
                                                                            test_loader,self.device, self.args)
                    acc_list.append(acc)
                    acc_label_list.append(class_acc)
                    loss_list.append(loss_array)

                self.acc_mean_second = np.mean(np.stack(acc_list), axis=0)
                self.acc_label_mean_second = np.mean(np.array(acc_label_list), axis=0)
                record = {
                            "first_acc": self.acc_mean_first,
                            "second_acc": self.acc_mean_second,
                            "first_acc_label": self.acc_label_mean_first,
                            "second_acc_label": self.acc_label_mean_second
                        }
                topk_acc_record.append(record)

            modal_record[modal] = topk_acc_record

        return modal_record








    def synthetic_data(self,model,model_oral,train_dataloader,test_dataloader,temp):
        # 构建参数
        train_all_features,train_all_labels,train_loader = self.get_feature_data(model["feature"],train_dataloader,temp)
        test_all_features,test_all_labels,test_loader = self.get_feature_data(model["feature"],test_dataloader,temp)

        eval_it_pool = np.arange(0, self.args.Iteration + 1, self.args.eval_interval).tolist()[1:] # 评估间隔内执行评估和数据，进出模型队列，训练模型队列
        lr_schedule = np.arange(0, self.args.Iteration + 1, 100).tolist()[1:]  #在训练中途调整学习率


        best_acc = 0
        for exp in range(self.args.num_exp):  #最外层循环（表示实验次数）

            #初始化合成数据
            noise_std = 1.4
            train_syn = train_all_features.clone().detach().to(self.device)
            noise = torch.randn_like(train_syn) * noise_std
            train_syn = train_syn + noise
            label_syn = train_all_labels.to(device=self.device)

            #构建优化器
            if self.args.optim == 'sgd':
                optimizer_syn = torch.optim.SGD([train_syn, ], lr=self.args.lr_syn,momentum=0.5)  # optimizer_img for synthetic data
            elif self.args.optim == 'adam':
                optimizer_syn = torch.optim.Adam([train_syn, ], lr=self.args.lr_syn)
                train_syn = train_syn.requires_grad_(True).to(self.device)
            else:
                raise NotImplemented()
            optimizer_syn.zero_grad()

            # 初始化模型队列
            net_num = self.args.net_num
            net_list = list()
            optimizer_list = list()
            epoch_list = list()
            for net_index in range(1):
                net = model_oral(5)  # get a random model
                net = net.to(self.device)
                net.train()
                if self.args.net_decay:
                    optimizer_net = torch.optim.SGD(net.parameters(), lr=self.args.lr_net, momentum=0.9,
                                                    weight_decay=0.0005)
                else:
                    optimizer_net = torch.optim.SGD(net.parameters(),
                                                    lr=self.args.lr_net)  # optimizer_img for synthetic data
                    for i in range(self.args.epoch_eval_train):
                        _, _,=client_utils.epoch('train', train_loader, net, optimizer_net, self.device,self.args)
                # acc_avg, class_accuracy = client_utils.epoch('test', test_loader, net, optimizer_net, self.device,self.args)
                optimizer_net.zero_grad()
                net_list.append(net)
                optimizer_list.append(optimizer_net)
                epoch_list.append(18-net_index)

            #开始迭代
            for it in range(self.args.Iteration):
                it=it+1
                if it in eval_it_pool:
                    
                    #评估合成数据
                    acc_list = []
                    acc_label_list = []
                    loss_list = []
                    for it_eval in range(self.args.num_eval):
                        model_eval = model_oral(5)
                        acc ,class_acc,loss_array = client_utils.evaluate_synset(model_eval, train_syn, label_syn, test_loader, self.device, self.args)
                        acc_list.append(acc)
                        acc_label_list.append(class_acc)
                        loss_list.append(loss_array)

                    mean_loss = np.mean(np.stack(loss_list), axis=0)
                    acc_mean = np.mean(np.stack(acc_list), axis=0)
                    acc_label_mean = np.mean(np.array(acc_label_list), axis=0)
                    # print(acc_mean)
                    # print(acc_label_mean)


                    # 计算loss
                    if it == (self.args.Iteration):
                        train_syn = train_syn.cpu().detach().numpy()
                        label_syn = label_syn.cpu().detach().numpy()
                        train_all_features = train_all_features.cpu().detach().numpy()
                        self.acc_label_mean_first = acc_label_mean
                        self.acc_mean_first  = acc_mean
                        indices = np.where(acc_label_mean >= np.mean(acc_label_mean))
                        syn_data = {}
                        for c in indices[0]:
                            indice = np.where(label_syn == c)
                            features = train_syn[indice[0]]
                            losses = mean_loss[indice[0]].squeeze(axis=1)
                            sorted_indices = np.argsort(losses)
                            num_selected = min(len(sorted_indices), 15)
                            top_50_indices = sorted_indices[:num_selected]
                            syn_data[c] = {"feature": features[top_50_indices],
                                            "loss": losses[top_50_indices]}
                            
                        np.save(f"××××××××××××/seed/client_id_{self.client_id}/modal_{temp}_up_data.npy",syn_data)
                        np.save(f"××××××××××××/seed/client_id_{self.client_id}/modal_{temp}_syn_features.npy",train_syn)
                        np.save(f"××××××××××××/seed/client_id_{self.client_id}/modal_{temp}_label.npy",label_syn)
                        np.save(f"××××××××××××/seed/client_id_{self.client_id}/modal_{temp}_real_features.npy",train_all_features)
                        
                        return self.client_id,syn_data,train_syn,label_syn


                        # if (np.isnan(self.client_id) or np.isnan(syn_data).any() or np.isnan(train_syn).any() or np.isnan(label_syn).any()):
                            
                        #     # 如果任何一个值是 NaN，返回客户端 ID
                        #     print(f"Client ID with NaN values: {self.client_id}")
                        # else:
                        #     # 否则，返回其他正常数据（可以根据你的需要返回相应的值）
                        #     return self.client_id,syn_data,train_syn,label_syn

                        # # return self.client_id,syn_data,train_syn,label_syn

                    # if best_acc < np.mean(accs):

                        # print('Best acc:', np.mean(accs))

                        # checkpoint_dir = '××××/Clients_model/client_id{}_model_{}_iteration{}/'.format(self.client_id, self.client_id,it)
                        # if not os.path.exists(checkpoint_dir):
                        #     os.mkdir(checkpoint_dir)
                        # best_synset_filename = checkpoint_dir + 'acc_{}.pkl'.format(np.mean(accs))
                        # with open(best_synset_filename, 'wb') as pkl_file:
                        #     pickle.dump((fusion_syn.detach(), label_syn.detach()), pkl_file)


                    last_net = len(net_list)
                    #模型队列进出队列
                    for _ in range(self.args.net_push_num):
                        if len(net_list) == net_num:
                            net_list.pop(0)
                            optimizer_list.pop(0)
                            acc_meters.pop(0)
                        net = MultiModalClassifier(self.args.mmultimmdal_dim,self.args.num_classes) # get a random model
                        net.train()
                        if self.args.net_decay:
                            optimizer_net = torch.optim.SGD(net.parameters(), lr=self.args.lr_net, momentum=0.9, weight_decay=0.0005)
                        else:
                            optimizer_net = torch.optim.SGD(net.parameters(), lr=self.args.lr_net)  # optimizer_img for synthetic data
                        optimizer_net.zero_grad()
                        net_list.append(net)
                        optimizer_list.append(optimizer_net)
                        acc_meters.append(torchnet.meter.ClassErrorMeter(accuracy=True))


                    # 训练队列模型
                    for j in range(len(net_list)):
                        if j <= last_net - 1 : #当队列已满，新加入的模型与队列中原有的模型执行相同的训练方式
                            net_train = net_list[j]
                            optimizer_net_train = optimizer_list[j]
                            for i in range(self.args.model_train_steps):
                                _, _ =client_utils.epoch('train', self.fusion_dataloader, net_train, optimizer_net_train, self.device)
                            epoch_list[j] += self.args.model_train_steps
                        else: #当队列还没有满的时候，新加入的模型执行下面的训练方式，接力训练
                            net_train = net_list[j]
                            optimizer_net_train = optimizer_list[j]
                            for i in range(epoch_list[j-1]):
                                _, _ =client_utils.epoch('train', self.fusion_dataloader, net_train, optimizer_net_train, self.device)
                            epoch_list.append(epoch_list[j-1]-1)

                # 生成Beta分布,根据Beta分布从模型队列中抽样
                # 调整α和β的值
                x = np.linspace(0, 1, len(net_list))
                alpha = 1 + client_utils.sigmoid(it  / self.args.Iteration) * 10
                beta_param = 1 + (1 - client_utils.sigmoid(it  / self.args.Iteration)) * 10
                weights = beta.pdf(x, alpha, beta_param)  # 生成Beta分布

                peak_value = 10  # 设置分布的峰值
                weights = weights / np.max(weights) * peak_value  # 峰值归一化到目标值
                weights = weights / np.sum(weights)  # 权重归一化为1

                samples_model = np.random.choice(net_list, size=1, replace=False, p=weights)
                samples_model = net_list


                #训练合成数据
                if it in lr_schedule:
                    optimizer_syn = torch.optim.Adam([train_syn, ], lr=self.args.lr_syn*00.1)

                for _ in range(self.args.outer_loop):
                    loss_avg = 0
                    metrics = {'syn': 0, 'real': 0}
                    acc_avg = {'syn': torchnet.meter.ClassErrorMeter(accuracy=True)}

                    ''' update synthetic data '''

                    for sign, syn_data in [['syn', train_syn]]:
                        loss = torch.tensor(0.0).to(self.device)
                        for net_ind in range(len(samples_model)):
                            net = samples_model[net_ind]
                            net.eval()
                            # embed = samples_model[net_ind]
                            # net_acc = train_acc_list[net_ind]
                            loss_c = torch.tensor(0.0).to(self.device)

                            #按照类别计算数据的损失
                            for c in range(self.args.num_classes):
                                c_lab =[b  for b in range(label_syn.shape[0]) if label_syn[b] == c]
                                c_fusion_syn = [syn_data[c_lab[b]]  for b in range(len(c_lab))]
                                if len(c_fusion_syn) == 0:
                                    continue
                                c_fusion_syn=torch.stack(c_fusion_syn, dim=0).to(self.device)

                                c_fusion_real=[train_all_features[c_lab[b]] for b in range(len(c_lab))]
                                c_fusion_real=torch.stack(c_fusion_real, dim=0).to(self.device)

                                output_real, layer_outputs_real = net(c_fusion_real,return_intermediate=True)
                                output_syn,layer_outputs_syn = net(c_fusion_syn,return_intermediate=True)
                                #分布损失
                                loss_s_bn1 = torch.mean((layer_outputs_real[0] - layer_outputs_syn[0]) ** 2)
                                loss_s = torch.mean((output_real - output_syn) ** 2) + loss_s_bn1
                                #二范数损失
                                loss_y = torch.mean((c_fusion_real - c_fusion_syn) ** 2)
                                #分类损失
                                c_tensor = torch.tensor([c] * len(c_lab)).to(self.device)
                                loss_l=F.cross_entropy(output_syn,c_tensor)
                                #总损失
                                loss_c = 0.9 *loss_s + 0.1 *loss_l + 0.9 *loss_y
                                loss_c=loss_c/len(c_lab)
                                optimizer_syn.zero_grad()
                                loss_c.backward()
                                optimizer_syn.step()

    def run(self):
        # 完整的数据集加载类

        self.train_dataloader , self.test_dataloader, self.train_dataset, self.test_dataset = client_utils.initialize_dataset(self.data,self.args)

        self.model,self.model_oral = client_utils.initialize_model(self.args,self.client_id, pretrained = False)
        self.train(self.model, self.train_dataloader,self.test_dataloader,self.args)
        modal = {}
        for i in range(len(self.model)):
            temp = i
            client_id,syn_data,syn_features,syn_label = self.synthetic_data(self.model[i], self.model_oral, self.train_dataloader, self.test_dataloader,temp)
            modal[i] = [client_id, syn_data, syn_features, syn_label]

        return modal



























