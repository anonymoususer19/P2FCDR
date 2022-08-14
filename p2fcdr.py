import argparse
from time import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import Dataset
from evaluate import evaluate_model
import numpy as np


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_model(model, epoch_id, dataset_name, hit_ratio, ndcg, flag):
    if flag == 'A':
        dataset = dataset_name.split("+")[0]
    else:
        dataset = dataset_name.split("+")[1]
    
    model_dir = 'checkpoints/Epoch{}_Dataset{}_HR{:.4f}_NDCG{:.4f}.model'.format(epoch_id, dataset, hit_ratio, ndcg)
    torch.save(model.state_dict(), model_dir)


def laplace_noisy(lambda_, size):
    noise = []
    for i in range(size):
        n_value = np.random.laplace(0, lambda_, 1)
        noise.append(n_value)
    return torch.Tensor(np.array(noise)).squeeze(-1).unsqueeze(0)
    

class CDR(nn.Module):
    def __init__(self, num_users, num_items, layers, layers1, dataset):
        super(CDR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = layers[0]
        self.layers = layers
        self.layers1 = layers1
        
        self.user_item_indices = torch.LongTensor([dataset.user_indices, dataset.item_incides])
        self.rating_data = torch.FloatTensor(dataset.rating_data)
        self.user_item_matrix = torch.sparse_coo_tensor(self.user_item_indices, self.rating_data, torch.Size((self.num_users, self.num_items))).to_dense().to(device)
        
        self.linear_user_1 = nn.Linear(in_features = self.num_items, out_features = self.latent_dim)
        self.linear_user_1.weight.detach().normal_(0, 0.01)
        self.linear_item_1 = nn.Linear(in_features = self.num_users, out_features = self.latent_dim)
        self.linear_item_1.weight.detach().normal_(0, 0.01)
        self.linear_select_1 = nn.Linear(in_features = self.latent_dim, out_features = self.latent_dim)
        self.linear_select_1.weight.detach().normal_(0, 0.01)
        
        self.logistic = torch.nn.Sigmoid()
        
        self.bridge = torch.nn.Parameter(torch.nn.init.orthogonal_(torch.empty(self.layers[1], self.layers[1])))
        
        self.user_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):
            self.user_fc_layers.append(nn.Linear(in_features = self.layers[idx - 1], out_features = self.layers[idx]))

        self.item_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):
            self.item_fc_layers.append(nn.Linear(in_features = self.layers[idx - 1], out_features = self.layers[idx]))
        
        self.feature_select_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers1)):
            self.feature_select_fc_layers.append(nn.Linear(in_features = self.layers1[idx - 1], out_features = self.layers1[idx]))
    
    def compute_user_item_embedding(self, user_indices, item_indices):
        self.user = self.user_item_matrix[user_indices]
        self.item = self.user_item_matrix[:, item_indices].t()

        self.user = self.linear_user_1(self.user)
        self.item = self.linear_item_1(self.item)

        for idx in range(len(self.layers) - 1):
            self.user = F.relu(self.user)
            self.user = self.user_fc_layers[idx](self.user)

        for idx in range(len(self.layers) - 1):
            self.item = F.relu(self.item)
            self.item = self.item_fc_layers[idx](self.item)
    
    def compute_score(self):
        self.diff = self.user - self.user_other
        self.diff = self.linear_select_1(self.diff)
        
        for idx in range(len(self.layers1) - 1):
            self.diff = F.relu(self.diff)
            self.diff = self.feature_select_fc_layers[idx](self.diff)
        
        self.diff = self.logistic(self.diff)
        self.reduce_diff = 1 - self.diff
        self.com_user = (self.diff * self.user) + (self.reduce_diff * self.user_other)
        # self.com_user = self.user + self.user_other
        
#         vector = self.logistic(torch.mul(self.com_user, self.item).sum(-1))

        
        vector = torch.cosine_similarity(self.com_user, self.item).view(-1, 1)
        vector = torch.clamp(vector, min = 1e-6, max = 1)
        
        return vector
        
    
    def return_bridge_weight(self):
        return self.bridge
    
    def set_bridge_weight(self, weight):
        self.bridge = weight
        
    def return_user_other_embedding(self, flag):
        if flag =='s':
            return torch.matmul(self.user.detach(), self.bridge.detach())
#             return self.user.detach()
        else:
            return torch.matmul(self.user.detach(), self.bridge.detach().t())
#             return self.user.detach()
    
    def set_other_embedding(self, user_other_embedding):
        self.user_other = user_other_embedding
    
    def save_pre_bridge(self):
        self.pre_bridge = self.bridge.detach().clone()
    
    def set_new_bridge_weight(self, weight):
        self.bridge.data = (self.pre_bridge + weight) / 2
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', nargs = '?', default = '/split_data/', help = 'Input data path.')
    parser.add_argument('--datasets', nargs = '?', default = 'movie_book_5_10_+book_movie_5_10_', help = 'Choose a dataset.')
    parser.add_argument('--epochs', type = int, default = 200, help = 'Number of epochs.')
    parser.add_argument('--batch_size', type = int, default = 512, help = 'Batch size.')
    parser.add_argument('--layers', nargs = '?', default = '[64, 64]', help = "Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--layers1', nargs = '?', default = '[64, 64]', help = "Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg', type = float, default = '0.0', help = "Regularization for each layer")
    parser.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate.')
    parser.add_argument('--learner', nargs = '?', default = 'adam', help = 'Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--pretrain', nargs = '?', default = False, help = 'Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    datasets = args.datasets
    epochs = args.epochs
    batch_size = args.batch_size
    layers = eval(args.layers)
    layers1 = eval(args.layers1)
    latent_dim = layers[0]
    reg = args.reg
    learning_rate = args.lr
    pretrain = args.pretrain
    topK = 10
    
    source_data_path = path + datasets.split("+")[0]
    target_data_path = path + datasets.split("+")[1]
    
    s_dataset = Dataset(source_data_path)
    t_dataset = Dataset(target_data_path)
    
    s_num_users, s_num_items = s_dataset.num_users, s_dataset.num_items
    t_num_users, t_num_items = t_dataset.num_users, t_dataset.num_items
    
    s_test_rating, s_test_negative = s_dataset.test_ratings, s_dataset.test_negative
    t_test_rating, t_test_negative = t_dataset.test_ratings, t_dataset.test_negative
    
    s_model = CDR(s_num_users, s_num_items, layers, layers1, s_dataset)
    s_model = s_model.to(device)
    t_model = CDR(t_num_users, t_num_items, layers, layers1, t_dataset)
    t_model = t_model.to(device)
    
    criterion = nn.BCELoss()
    

    s_optimizer = optim.Adam(s_model.parameters(), lr = learning_rate, weight_decay = reg)
    t_optimizer = optim.Adam(t_model.parameters(), lr = learning_rate, weight_decay = reg)
    print(s_model)
    
    print('header: dataset:{}, batch_size:{}, epochs:{}, latent_dim:{}, topK:{}, lr:{}, reg:{}'.format(datasets, batch_size, epochs, latent_dim, topK, learning_rate, reg))
    
    best_hrA, best_ndcgA = 0, 0
    best_hrB, best_ndcgB = 0, 0
    t_model.set_bridge_weight(s_model.return_bridge_weight())
    for epoch in range(epochs):
        s_model.train()
        t_model.train()
        epoch = epoch + 1
        
        t1 = time()
        
        permut = torch.randperm(s_num_users)
        for batch in range(0, s_num_users, batch_size):
            idx = permut[batch: batch + batch_size]
            s_user, s_item, s_y = s_dataset.get_train_instances(idx)
            t_user, t_item, t_y = t_dataset.get_train_instances(idx)
            
            s_user, s_item, s_y = s_user.cuda(), s_item.cuda(), s_y.cuda()
            t_user, t_item, t_y = t_user.cuda(), t_item.cuda(), t_y.cuda()
            
#             t_model.set_bridge_weight(s_model.return_bridge_weight())
                      
            s_model.compute_user_item_embedding(s_user, s_item)
            t_model.compute_user_item_embedding(t_user, t_item)
            
#             s_model.set_other_embedding(t_model.return_user_other_embedding(flag = 't') + laplace_noisy(0.1, 64).cuda())
#             t_model.set_other_embedding(s_model.return_user_other_embedding(flag = 's') + laplace_noisy(0.1, 64).cuda())
            
            s_model.set_other_embedding(t_model.return_user_other_embedding(flag = 't'))
            t_model.set_other_embedding(s_model.return_user_other_embedding(flag = 's'))
            
            s_model.save_pre_bridge()
            t_model.save_pre_bridge()
            
            s_y_hat = s_model.compute_score()
            t_y_hat = t_model.compute_score()
            
            s_loss = criterion(s_y_hat, s_y.view(-1, 1))
            s_loss_1 = (torch.matmul(s_model.return_bridge_weight(), s_model.return_bridge_weight().t()) - torch.eye(64).cuda()).sum(-1).sum(-1)
            # s_loss += s_loss_1
            
            t_loss = criterion(t_y_hat, t_y.view(-1, 1))
            t_loss_1 = (torch.matmul(t_model.return_bridge_weight(), t_model.return_bridge_weight().t()) - torch.eye(64).cuda()).sum(-1).sum(-1)
            # t_loss += t_loss_1
            
            s_optimizer.zero_grad()
            t_optimizer.zero_grad()
            s_loss.backward()
            t_loss.backward()
            s_optimizer.step()
            t_optimizer.step()
            
        s_model.set_new_bridge_weight(t_model.return_bridge_weight())
        t_model.set_new_bridge_weight(s_model.return_bridge_weight())
        
        t2 = time()
        
        s_model.eval()
        t_model.eval()
        if epoch % 1 == 0:
            (hit_ratioA, ndcgA, hit_ratioB, ndcgB) = evaluate_model(s_model, t_model, s_test_rating, s_test_negative, t_test_rating, t_test_negative, topK)
            print('epoch:{}, train_time:{:.1f}s, HRA:{:.4f}, NDCGA:{:.4f}, HRB:{:.4f}, NDCGB:{:.4f}, test_time:{:.1f}s'.format(epoch, t2 - t1, hit_ratioA, ndcgA, hit_ratioB, ndcgB, time() - t2))
            if hit_ratioA > best_hrA:
                count1 = 0
                best_hrA, best_ndcgA = hit_ratioA, ndcgA
                save_model(s_model, epoch, datasets, hit_ratioA, ndcgA, 'A')
            else:
                count1 += 1
            if hit_ratioB > best_hrB:
                count2 = 0
                best_hrB, best_ndcgB = hit_ratioB, ndcgB
                save_model(t_model, epoch, datasets, hit_ratioB, ndcgB, 'B')
            else:
                count2 += 1
            if count1 == 40 and count2 == 40:
                print(best_hrA, best_ndcgA, best_hrB, best_ndcgB)
                exit()
        print(best_hrA, best_ndcgA, best_hrB, best_ndcgB)
            
            
            
            
            
            
            
    
    
    

    
    