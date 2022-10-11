from shutil import copyfile

import math
import torch

from torch.autograd import Variable
import logging
import sklearn.metrics.pairwise as smp
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import time
from scipy.linalg import eigh
from scipy.special import rel_entr

logger = logging.getLogger("logger")
import os
import json
import numpy as np
import config
import copy
import utils.csv_record
import tqdm
import random

MAX_ITER = 10
ITV = 1000

class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'saved_models/model_{self.name}_{current_time}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
        self.fg= FoolsGold(use_memory=self.params['fg_use_memory'])
        self.history_prev_average_grad = None
        self.history_tau = 10.

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_max_values(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
        return squared_sum

    @staticmethod
    def model_max_values_var(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer - target_params[name])))
        return sum(squared_sum)

    @staticmethod
    def get_one_vec(model, variable=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if variable:
            sum_var = Variable(torch.cuda.FloatTensor(size).fill_(0))
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var= sum_var.to(config.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model, variable=True)
        target_var = Variable(target_vec, requires_grad=False)
        cs_sim = torch.nn.functional.cosine_similarity(
            self.params['scale_weights'] * (model_vec - target_var) + target_var, target_var, dim=0)
        logger.info("los")
        logger.info(cs_sim.data[0])
        logger.info(torch.norm(model_vec - target_var).data[0])
        loss = 1 - cs_sim

        return 1e3 * loss

    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight':
                continue

            model_update = 100 * (data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[
                name].view(-1)

            cs = F.cosine_similarity(model_update,
                                     target_params_variables[name].view(-1), dim=0)
            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))
        logger.info(model_id)
        logger.info((sum(cs_list) / len(cs_list)).data[0])
        return 1e3 * sum(cos_los_submit)

    def accum_similarity(self, last_acc, new_acc):

        cs_list = list()

        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, layer in last_acc.items():
            cs = cs_loss(Variable(last_acc[name], requires_grad=False).view(-1),
                         Variable(new_acc[name], requires_grad=False).view(-1))
            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))

        return sum(cos_los_submit)

    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def accumulate_weight(self, weight_accumulator, epochs_submit_update_dict, state_keys,num_samples_dict):
        """
         return Args:
             updates: dict of (num_samples, update), where num_samples is the
                 number of training samples corresponding to the update, and update
                 is a list of variable weights
         """
        if self.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_gradients = epochs_submit_update_dict[state_keys[i]][0] # agg 1 interval
                num_samples = num_samples_dict[state_keys[i]]
                updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients))
            return None, updates

        else:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                update= dict()
                num_samples=num_samples_dict[state_keys[i]]

                for name, data in local_model_update_list[0].items():
                    update[name] = torch.zeros_like(data)

                for j in range(0, len(local_model_update_list)):
                    local_model_update_dict= local_model_update_list[j]
                    for name, data in local_model_update_dict.items():
                        weight_accumulator[name].add_(local_model_update_dict[name])
                        update[name].add_(local_model_update_dict[name])
                        detached_data= data.cpu().detach().numpy()
                        detached_data=detached_data.tolist()
                        local_model_update_dict[name]=detached_data # from gpu to cpu

                updates[state_keys[i]]=(num_samples,update)

            return weight_accumulator,updates

    def init_weight_accumulator(self, target_model):
        weight_accumulator = dict()
        for name, data in target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        return weight_accumulator

    def average_shrink_models(self, weight_accumulator, target_model, epoch_interval):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["no_models"])
            # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])

            # update_per_layer = update_per_layer * 1.0 / epoch_interval
            if self.params['diff_privacy']:
                update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        return True

    def fed_avg(self, target_model, updates):
        is_updated = False
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')
        
        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])

        size = len(samples)
        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            metric = []
            for idx in range(size):
                metric.append(samples[idx][layer_name].unsqueeze(0).type(torch.float))
            metric = torch.cat(metric, dim=0)
            chosen[layer_name] = torch.mean(metric, dim=0)

        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True
        return is_updated

    def foolsgold_update(self,target_model,updates):
        client_grads = []
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)
        # print(updates)
        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(f'[foolsgold agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[foolsgold agg] considering poison per batch poison_fraction: {poison_fraction}')

        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        agg_grads, wv,alpha = self.fg.aggregate_gradients(client_grads,names)
        for i, (name, params) in enumerate(target_model.named_parameters()):
            agg_grads[i]=agg_grads[i] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        optimizer.step()
        wv=wv.tolist()
        utils.csv_record.add_weight_result(names, wv, alpha)
        return True, names, wv, alpha

    def geometric_median_update(self, target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm= None):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm"""
        points = []
        alphas = []
        names = []
        for name, data in updates.items():
            points.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)

        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        alphas = torch.from_numpy(alphas).float()

        median = Helper.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging
        obj_val = Helper.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            logger.info('Starting Weiszfeld algorithm')
            logger.info(log_entry)
        logger.info(f'[rfa agg] init. name: {names}, weight: {alphas}')
        # start
        wv=None
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, Helper.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = Helper.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = Helper.geometric_median_objective(median, points, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         Helper.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                logger.info(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
            logger.info(f'[rfa agg] iter:  {i}, prev_obj_val: {prev_obj_val}, obj_val: {obj_val}, abs dis: { abs(prev_obj_val - obj_val)}')
            logger.info(f'[rfa agg] iter:  {i}, weight: {weights}')
            wv=copy.deepcopy(weights)
        alphas = [Helper.l2dist(median, p) for p in points]

        update_norm = 0
        for name, data in median.items():
            update_norm += torch.sum(torch.pow(data, 2))
        update_norm= math.sqrt(update_norm)

        if max_update_norm is None or update_norm < max_update_norm:
            for name, data in target_model.state_dict().items():
                update_per_layer = median[name] * (self.params["eta"])
                if self.params['diff_privacy']:
                    update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
                data.add_(update_per_layer)
            is_updated = True
        else:
            logger.info('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            is_updated = False

        utils.csv_record.add_weight_result(names, wv.cpu().numpy().tolist(), alphas)

        return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(),alphas

    def ex_noregret_(self, samples, eps=1./12, sigma=1, expansion=20, dis_threshold=0.7):
        """
        samples: data samples in numpy array
        sigma: operator norm of covariance matrix assumption
        """
        samples = samples.cpu().numpy()
        size = len(samples)
        f = int(np.ceil(eps*size))
        metric = self.krum_(list(samples), f)
        indices = np.argpartition(metric, -f)[:-f]
        samples = samples[indices]
        size = samples.shape[0]
        
        dis_list = []
        for i in range(size):
            for j in range(i+1, size):
                dis_list.append(np.linalg.norm(samples[i]-samples[j]))
        step_size = 0.5 / (np.amax(dis_list) ** 2)
        size = samples.shape[0]
        feature_size = samples.shape[1]
        samples_ = samples.reshape(size, 1, feature_size)

        c = np.ones(size)
        for i in range(int(2 * eps * size)):
            avg = np.average(samples, axis=0, weights=c)
            cov = np.average(np.array([np.matmul((sample - avg).T, (sample - avg)) for sample in samples_]), axis=0, weights=c)
            eig_val, eig_vec = eigh(cov, eigvals=(feature_size-1, feature_size-1), eigvals_only=False)
            eig_val = eig_val[0]
            eig_vec = eig_vec.T[0]

            if eig_val * eig_val <= expansion * sigma * sigma:
                return torch.from_numpy(avg).to(config.device)

            tau = np.array([np.inner(sample-avg, eig_vec)**2 for sample in samples])
            c = c * (1 - step_size * tau)
            ordered_c_index = np.flip(np.argsort(c))
            min_KL = None
            projected_c = None
            for i in range(len(c)):
                c_ = np.copy(c)
                for j in range(i+1):   
                    c_[ordered_c_index[j]] = 1./(1-eps)/len(c)
                clip_norm = 1 - np.sum(c_[ordered_c_index[:i+1]])
                norm = np.sum(c_[ordered_c_index[i+1:]])
                if clip_norm <= 0:
                    break
                scale = clip_norm / norm
                for j in range(i+1, len(c)):
                    c_[ordered_c_index[j]] = c_[ordered_c_index[j]] * scale
                if c_[ordered_c_index[i+1]] > 1./(1-eps)/len(c):
                    continue
                KL = np.sum(rel_entr(c, c_))
                if min_KL is None or KL < min_KL:
                    min_KL = KL
                    projected_c = c_

            c = projected_c
            
        avg = np.average(samples, axis=0, weights=c)
        return torch.from_numpy(avg).to(config.device)

    def ex_noregret(self, target_model, updates, eps=1./12, sigma=1, expansion=20, itv=ITV):
        """
        samples: data samples in numpy array
        sigma: operator norm of covariance matrix assumption
        """
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])
        
        is_updated = False

        size = len(samples)
        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            feature_shape = samples[0][layer_name].shape
            samples_flatten = []
            for i in range(size):
                samples_flatten.append(samples[i][layer_name].flatten().unsqueeze(0).type(torch.float))
            feature_size = samples_flatten[0].shape[-1]
            if itv == None:
                itv = int(np.sqrt(feature_size))
            cnt = int(feature_size // itv)
            sizes = []
            for i in range(cnt):
                sizes.append(itv)
            if feature_size % itv:
                sizes.append(feature_size - cnt * itv)

            samples_flatten =  torch.cat(samples_flatten, dim=0)
            idx = 0
            res = []
            for size_ in sizes:
                temp_res = self.ex_noregret_(samples_flatten[:, idx:idx+size_], eps=eps, sigma=sigma, expansion=expansion)
                res.append(temp_res)
                idx += size_
            res = torch.cat(res)
            res = res.reshape(feature_shape)
            chosen[layer_name] = res

        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True
        return is_updated

    def median(self, target_model, updates):
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])
        
        is_updated = False

        size = len(samples)
        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            feature_shape = samples[0][layer_name].shape
            samples_flatten = []
            for i in range(size):
                samples_flatten.append(samples[i][layer_name].flatten().unsqueeze(0).type(torch.float))

            samples_flatten =  torch.cat(samples_flatten, dim=0)
            res = torch.median(samples_flatten, dim=0).values
            res = res.reshape(feature_shape)
            chosen[layer_name] = res

        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True
        return is_updated

    def filterl2_(self, samples, eps=0.2, sigma=1, expansion=20):
        size = samples.shape[0]
        feature_size = samples.shape[1]
        samples_ = samples.reshape(size, 1, feature_size)

        c = torch.ones(size).unsqueeze(0).to(config.device)
        count = 0
        for i in range(2 * int(eps * size)):
            avg = torch.mm(c, samples) / (torch.sum(c))
            cov = [((sample - avg).T.mul(sample - avg)).unsqueeze(0) for sample in samples_]
            cov = torch.cat(cov, dim=0)
            cov_size = cov.shape[-1]
            cov = cov.reshape(cov.shape[0], -1)
            cov = torch.mm(c, cov) / (torch.sum(c))
            cov = cov.reshape(cov_size, cov_size)
            eig_val, eig_vec = eigh(cov.cpu().numpy(), eigvals=(feature_size-1, feature_size-1), eigvals_only=False)
            eig_val = eig_val[0]
            eig_vec = torch.from_numpy(eig_vec.T[0]).to(config.device)
            avg = avg.squeeze(0)
            if eig_val * eig_val <= expansion * sigma * sigma:
                # print(c)
                return avg
            tau = torch.cat([(torch.dot(sample - avg, eig_vec)**2).unsqueeze(0) for sample in samples])
            tau_max_idx = torch.argmax(tau)
            tau_max = tau[tau_max_idx]
            c = c * (1 - tau/tau_max)
            samples = torch.cat((samples[:tau_max_idx], samples[tau_max_idx+1:]))
            samples_ = samples.reshape(-1, 1, feature_size)
            c = torch.cat((c[:, :tau_max_idx], c[:, tau_max_idx+1:]), dim=1)
            c = c / torch.linalg.norm(c, ord=1)
        avg = torch.mm(c, samples) / (torch.sum(c))
        avg = avg.squeeze(0)
        return avg

    def filterl2(self, target_model, updates, sigma=1, expansion=1, itv=None):
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])

        is_updated = False
        size = len(samples)
        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            feature_shape = samples[0][layer_name].shape
            samples_flatten = []
            for i in range(size):
                samples_flatten.append(samples[i][layer_name].flatten().unsqueeze(0).type(torch.float))
            feature_size = samples_flatten[0].shape[-1]
            
            itv = ITV
            cnt = int(feature_size // itv)
            sizes = []
            for i in range(cnt):
                sizes.append(itv)
            if feature_size % itv:
                sizes.append(feature_size - cnt * itv)
            samples_flatten =  torch.cat(samples_flatten, dim=0)
            idx = 0
            res = []
            for size_ in tqdm.tqdm(sizes):
                temp_res = self.filterl2_(samples_flatten[:, idx:idx+size_], sigma=sigma, expansion=expansion)
                res.append(temp_res)
                idx += size_
            res = torch.cat(res)
            res = res.reshape(feature_shape)
            chosen[layer_name] = res

        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True
        return is_updated

    def krum_(self, samples, f):
        size = len(samples)
        size_ = size - f - 2
        metric = []
        for idx in range(size):
            sample = samples[idx]
            samples_ = samples.copy()
            del samples_[idx]
            dis = np.array([np.linalg.norm(sample-sample_) for sample_ in samples_])
            metric.append(np.sum(dis[np.argsort(dis)[:size_]]))
        return metric

    def krum(self, target_model, updates, f=0):
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)

        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])

        is_updated = False
        size = len(samples)
        size_ = size - f - 2
        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            metric = []
            for idx in range(size):
                sample = samples[idx][layer_name].type(torch.float)
                samples_ = []
                for iidx in range(size):
                    samples_.append(copy.deepcopy(samples[iidx][layer_name]))
                del samples_[idx]
                dis = torch.tensor([torch.norm(sample - sample_.type(torch.float)) for sample_ in samples_])
                metric.append(torch.sum(dis[torch.argsort(dis)[:size_]]))
            index = np.argmin(metric)
            chosen[layer_name] = samples[index][layer_name]
        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True
        return is_updated

    def history(self, target_model, updates):
        is_updated = False
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')
        
        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])

        if self.history_prev_average_grad is None:
            self.history_prev_average_grad = copy.deepcopy(samples[0])
            for layer_name in self.history_prev_average_grad.keys():
                self.history_prev_average_grad[layer_name].zero_()
        

        size = len(samples)

        for c in range(size):
            norm = 0.
            for layer_name in self.history_prev_average_grad.keys():
                norm += torch.norm(samples[c][layer_name] - self.history_prev_average_grad[layer_name])**2
                norm = torch.sqrt(norm)
            for layer_name in self.history_prev_average_grad.keys():
                samples[c][layer_name] = (samples[c][layer_name] - self.history_prev_average_grad[layer_name]) * min(1, self.history_tau / norm)

        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            metric = []
            for idx in range(size):
                metric.append(samples[idx][layer_name].unsqueeze(0).type(torch.float))
            metric = torch.cat(metric, dim=0)
            chosen[layer_name] = torch.mean(metric, dim=0)

        self.history_prev_average_grad = copy.deepcopy(chosen)

        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True
        return is_updated

    def bucketing(self, target_model, updates):
        is_updated = False
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')
        
        samples = self.sharding(samples, self.params['shard_size'])

        if self.history_prev_average_grad is None:
            self.history_prev_average_grad = copy.deepcopy(samples[0])
            for layer_name in self.history_prev_average_grad.keys():
                self.history_prev_average_grad[layer_name].zero_()
        
        size = len(samples)

        for c in range(size):
            norm = 0.
            for layer_name in self.history_prev_average_grad.keys():
                norm += torch.norm(samples[c][layer_name] - self.history_prev_average_grad[layer_name])**2
                norm = torch.sqrt(norm)
            for layer_name in self.history_prev_average_grad.keys():
                samples[c][layer_name] = (samples[c][layer_name] - self.history_prev_average_grad[layer_name]) * min(1, self.history_tau / norm)

        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            metric = []
            for idx in range(size):
                metric.append(samples[idx][layer_name].unsqueeze(0).type(torch.float))
            metric = torch.cat(metric, dim=0)
            chosen[layer_name] = torch.mean(metric, dim=0)
        self.history_prev_average_grad = copy.deepcopy(chosen)

        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True
        return is_updated

    def mom_krum(self, target_model, updates, f=0, bucket_size=3):
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)

        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])

        bucket_num = int(np.ceil(len(samples) * 1. / bucket_size))

        chosen = copy.deepcopy(samples[0])

        bucketed_samples = [copy.deepcopy(samples[0])] * bucket_num
        for layer_name in chosen.keys():
            for i in range(bucket_num):
                bucketed_samples[i][layer_name].zero_()
                for j in range(i*bucket_size, min((i+1)*bucket_size, len(samples))):
                    bucketed_samples[i][layer_name] += samples[j][layer_name]
                bucketed_samples[i][layer_name] /= (min((i+1)*bucket_size, len(samples)) - i*bucket_size + 1)
        
        samples = bucketed_samples

        is_updated = False
        size = len(samples)
        size_ = size - f - 2
        for layer_name in chosen.keys():
            metric = []
            for idx in range(size):
                sample = samples[idx][layer_name].type(torch.float)
                samples_ = []
                for iidx in range(size):
                    samples_.append(copy.deepcopy(samples[iidx][layer_name]))
                del samples_[idx]
                dis = torch.tensor([torch.norm(sample - sample_.type(torch.float)) for sample_ in samples_])
                metric.append(torch.sum(dis[torch.argsort(dis)[:size_]]))
            index = np.argmin(metric)
            chosen[layer_name] = samples[index][layer_name]
        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True
        return is_updated

    def trimmed_mean(self, target_model, updates, beta=0.1):
        is_updated = False
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')
        
        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])

        size = len(samples)
        beyond_choose = int(size * beta)
        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            metric = []
            for idx in range(size):
                metric.append(samples[idx][layer_name].unsqueeze(0).type(torch.float))
            metric = torch.cat(metric, dim=0)
            metric, _ = torch.sort(metric, 0)
            chosen[layer_name] = torch.mean(metric[beyond_choose:size-beyond_choose], dim=0)

        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True
        return is_updated

    def bulyan_one_coordinate(self, arr, beta):
        distance = arr.view(1, -1) - arr.view(-1, 1)
        distance = torch.abs(distance)
        total_dis = torch.sum(distance, axis=-1)
        median_index = torch.argmin(total_dis)
        neighbours = arr[torch.argsort(distance[median_index])[:beta]]
        return torch.mean(neighbours)

    def bulyan_krum(self, target_model, updates, f=20):
        is_updated = False
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])

        size = len(samples)
        theta = size - 2 * f
        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            feature_shape = chosen[layer_name].shape
            samples_flatten = []
            for i in range(size):
                samples_flatten.append(samples[i][layer_name].flatten())
            selected_grads = []
            for i in range(theta):
                metric = []
                size_ = size - i - f - 2
                for j in range(size - i):
                    sample = samples_flatten[j].type(torch.float)
                    dis = torch.tensor([torch.norm(sample - sample_.type(torch.float)) for sample_ in samples_flatten])
                    metric.append(torch.sum(dis[torch.argsort(dis)[:size_]]))
                index = np.argmin(metric)
                selected_grads.append(samples_flatten[index].unsqueeze(0).type(torch.float))
                del samples_flatten[index]
            beta = theta - 2 * f
            grads_dim = selected_grads[0].shape[-1]
            selected_grads = torch.cat(selected_grads, dim=0)
            selected_grads_by_cod = torch.zeros(grads_dim, 1).to(config.device)
            for i in range(grads_dim):
                selected_grads_by_cod[i, 0] = self.bulyan_one_coordinate(selected_grads[:, i], beta)
            selected_grads_by_cod = selected_grads_by_cod.reshape(feature_shape)
            chosen[layer_name] = selected_grads_by_cod
        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True

        return is_updated

    def bulyan_median(self, target_model, updates, f=20):
        is_updated = False
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])

        size = len(samples)
        theta = size - 2 * f
        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            feature_shape = chosen[layer_name].shape
            samples_flatten = []
            for i in range(size):
                samples_flatten.append(samples[i][layer_name].flatten().type(torch.float))
            selected_grads = []
            for i in range(theta):
                metric = []
                for idx in range(len(samples_flatten)):
                    metric.append(samples_flatten[idx].unsqueeze(0))
                metric = torch.cat(metric, dim=0)
                metric = torch.median(metric, dim=0).values
                selected_grads.append(metric.unsqueeze(0).type(torch.float))
                min_dis = np.inf
                min_index = None
                for j in range(len(samples_flatten)):
                    temp_dis = torch.norm(selected_grads[-1] - samples_flatten[j])
                    if temp_dis < min_dis:
                        min_dis = temp_dis
                        min_index = j
                assert min_index != None
                del samples_flatten[min_index]
            beta = theta - 2 * f
            grads_dim = selected_grads[0].shape[-1]
            selected_grads = torch.cat(selected_grads, dim=0)
            selected_grads_by_cod = torch.zeros(grads_dim, 1).to(config.device)
            for i in range(grads_dim):
                selected_grads_by_cod[i, 0] = self.bulyan_one_coordinate(selected_grads[:, i], beta)
            selected_grads_by_cod = selected_grads_by_cod.reshape(feature_shape)
            chosen[layer_name] = selected_grads_by_cod
        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True

        return is_updated


    def bulyan_trimmed_mean(self, target_model, updates, f=20):
        # FIXME: add bulyan here
        is_updated = False
        samples = []
        alphas = []
        names = []
        for name, data in updates.items():
            samples.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)
        
        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.params['adversary_list']:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        if self.params['sharding']:
            samples = self.sharding(samples, self.params['shard_size'])

        size = len(samples)
        theta = size - 2 * f
        chosen = copy.deepcopy(samples[0])
        for layer_name in chosen.keys():
            feature_shape = chosen[layer_name].shape
            samples_flatten = []
            for i in range(size):
                samples_flatten.append(samples[i][layer_name].flatten().type(torch.float))
            selected_grads = []
            
            for i in range(theta):
                beyond_choose = int((size - i) * 0.1)
                metric = []
                for idx in range(len(samples_flatten)):
                    metric.append(samples_flatten[idx].unsqueeze(0))
                metric = torch.cat(metric, dim=0)
                if beyond_choose > 0:
                    metric, _ = torch.sort(metric, 0)
                    selected_grads.append(torch.mean(metric[beyond_choose:size-i-beyond_choose], dim=0).unsqueeze(0))
                else:
                    selected_grads.append(torch.mean(metric, dim=0).unsqueeze(0))
                min_dis = np.inf
                min_index = None
                for j in range(len(samples_flatten)):
                    temp_dis = torch.norm(selected_grads[-1] - samples_flatten[j])
                    if temp_dis < min_dis:
                        min_dis = temp_dis
                        min_index = j
                assert min_index != None
                del samples_flatten[min_index]

            beta = theta - 2 * f
            grads_dim = selected_grads[0].shape[-1]
            selected_grads = torch.cat(selected_grads, dim=0)
            selected_grads_by_cod = torch.zeros(grads_dim, 1).to(config.device)
            for i in range(grads_dim):
                selected_grads_by_cod[i, 0] = self.bulyan_one_coordinate(selected_grads[:, i], beta)
            selected_grads_by_cod = selected_grads_by_cod.reshape(feature_shape)
            chosen[layer_name] = selected_grads_by_cod
        for name, data in target_model.state_dict().items():
            update_per_layer = chosen[name] * (self.params["eta"])
            if update_per_layer.type() != data.type():
                update_per_layer = update_per_layer.type_as(data)
            data.add_(update_per_layer)
        is_updated = True

        return is_updated

    def sharding(self, samples, eps=0.2, delta=np.exp(-5)):
        def discrete(samples):
            temp_min = -1.5
            temp_max = 1.5
            splitnum = 1000000
            for i in range(len(samples)):
                temp = samples[i]
                for layer_name in temp.keys():
                    if torch.min(temp[layer_name]) < temp_min:
                        temp_min = torch.min(temp[layer_name])
                    if torch.max(temp[layer_name]) > temp_max:
                        temp_max = torch.max(temp[layer_name])
                    temp[layer_name] = torch.clamp(temp[layer_name], temp_min, temp_max)
                    temp[layer_name] = ((temp[layer_name] - temp_min) * splitnum / (temp_max - temp_min)).type(torch.int)
                    temp[layer_name] = temp[layer_name] * (temp_max - temp_min) / splitnum + temp_min
                samples[i] = copy.deepcopy(temp)
            print('discrete: ', len(samples), temp_min, temp_max)
            return samples

        bucket_num = 50
        bucket_size = int(np.ceil(len(samples) * 1. / bucket_num))
        shard_grads = []
        random.shuffle(samples)
        for i in range(bucket_num):
            begin_index = i*bucket_size
            end_index = min((i+1)*bucket_size, len(samples))
            shard_average = copy.deepcopy(samples[begin_index])
            if end_index - begin_index > 1:
                for layer_name in shard_average.keys():
                    shard_average[layer_name] = shard_average[layer_name].type(torch.float)
                    for j in range(begin_index+1, end_index):
                        shard_average[layer_name] += samples[j][layer_name].type(torch.float)
                    shard_average[layer_name] /= (end_index - begin_index)
            shard_grads.append(shard_average)
        return shard_grads

    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        squared_sum = 0
        for name, data in p1.items():
            squared_sum += torch.sum(torch.pow(p1[name]- p2[name], 2))
        return math.sqrt(squared_sum)


    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """Compute geometric median objective."""
        temp_sum= 0
        for alpha, p in zip(alphas, points):
            temp_sum += alpha * Helper.l2dist(median, p)
        return temp_sum

        # return sum([alpha * Helper.l2dist(median, p) for alpha, p in zip(alphas, points)])

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = torch.sum(weights)

        weighted_updates= dict()

        for name, data in points[0].items():
            weighted_updates[name]=  torch.zeros_like(data)
        for w, p in zip(weights, points): # 对每一个agent
            for name, data in weighted_updates.items():
                temp = (w / tot_weights).float().to(config.device)
                temp= temp* (p[name].float())
                # temp = w / tot_weights * p[name]
                if temp.dtype!=data.dtype:
                    temp = temp.type_as(data)
                data.add_(temp)

        return weighted_updates

    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params['save_on_epochs']:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def update_epoch_submit_dict(self,epochs_submit_update_dict,global_epochs_submit_dict, epoch,state_keys):

        epoch_len= len(epochs_submit_update_dict[state_keys[0]])
        for j in range(0, epoch_len):
            per_epoch_dict = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                local_model_update_dict = local_model_update_list[j]
                per_epoch_dict[state_keys[i]]= local_model_update_dict

            global_epochs_submit_dict[epoch+j]= per_epoch_dict

        return global_epochs_submit_dict


    def save_epoch_submit_dict(self, global_epochs_submit_dict):
        with open(f'{self.folder_path}/epoch_submit_update.json', 'w') as outfile:
            json.dump(global_epochs_submit_dict, outfile, ensure_ascii=False, indent=1)

    def estimate_fisher(self, model, criterion,
                        data_loader, sample_size, batch_size=64):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        if self.params['type'] == 'text':
            data_iterator = range(0, data_loader.size(0) - 1, self.params['bptt'])
            hidden = model.init_hidden(self.params['batch_size'])
        else:
            data_iterator = data_loader

        for batch_id, batch in enumerate(data_iterator):
            data, targets = self.get_batch(data_loader, batch,
                                           evaluation=False)
            if self.params['type'] == 'text':
                hidden = self.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, self.n_tokens), targets)
            else:
                output = model(data)
                loss = log_softmax(output, dim=1)[range(targets.shape[0]), targets.data]

            loglikelihoods.append(loss)

        logger.info(loglikelihoods[0].shape)
        # estimate the fisher information of the parameters.
        loglikelihood = torch.cat(loglikelihoods).mean(0)
        logger.info(loglikelihood.shape)
        loglikelihood_grads = torch.autograd.grad(loglikelihood, model.parameters())

        parameter_names = [
            n.replace('.', '__') for n, p in model.named_parameters()
        ]
        return {n: g ** 2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, model, fisher):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            model.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            model.register_buffer('{}_estimated_fisher'
                                  .format(n), fisher[n].data.clone())

    def ewc_loss(self, model, lamda, cuda=False):
        try:
            losses = []
            for n, p in model.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(model, '{}_estimated_mean'.format(n))
                fisher = getattr(model, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

class FoolsGold(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict=dict()
        self.wv_history = []
        self.use_memory = use_memory

    def aggregate_gradients(self, client_grads,names):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]]+=grads[i]
            else:
                self.memory_dict[names[i]]=copy.deepcopy(grads[i])
            self.memory[i]=self.memory_dict[names[i]]

        if self.use_memory:
            wv, alpha = self.foolsgold(self.memory)  # Use FG
        else:
            wv, alpha = self.foolsgold(grads)  # Use FG
        logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads, wv, alpha

    def foolsgold(self,grads):
        """
        :param grads:
        :return: compute similatiry and return weightings
        """
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv,alpha(base)
