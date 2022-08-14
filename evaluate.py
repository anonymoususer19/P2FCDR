import math
import heapq
import multiprocessing
import numpy as np
import torch


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


_modelA = None
_testRatingsA = None
_testNegativesA = None
_modelB = None
_testRatingsB = None
_testNegativesB = None
_K = None


def evaluate_model(s_model, t_model, s_test_rating, s_test_negative, t_test_rating, t_test_negative, topK):
    global _modelA
    global _testRatingsA
    global _testNegativesA
    global _modelB
    global _testRatingsB
    global _testNegativesB
    global _K

    _modelA = s_model
    _modelB = t_model
    _testRatingsA = s_test_rating
    _testRatingsB = t_test_rating
    _testNegativesA = s_test_negative
    _testNegativesB = t_test_negative
    _K = topK
    
    hitsA, ndcgsA, hitsB, ndcgsB = [], [], [], []
    
    for idx in range(len(_testRatingsA)):
        (hrA, ndcgA, hrB, ndcgB) = eval_one_rating(idx)
        hitsA.append(hrA)
        ndcgsA.append(ndcgA)
        hitsB.append(hrB)
        ndcgsB.append(ndcgB)
    return (np.mean(hitsA), np.mean(ndcgsA), np.mean(hitsB), np.mean(ndcgsB))


def eval_one_rating(idx):
    ratingA = _testRatingsA[idx]
    ratingB = _testRatingsB[idx]
    
    itemsA = _testNegativesA[idx][0:999]
    itemsB = _testNegativesB[idx][0:999]
    
    uA = ratingA[0]
    uB = ratingB[0]
    
    gtItemA = ratingA[1]
    gtItemB = ratingB[1]
    
    itemsA.append(gtItemA)
    itemsB.append(gtItemB)

    map_item_scoreA = {}
    map_item_scoreB = {}
    
    usersA = np.full(len(itemsA), uA, dtype = 'int64')
    usersB = np.full(len(itemsB), uB, dtype = 'int64')
    
    batch_usersA, batch_itemsA = torch.LongTensor(usersA), torch.LongTensor(itemsA)
    tensor_usersA, tensor_itemsA = batch_usersA.to(device), batch_itemsA.to(device)
    
    batch_usersB, batch_itemsB = torch.LongTensor(usersB), torch.LongTensor(itemsB)
    tensor_usersB, tensor_itemsB = batch_usersB.to(device), batch_itemsB.to(device)
    
    _modelA.compute_user_item_embedding(tensor_usersA, tensor_itemsA)
    _modelB.compute_user_item_embedding(tensor_usersB, tensor_itemsB)
    
    _modelA.set_other_embedding(_modelB.return_user_other_embedding(flag = 't'))
    _modelB.set_other_embedding(_modelA.return_user_other_embedding(flag = 's'))
    
    y_predA = _modelA.compute_score()
    y_predB = _modelB.compute_score()

    y_predA = y_predA.cpu()
    y_predA = y_predA.detach().numpy()
    
    y_predB = y_predB.cpu()
    y_predB = y_predB.detach().numpy()

    for i in range(len(itemsA)):
        item = itemsA[i]
        map_item_scoreA[item] = y_predA[i]
    
    for i in range(len(itemsB)):
        item = itemsB[i]
        map_item_scoreB[item] = y_predB[i]

    ranklistA = heapq.nlargest(_K, map_item_scoreA, key = map_item_scoreA.get)
    ranklistB = heapq.nlargest(_K, map_item_scoreB, key = map_item_scoreB.get)
    
    hrA = getHitRatio(ranklistA, gtItemA)
    ndcgA = getNDCG(ranklistA, gtItemA)
    
    hrB = getHitRatio(ranklistB, gtItemB)
    ndcgB = getNDCG(ranklistB, gtItemB)
    
    return (hrA, ndcgA, hrB, ndcgB)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0
    