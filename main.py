import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import numpy as np
from Net import SeG
from sklearn import metrics
from data_loader import data_loader
from config import config
from utils import AverageMeter
from losses import LDAMLoss,SCELoss

def adjust_learning_rate(optimizer, epoch,opt):
    """Sets the learning rate"""
    if epoch < opt['stage2'] :# 200
        lr = opt['lr']# 0.02
                    #（320- 200）/3 + 200 = 240
    elif epoch < (opt['epoch'] - opt['stage2'])/3 + opt['stage2']:
        lr = opt['lr2']  #  0.2
                    # 2*（320-200）/3 + 200 = 280
    elif epoch < 2 * (opt['epoch'] - opt['stage2'])/3 + opt['stage2']:
        lr = opt['lr2']/10  # 0.02
    else:
        lr = opt['lr2']/100 # 0.002

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, test_loader, opt,model ,criterion, optimizer):

    not_best_count = 0
    best_auc = 0
    ckpt = os.path.join(opt['save_dir'], 'model.pth.tar')
    # label标签的个数
    cls_num_list = list(train_loader.dataset.class_numdict.values())
    num_classes = len(cls_num_list)
    # bag的num
    bagnum = train_loader.dataset.__len__()
    for epoch in range(opt['epoch']):
        # 调整学习率 lr
        adjust_learning_rate(optimizer, epoch,opt)
        # load y_tilde，开始都没有数据，存上一次循环里的得到的预测的new label 作为下一次里面的y～
        if os.path.isfile(y_file):
            y = np.load(y_file,allow_pickle=True)
        else:
            y = []

        if epoch >= opt['stage2']:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1

        model.train()
        print("\n=== Epoch %d train ===" % epoch)
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        avg_pos_acc = AverageMeter()

        # new y is y_tilde after updating,存放更新后的y～
        new_y = np.zeros([bagnum, num_classes])
        for i, data in enumerate(train_loader):

            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            word, pos1, pos2, ent1, ent2, mask, length, scope, rel ,index= data

            if torch.cuda.is_available():
                word = word.cuda()
                pos1 = pos1.cuda()
                pos2 = pos2.cuda()
                ent1 = ent1.cuda()
                ent2 = ent2.cuda()
                mask = mask.cuda()
                length = length.cuda()
                scope = scope.cuda()
                rel = rel.cuda()

            output = model(word, pos1, pos2, ent1, ent2, mask, scope)
            _, pred = torch.max(output, -1)
            logsoftmax = nn.LogSoftmax(dim=1).to(device)
            softmax = nn.Softmax(dim=1).to(device)
            if epoch < opt['stage1']:  # < 70
                # lc is classification loss
                lc = criterion(output, rel)
                # init y_tilde, let softmax(y_tilde) is noisy labels
                onehot = torch.zeros(rel.size(0), num_classes, device=device).scatter_(1, rel.view(-1, 1), 10.0)# 10.0 it type 8 in paper
                onehot = onehot.to(torch.device("cpu")).numpy()
                new_y[index, :] = onehot
            else:
                yy = y
                yy = yy[index, :]
                yy = torch.tensor(yy, requires_grad=True, device=device)
                # obtain label distributions (y_hat)
                last_y_var = softmax(yy).type(torch.cuda.FloatTensor)
                # print(last_y_var)
                # print(type(last_y_var))
                # print(rel)
                lc = torch.mean(softmax(output) * (logsoftmax(output) - torch.log((last_y_var))))
                # lo is compatibility loss

                lo = criterion(last_y_var, rel)
            # le is entropy loss
            le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))

            if epoch < opt['stage1']:  # < 70
                loss = lc
            elif epoch < opt['stage2']:  # < 200
                loss = lc + opt['alpha'] * lo + opt['beta'] * le
            else:
                loss = lc



            acc = (pred == rel).sum().item() / rel.shape[0]
            pos_total = (rel != 0).sum().item()
            pos_correct = ((pred == rel) & (rel != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | loss: %f, acc: %f, pos_acc: %f'%(i+1, avg_loss.avg, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            # Optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if epoch >= opt['stage1'] and epoch < opt['stage2']:
                lambda1 = opt['lambda1']
                # update y_tilde by back-propagation
                yy.data.sub_(lambda1 * yy.grad.data)
                new_y[index, :] = yy.data.cpu().numpy()

        # save y_tilde
        if epoch < opt['stage2']:
            y = new_y
            np.save(y_file, y)

        if (epoch + 1) % opt['val_iter'] == 0 :
            # and avg_pos_acc.avg > 0.5
            print("\n=== Epoch %d val ===" % epoch)
            y_true, y_pred = valid(test_loader, model,criterion)
            auc = metrics.average_precision_score(y_true, y_pred)
            print("\n[TEST] auc: {}".format(auc))
            if auc > best_auc:
                print("Best result!")
                best_auc = auc
                torch.save({'state_dict': model.state_dict()}, ckpt)
                not_best_count = 0
            else:
                not_best_count += 1
            if not_best_count >= opt['early_stop']:
                break


def valid(test_loader, model,criterion):
    model.eval()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    avg_pos_acc = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # if torch.cuda.is_available():
            #     data = [x.cuda() for x in data]

            word, pos1, pos2, ent1, ent2, mask, length, scope, rel,index = data
            if torch.cuda.is_available():
                word = word.cuda()
                pos1 = pos1.cuda()
                pos2 = pos2.cuda()
                ent1 = ent1.cuda()
                ent2 = ent2.cuda()
                mask = mask.cuda()
                length = length.cuda()
                scope = scope.cuda()
                rel = rel.cuda()

            output = model(word, pos1, pos2, ent1, ent2, mask, scope)
            label = rel.argmax(-1)
            # print(output)
            # print(rel)
            # print(rel.argmax(-1))
            loss = criterion(output, label)
            output = torch.softmax(output, -1)

            _, pred = torch.max(output, -1)
            acc = (pred == label).sum().item() / label.shape[0]
            pos_total = (label != 0).sum().item()
            pos_correct = ((pred == label) & (label != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_loss.update(loss.item(), 1)
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | loss: %f, acc: %f, pos_acc: %f'%(i+1, avg_loss.avg, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            y_true.append(rel[:, 1:])
            y_pred.append(output[:, 1:])
    y_true = torch.cat(y_true).reshape(-1).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).reshape(-1).detach().cpu().numpy()
    return y_true, y_pred


def test(test_loader,model,save_dir):
    print("\n=== Test ===")
    # Load model
    # save_dir = opt['save_dir']
    # model = SeG(test_loader.dataset.vec_save_dir, test_loader.dataset.rel_num(),
    #             lambda_pcnn=opt['lambda_pcnn'], lambda_san=opt['lambda_san'])
    # if torch.cuda.is_available():
    #     model = model.cuda()
    # state_dict = torch.load(os.path.join(save_dir, 'model.pth.tar'))['state_dict']
    # own_state = model.state_dict()
    # for name, param in state_dict.items():
    #     if name not in own_state:
    #         continue
    #     own_state[name].copy_(param)

    y_true, y_pred = valid(test_loader, model,criterion)
    # AUC
    auc = metrics.average_precision_score(y_true, y_pred)
    print("\n[TEST] auc: {}".format(auc))
    # P@N
    order = np.argsort(-y_pred)
    p100 = (y_true[order[:100]]).mean() * 100
    p200 = (y_true[order[:200]]).mean() * 100
    p300 = (y_true[order[:300]]).mean() * 100
    print("P@100: {0:.1f}, P@200: {1:.1f}, P@300: {2:.1f}, Mean: {3:.1f}".
          format(p100, p200, p300, (p100 + p200 + p300) / 3))
    # PR
    order = np.argsort(y_pred)[::-1]
    correct = 0.
    total = y_true.sum()
    precision = []
    recall = []
    for i, o in enumerate(order):
        correct += y_true[o]
        precision.append(float(correct) / (i + 1))
        recall.append(float(correct) / total)
    precision = np.array(precision)
    recall = np.array(recall)
    print("Saving result")
    np.save(os.path.join(save_dir, 'precision.npy'), precision)
    np.save(os.path.join(save_dir, 'recall.npy'), recall)
    return y_true, y_pred


if __name__ == '__main__':

    opt = vars(config())
    train_loader = data_loader(opt['train'], opt, shuffle=True, training=True)
    test_loader = data_loader(opt['test'], opt, shuffle=False, training=False)

    model = SeG(train_loader.dataset.vec_save_dir, train_loader.dataset.rel_num(),
                lambda_pcnn=opt['lambda_pcnn'], lambda_san=opt['lambda_san'])
    if torch.cuda.is_available():
        print('has cuda')
        model = model.cuda()
    cls_num_list = list(train_loader.dataset.class_numdict.values())
    num_classes = len(cls_num_list)
    criterion = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight())
    # criterion = SCELoss(alpha=opt['alpha'], beta=opt['beta'], num_classes=num_classes)

    # cls_num_list = list(train_loader.dataset.class_numdict.values())
    # # print(cls_num_list)
    # criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=train_loader.dataset.loss_weight()).cuda()
    # 'Focal':
    # criterion = FocalLoss(weight=train_loader.dataset.loss_weight(), gamma=1).cuda()

    optimizer = optim.SGD(model.parameters(), lr=opt['lr'], weight_decay=1e-5)
    # 存上一次循环里的得到的预测的new label 作为下一次里面的y～
    y_file = os.path.join(opt['save_dir'], "y.npy")
    # 对噪声标签应用交叉熵训练，不同的是只用大学习率训练，避免过拟合于噪声标签，
    # 得到的网络参数作为下一步训练的初始化网络参数
    # print("\n=== Backbone learning ===" )
    # vaild_loader = test_loader
    # _, _ = valid(vaild_loader,model,criterion)

    train(train_loader, test_loader, opt,model, criterion, optimizer)

    # 取效果最好的model取算test
    save_dir = opt['save_dir']
    model = SeG(test_loader.dataset.vec_save_dir, test_loader.dataset.rel_num(),
             lambda_pcnn=opt['lambda_pcnn'], lambda_san=opt['lambda_san'])
    if torch.cuda.is_available():
        model = model.cuda()
    state_dict = torch.load(os.path.join(save_dir, 'model.pth.tar'))['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)

    y_true, y_pred = test(test_loader,model,save_dir)

