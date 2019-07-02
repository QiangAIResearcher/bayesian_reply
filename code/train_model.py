import platform
print('python_version==',platform.python_version())
import torch
print("torch.__version__ ==",torch.__version__)
from torch import optim
import torch.nn.functional as F
import argparse
from reader import TextReader
from model import claim_reply
import visdom
import numpy as np
from sklearn.metrics import f1_score,roc_auc_score
import torch.nn as nn

## use the train_baseline4 as the final train_model

parser = argparse.ArgumentParser(description='Variational Document Model')
parser.add_argument("--learning-rate", type=float, default=5e-4,
                    help='semeval2017task8(default: 5e-4)')
parser.add_argument("--h-dim", type=int, default=30,
                    help='semeval2017task8(default: 30)')
parser.add_argument("--embed-dim", type=int, default=300)
parser.add_argument("--user-embed-dim", type=int, default=30)
parser.add_argument("--sample-number", type=int, default=20)
parser.add_argument("--var-dim", type=int, default=10,
                    help='semeval2017task8(default: 5)')
parser.add_argument('--batch-size', type=int, default=5, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='semeval2017task8(default: 40)')
parser.add_argument("--max-norm", type=float, default=1e-2,
                    help='semeval2017task8(default: 1e-2)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def test(model,epoch=0):

    n_train = len(model.reader.train_claim)
    x1, x2, alabel, blabel, user = model.reader.random(data_type="train", random_bool=False)
    with torch.no_grad():
        target_b = blabel
        input1 = x1
        input2 = x2

        target_b = torch.LongTensor(target_b).to(device)
        data1 = np.asarray(input1)
        len1 = np.asarray([len(sentence) for sentence in input1])
        data2 = np.asarray(input2)
        len2 = np.asarray([len(sentence) for sentence in input2])

        output_b = model(data1, data2, len1, len2, target_b)
        _, idx = torch.max(output_b.data, 1)
        predicted_b = 1 - idx  # if select the 0 column, it represents "true" (encoded as 1)
        f1_b = f1_score(target_b, predicted_b, average='micro')
        print(output_b[:10, :])
        print(predicted_b[:10], max(predicted_b), min(predicted_b))
        print(target_b[:10])
        train_auc_score = roc_auc_score(blabel, np.asarray(output_b.data)[:, 0], average='micro')

        print('====> Epoch: {} Veracity F1 on the train set: {:.4f}'.format(
            epoch, f1_b))
        print('====> Epoch: {} Veracity AUC on the train set: {:.4f}'.format(
            epoch, train_auc_score))

    n_test = len(model.reader.test_claim)
    x1, x2, alabel, blabel, user = model.reader.random(data_type="test", random_bool=False)
    with torch.no_grad():
        target_b = blabel
        input1 = x1
        input2 = x2

        target_b = torch.LongTensor(target_b).to(device)
        data1 = np.asarray(input1)
        len1 = np.asarray([len(sentence) for sentence in input1])
        data2 = np.asarray(input2)
        len2 = np.asarray([len(sentence) for sentence in input2])

        output_b = model(data1, data2, len1, len2, target_b)
        _, idx = torch.max(output_b.data, 1)
        predicted_b = 1 - idx  # if select the 0 column, it represents "true" (encoded as 1)
        f1_b = f1_score(target_b, predicted_b, average='micro')
        test_auc_score = roc_auc_score(blabel, np.asarray(output_b.data)[:, 0], average='micro')

        # accuracy_b = correct_b.float() / n_test
        # veracity_test_auc.append(test_auc_score)

        print('====> Epoch: {} Veracity F1 on the test set: {:.4f}'.format(
            epoch, f1_b))
        print('====> Epoch: {} Veracity AUC on the test set: {:.4f}'.format(
            epoch, test_auc_score))

    return train_auc_score,test_auc_score

def train(model,optimiser):
    train_loss_list = []
    veracity_train_auc = []
    veracity_test_auc = []
    for epoch in range(1,args.epochs+1):
        model.train()
        n_train = len(model.reader.train_claim)
        x1,x2,alabel,blabel,user = model.reader.random()
        for i in range(n_train // args.batch_size + 1):
            if (i + 1) * args.batch_size < n_train:
                target_b = blabel[i * args.batch_size: (i + 1) * args.batch_size]
                input1 = x1[i * args.batch_size: (i + 1) * args.batch_size]
                input2 = x2[i * args.batch_size: (i + 1) * args.batch_size]
            else:
                target_b = blabel[i * args.batch_size:]
                input1 = x1[i * args.batch_size:]
                input2 = x2[i * args.batch_size:]

            target_b = torch.LongTensor(target_b).to(device)
            data1 = np.asarray(input1)
            len1 = np.asarray([len(sentence) for sentence in input1])
            data2 = np.asarray(input2)
            len2 = np.asarray([len(sentence) for sentence in input2])

            optimiser.zero_grad()
            output_b = model(data1,data2,len1,len2,target_b)
            loss = model.loss_function(output_b,target_b)
            loss.backward()
            train_loss = loss.item()
            train_loss_list.append(train_loss)
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=args.max_norm)
            optimiser.step()

            _, idx = torch.max(output_b.data, 1)
            predicted_b = 1 - idx  # if select the 0 column, it represents "true" (encoded as 1)
            # correct_b = (predicted_b == target_b).sum()
            # accuracy_b = correct_b.float() / args.batch_size
            f1_b = f1_score(target_b,predicted_b,average='micro')

            if i % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.4f}'.format(
                    epoch, i * args.batch_size, n_train,
                           100. * i*args.batch_size / n_train,
                           train_loss/args.batch_size))

        model.eval()
        train_auc_score,test_auc_score= test(model,epoch=epoch)
        veracity_train_auc.append(train_auc_score)
        veracity_test_auc.append(test_auc_score)

    return train_loss_list,veracity_train_auc,veracity_test_auc

if __name__ == '__main__':
    mode = input('mode(train/load)? ')
    model_path = './saved_model/bayesian_reply.pt'
    data_path = "./saved_data"
    reader = TextReader(data_path)
    model = claim_reply(reader=reader, batch_size=args.batch_size,
                        h_dim=args.h_dim, var_dim=args.var_dim,
                        sample_number = args.sample_number).to(device)
    if mode == 'train':
        optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
        train_loss_list,veracity_train_auc,veracity_test_auc = train(model,optimiser)

        # vis = visdom.Visdom()
        # vis.line(train_loss_list)
        #
        # vis2 = visdom.Visdom()
        # vis2.line(veracity_train_auc)
        # vis2.line(veracity_test_auc)
        #
        # torch.save(model.state_dict(), model_path)

    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        _,_ = test(model,epoch=0)
print("Done!")