import torch
import argparse
import json
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import os
import pandas as pd
from torch.utils.data import DataLoader


# local
from utils import AverageMeter, make_dir
from dataset import MyDataset
from model import MyModel

# parser
parser = argparse.ArgumentParser(description='TravelCompetition')
parser.add_argument('--save_path', default='./exp', type=str,
                    help='save path')
parser.add_argument('--experiment_name', '--e', default=None, type=str,
                    help='experiment name')
parser.add_argument('--batch_size','--bs', default=32, type=int,
                    help='batch size')
parser.add_argument('--optim', default='adam', type=str,
                    help='optimizer', choices=['sgd','adam','adagrad'])
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')
parser.add_argument('--lr_decay', default=1e-3, type=float,
                    help='learning rate decay')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='weight_decay')
parser.add_argument('--epochs', default=100, type=int,
                    help='train epoch')

# For test
parser.add_argument('--test_only', action='store_true',
                    help='How To Make TRUE? : --test_only, Flase : default')
parser.add_argument('--test_path', default='DeFaUlT', type=str,
                    help='test할 model이 있는 폴더의 이름. 해당 폴더는 ./exp에 있어야한다')

# GPU srtting							
parser.add_argument('--gpu_id', default='0', type=str,
                    help='How To Check? : cmd -> nvidia-smi')
args = parser.parse_args()
start = time.time()



def train(model, train_loader, criterion, optimizer, epoch, num_epoch):
    model.train()
    train_loss = AverageMeter()
    for i, (txt, img, label) in enumerate(train_loader):
        txt, img, label = txt.cuda(), img.cuda(), label.cuda()
        y_pred = model(txt, img)
        
        loss = criterion(y_pred, label.squeeze(dim=-1))
        train_loss.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 and i != 0:
            print(f'Epoch : [{epoch}/{num_epoch}] [{i}/{len(train_loader)}]  Train Loss : {loss:.4f}')






# test set의 정답을 우리가 알 수 없으므로 사실상 inference의 기능만 한다
def test(model, test_loader, save_path):
    print("=================== Test Start ====================")

    # no metrics

    # model test
    position = 0
    answer = pd.read_csv('E:/TravelCompetition/sample_submission.csv')
    targets = pd.read_csv('E:/TravelCompetition/train.csv')['cat3'].unique()
    category_dict = {category: idx for idx, category in enumerate(targets)}
    category_dict_reverse = dict(zip(category_dict.values(), category_dict.keys())) # key가 숫자 value가 관광지로 바뀌었다
    model.eval()
    with torch.no_grad():
        for i, (txt, img) in enumerate(test_loader):
            txt, img = txt.cuda(), img.cuda()
            y_pred = model(txt, img) # (bs, 128)
            y_pred = torch.argmax(y_pred,-1)
            y_pred = y_pred.detach().cpu()
            for i in y_pred.tolist():
                target = category_dict_reverse[i]
                answer.iloc[position, 1] = target
                position += 1
    
    # Save result
    answer.to_csv(f'{save_path}/answer.csv', index=False)
    
    print("=================== Test End ====================")







def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    ## Save path
    save_path = args.save_path + '/' + args.experiment_name
    make_dir(save_path)

    ## Save configuration
    with open(save_path + '/configuration.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # model load
    model = MyModel().cuda()

    # define criterion
    criterion = torch.nn.CrossEntropyLoss()

    # define Optimizer
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_deca)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    milestones = [int(args.epochs/3),int(args.epochs/2)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.7)
    # scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)


    # Train, Test
    if args.test_only: # test only
        model = MyModel().cuda()
        model.load_state_dict(torch.load(os.path.join('./exp', args.experiment_name, 'model.pth')))

        test_dataset = MyDataset('test')
        test_loader = DataLoader(test_dataset, args.batch_size, False, drop_last=False)
        test(model, test_loader, './exp/'+args.experiment_name, args)
    else: # train
        train_dataset = MyDataset('train')
        train_loader = DataLoader(train_dataset, args.batch_size, True, drop_last=False)
        for epoch in tqdm(range(1,args.epochs+1)):
            train(model, train_loader, criterion, optimizer, epoch, args.epochs)
            scheduler.step()
        
        torch.save(model.state_dict(), f'{save_path}/model.pth') 
        print(f'=== Training Completed ===')
        


        # test
        del train_dataset, train_loader
        test_dataset = MyDataset('test')
        test_loader = DataLoader(test_dataset, args.batch_size, False, drop_last=False)
        test(model, test_loader, save_path)

    print(f"Process Complete : it took {((time.time()-start)/60):.2f} minutes")



if __name__ == '__main__':
    main()