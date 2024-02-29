import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data', help='root of data files')
    parser.add_argument('--train', default='train.txt')
    parser.add_argument('--test', default='test.txt')
    parser.add_argument('--rel', default='relation2id.txt')
    parser.add_argument('--vec', default='vec.txt')
    parser.add_argument('--save_dir', default='result')
    parser.add_argument('--processed_data_dir', default='_processed_data')
    parser.add_argument('--batch_size', default=1, type=int)#160
    parser.add_argument('--max_bag_size', default=1, type=int)#20
    parser.add_argument('--max_length', default=120, type=int)#120
    parser.add_argument('--max_pos_length', default=20
                        , type=int)#100
    parser.add_argument('--epoch', default=1, type=int)#60
    parser.add_argument('--lr', default=0.1, type=float)# 0.1
    parser.add_argument('--val_iter', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lambda_pcnn', default=0.05, type=int)
    parser.add_argument('--lambda_san', default=1.0, type=int)
    parser.add_argument('--early_stop', default=30, type=int)#10

    parser.add_argument('--alpha', type=float, default=0.1, help='alpha scale')
    parser.add_argument('--beta', type=float, default=0.4, help='beta scale')
    parser.add_argument('--stage1', default=15, type=int,
                        metavar='H-P', help='number of epochs utill stage1')
    parser.add_argument('--stage2', default=30, type=int,
                        metavar='H-P', help='number of epochs utill stage2')
    parser.add_argument('--lr2', '--learning-rate2', default=0.2, type=float,
                        metavar='H-P', help='initial learning rate of stage3')
    parser.add_argument('--lambda1', default=1000, type=int,  # 200
                        metavar='H-P', help='the value of lambda')
    return parser.parse_args()

