import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--blr', type=float, default=5e-5, help='base_learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--testsize', type=int, default=384, help='testing dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=5e-2, help='decay rate of learning rate')

parser.add_argument('--load', type=str, default=r'/home/typ123321/projects/SFINet/pretrained/.pth')


parser.add_argument('--train_data_root', type=str, default=r'/home/cjb123321/datasets/VT5000-Train_unalign', help='the training datasets root')
parser.add_argument('--val_data_root', type=str, default=r'/home/cjb123321/datasets/VT821_unalign', help='the value datasets root')
parser.add_argument('--test_data_root', type=str, default=r'/home/cjb123321/datasets/VT821_unalign', help='the test datasets root')


parser.add_argument('--save_path', type=str, default='./res/', help='the path to save models and logs')
parser.add_argument('--test_model', type=str, default='/home/cjb123321/projects/SFCNet/res/.pth', help='saved model path')
parser.add_argument('--maps_path', type=str, default='./maps/test/', help='saved out path')



opt = parser.parse_args()
