import os
import csv

class Logger:
    def __init__(self, args):
        # save results
        if args.save_dir is None:
            result_dir = './save/'
        else:
            result_dir = './save/{}/'.format(args.save_dir)

        result_f = 'RFL_{}_EP{}_C{}_LBS[{}]_LE[{}]_IID[{}]_LR[{}]_MMT[{}]_NT[{}]_NR[{}]_TPL[{}]'.format(
            args.dataset,
            args.epochs,
            args.frac,
            args.local_bs,
            args.local_ep,
            args.iid,
            args.lr,
            args.momentum,
            args.noise_type,
            args.noise_rate,
            args.T_pl,
        )
          
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.f = open(result_dir + result_f + ".csv", 'w', newline='')
        self.wr = csv.writer(self.f)
        self.wr.writerow(['epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])


    def write(self, epoch, train_acc, train_loss, test_acc, test_loss,
              train_acc2=None, train_loss2=None, test_acc2=None, test_loss2=None):
        self.wr.writerow([epoch, train_acc, train_loss, test_acc, test_loss])

    def close(self):
        self.f.close()

