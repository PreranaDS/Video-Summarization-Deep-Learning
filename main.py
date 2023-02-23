from utils import *
from models import *

import sys
sys.path.insert(0,'VASNet/VASNet')
from main import eval_split
import torch
from torchvision import transforms
import numpy as np
import time
import glob
import random
import argparse
import h5py
import json
import torch.nn.init as init
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from config import  *
from sys_utils import *
from vsum_tools import  *
from vasnet_model import  *



def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)

def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum
    dataset_type = sfname.split('_')[1]  # augmentation type e.g. aug

    # The keyword 'splits' is used as the filename fields terminator from historical reasons.
    if dataset_type == 'splits':
        # Split type is not present
        dataset_type = ''

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    return dataset_name, dataset_type, splits

def lookup_weights_splits_file(path, dataset_name, dataset_type, split_id):
    
    dataset_type_str = '' if dataset_type == '' else dataset_type + '_'
    weights_filename = path + 'models/{}_{}splits_{}_*.tar.pth'.format(dataset_name, dataset_type_str, split_id)
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return ''

    # Get the first weights file in the dir
    weights_filename = weights_filename[0]
    splits_file = path + 'splits/{}_{}splits.json'.format(dataset_name, dataset_type_str)

    return weights_filename, splits_file


class AONet:

    def __init__(self, hps: HParameters):
        self.hps = hps
        self.model = None
        self.log_file = None
        self.verbose = hps.verbose


    def fix_keys(self, keys, dataset_name = None):
        """
        :param keys:
        :return:
        """
        # dataset_name = None
        if len(self.datasets) == 1:
            dataset_name = next(iter(self.datasets))

        keys_out = []
        for key in keys:
            t = key.split('/')
            if len(t) != 2:
                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(len(self.datasets))

                key_name = dataset_name+'/'+key
                keys_out.append(key_name)
            else:
                keys_out.append(key)

        return keys_out


    def load_datasets(self, datasets = None):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
        :return:
        """
        if datasets is None:
            datasets = self.hps.datasets

        datasets_dict = {}
        for dataset in datasets:
            _, base_filename = os.path.split('VASNet/VASNet'+dataset) #Remove VASNet/VASNet/datasets if you want to run main here
            base_filename, _ = os.path.splitext(base_filename)
            print("Loading:", 'VASNet/VASNet/'+dataset)
            # dataset_name = base_filename.split('_')[2]
            # print("\tDataset name:", dataset_name)
            datasets_dict[base_filename] = h5py.File('VASNet/VASNet/'+dataset, 'r')

        self.datasets = datasets_dict
        return datasets_dict


    def load_split_file(self, splits_file):

        self.dataset_name, self.dataset_type, self.splits = parse_splits_filename('VASNet/VASNet/'+splits_file)
        n_folds = len(self.splits)
        self.split_file = splits_file
        print("Loading splits from: ",'VASNet/VASNet/'+splits_file)

        return n_folds


    def select_split(self, split_id):
        print("Selecting split: ",split_id)

        self.split_id = split_id
        n_folds = len(self.splits)
        assert self.split_id < n_folds, "split_id (got {}) exceeds {}".format(self.split_id, n_folds)

        split = self.splits[self.split_id]
        self.train_keys = split['train_keys']
        self.test_keys = split['test_keys']

        dataset_filename = self.hps.get_dataset_by_name(self.dataset_name)[0]
        _,dataset_filename = os.path.split(dataset_filename)
        dataset_filename,_ = os.path.splitext(dataset_filename)
        self.train_keys = self.fix_keys(self.train_keys, dataset_filename)
        self.test_keys = self.fix_keys(self.test_keys, dataset_filename)
        return



    def load_model(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        return


    def initialize(self, cuda_device=None):
        rnd_seed = 12345
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

        self.model = VASNet()
        self.model.eval()
        self.model.apply(weights_init)
        #print(self.model)

##        cuda_device = cuda_device or self.hps.cuda_device
##
##        if self.hps.use_cuda:
##            print("Setting CUDA device: ",cuda_device)
##            #torch.cuda.set_device(cuda_device)
##            #torch.cuda.manual_seed(rnd_seed)
##
##        if self.hps.use_cuda:
##            self.model.cuda()

        return


    def get_data(self, key):
        key_parts = key.split('/')
        assert len(key_parts) == 2, "ERROR. Wrong key name: "+key
        dataset, key = key_parts
        return self.datasets[dataset][key]

    def lookup_weights_file(self, data_path):
        dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
        weights_filename = data_path + '/models/{}_{}splits_{}_*.tar.pth'.format(self.dataset_name, dataset_type_str, self.split_id)
        print(weights_filename)
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ''

        # Get the first weights filename in the dir
        weights_filename = weights_filename[0]
        splits_file = data_path + '{}_{}splits.json'.format(self.dataset_name, dataset_type_str)

        return weights_filename, splits_file


    def train(self, output_dir='EX-0'):

        print("Initializing VASNet model and optimizer...")
        self.model.train()

        criterion = nn.MSELoss()

##        if self.hps.use_cuda:
##            criterion = criterion.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.hps.lr[0], weight_decay=self.hps.l2_req)

        print("Starting training...")

        max_val_fscore = 0
        max_val_fscore_epoch = 0
        train_keys = self.train_keys[:]

        lr = self.hps.lr[0]
        for epoch in range(self.hps.epochs_max):

            print("Epoch: {0:6}".format(str(epoch)+"/"+str(self.hps.epochs_max)), end='')
            self.model.train()
            avg_loss = []

            random.shuffle(train_keys)

            for i, key in enumerate(train_keys):
                dataset = self.get_data(key)
                seq = dataset['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                target = dataset['gtscore'][...]
                target = torch.from_numpy(target).unsqueeze(0)

                # Normalize frame scores
                target -= target.min()
                target /= target.max()

                if self.hps.use_cuda:
                    seq, target = seq.float(), target.float()

                seq_len = seq.shape[1]
                y, _ = self.model(seq,seq_len)
                loss_att = 0

                loss = criterion(y, target)
                # loss2 = y.sum()/seq_len
                loss = loss + loss_att
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_loss.append([float(loss), float(loss_att)])

            # Evaluate test dataset
            val_fscore, video_scores = self.eval(self.test_keys)
            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                max_val_fscore_epoch = epoch

            avg_loss = np.array(avg_loss)
            print("   Train loss: {0:.05f}".format(np.mean(avg_loss[:, 0])), end='')
            print('   Test F-score avg/max: {0:0.5}/{1:0.5}'.format(val_fscore, max_val_fscore))

            if self.verbose:
                video_scores = [["No", "Video", "F-score"]] + video_scores
                print_table(video_scores, cell_width=[3,40,8])

            # Save model weights
            path, filename = os.path.split(self.split_file)
            base_filename, _ = os.path.splitext(filename)
            path = os.path.join(output_dir, 'models_temp', base_filename+'_'+str(self.split_id))
            os.makedirs(path, exist_ok=True)
            filename = str(epoch)+'_'+str(round(val_fscore*100,3))+'.pth.tar'
            torch.save(self.model.state_dict(), os.path.join(path, filename))

        return max_val_fscore, max_val_fscore_epoch


    def eval(self, keys, results_filename=None):

        self.model.eval()
        summary = {}
        att_vecs = {}

        print("Initializa the reinforcement model")
        modelReinforcement = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
        print("Model size: {:.5f}M".format(sum(p.numel() for p in modelReinforcement.parameters())/1000000.0))

        optimizer = torch.optim.Adam(modelReinforcement.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.stepsize > 0:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

        
        print("Initialize MCSF model")
        # model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
        modelMCSF = SumInd(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers)
        print("Model size: {:.5f}M".format(sum(p.numel() for p in modelMCSF.parameters()) / 1000000.0))

        optimizer = torch.optim.Adam(modelMCSF.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.stepsize > 0:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

        if args.resume:
            print("Loading checkpoint from '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            modelMCSF.load_state_dict(checkpoint)
        else:
            start_epoch = 0
            
        with torch.no_grad():
            modelMCSF.eval()
            for i, key in enumerate(keys):
                data = self.get_data(key)
                # seq = self.dataset[key]['features'][...]
                seq = data['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)

##                if self.hps.use_cuda:
##                    seq = seq.float().cuda()

                y, att_vec = self.model(seq, seq.shape[1])
                summary[key] = y[0].detach().cpu().numpy()
                att_vecs[key] = att_vec.detach().cpu().numpy()

                
         


        f_score, video_scores = self.eval_summary(summary, seq, keys, 
                     modelMCSF=modelMCSF, modelReinforcement=modelReinforcement, results_filename=results_filename, metric=self.dataset_name, att_vecs=att_vecs)

        return f_score, video_scores

    
    def eval_summary(self, machine_summary_activations, seq, test_keys, modelMCSF, modelReinforcement, results_filename=None, metric='tvsum', att_vecs=None ):
            
        eval_metric = 'avg' if metric == 'tvsum' else 'max'

        if results_filename is not None:
            h5_res = h5py.File(results_filename, 'w')

        fms = []
        video_scores = []
        probsVASNet = 0
        for key_idx, key in enumerate(test_keys):
            d = self.get_data(key)
            probsVASNet = machine_summary_activations[key]
            
        probsMCSF = modelMCSF(seq)
        probsMCSF = probsMCSF.data.cpu().squeeze().numpy()
        probsReinforcement = modelReinforcement(seq)
        probs = np.zeros(len(probsVASNet))
        probs = (probsVASNet + probsMCSF + probsReinforcement[0].detach().numpy().reshape(probsVASNet.shape))/3
####        for i in range(len(probsVASNet)):
####            if probsVASNet[i]>0.6 and probsMCSF[i]>0.6 and probsReinforcement[i]>0.6:
####                probs[i]=1
####            elif (sum(probsVASNet[i] + probsMCSF[i] + probsReinforcement[i])/3) > 0.5:
####                probs[i]=0.7
####            else:
####                probs[i]=0
        if 'change_points' not in d:
            print("ERROR: No change points in dataset/video ",key)

        cps = d['change_points'][...]
        num_frames = d['n_frames'][()]
        nfps = d['n_frame_per_seg'][...].tolist()
        positions = d['picks'][...]
        user_summary = d['user_summary'][...]

        machine_summary = generate_summary(probs, cps, num_frames, nfps, positions)
        fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
        fms.append(fm)

        # Reporting & logging
        video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])

        if results_filename:
            gt = d['gtscore'][...]
            h5_res.create_dataset(key + '/score', data=probs)
            h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
            h5_res.create_dataset(key + '/gtscore', data=gt)
            h5_res.create_dataset(key + '/fm', data=fm)
            h5_res.create_dataset(key + '/picks', data=positions)

            video_name = key.split('/')[1]
            if 'video_name' in d:
                video_name = d['video_name'][...]
            h5_res.create_dataset(key + '/video_name', data=video_name)

            if att_vecs is not None:
                h5_res.create_dataset(key + '/att', data=att_vecs[key])

        mean_fm = np.mean(fms)

        # Reporting & logging
        if results_filename is not None:
            h5_res.close()

        return mean_fm, video_scores

        
#==========================================================================================================================================================================================

def eval_split(hps, splits_filename, data_dir='test'):

    print("\n")
    ao = AONet(hps)
    ao.initialize()
    ao.load_datasets()
    ao.load_split_file(splits_filename)

    val_fscores = []
    for split_id in range(len(ao.splits)):
        ao.select_split(split_id)
        weights_filename, _ = ao.lookup_weights_file('VASNet/VASNet/'+data_dir)
        print("Loading model:", weights_filename)
        ao.load_model(weights_filename)
        val_fscore, video_scores = ao.eval(ao.test_keys)
        val_fscores.append(val_fscore)

        val_fscore_avg = np.mean(val_fscores)

        if hps.verbose:
            video_scores = [["No.", "Video", "F-score"]] + video_scores
            print_table(video_scores, cell_width=[4,45,5])

        print("Avg F-score: ", val_fscore)
        print("")

    print("Total AVG F-score: ", val_fscore_avg)
    return val_fscore_avg

if __name__ == "__main__":
    print_pkg_versions()

    parser = argparse.ArgumentParser("PyTorch implementation of paper \"Ensemble Video Summarization\"")
    parser.add_argument('-r', '--root', type=str, default='', help="Project root directory")
    parser.add_argument('-d', '--datasets', type=str, help="Path to a comma separated list of h5 datasets")
    parser.add_argument('-s', '--splits', type=str, help="Comma separated list of split files.")
    parser.add_argument('-t', '--train', action='store_true', help="Train")
    parser.add_argument('-v', '--verbose', action='store_true', help="Prints out more messages")
    parser.add_argument('-o', '--output-dir', type=str, default='data', help="Experiment name")
    hps = HParameters()
    args = parser.parse_args()
    hps.load_from_args(args.__dict__)
    # Model options
    parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
    parser.add_argument('--hidden-dim', type=int, default=128, help="hidden unit dimension of SumInd (default: 128)")
    parser.add_argument('--num-layers', type=int, default=2, help="number of SumInd layers (default: 2)")
    parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
    # Optimization options
    parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
    parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
    parser.add_argument('--max-epoch', type=int, default=200, help="maximum epoch for training (default: 60)")
    parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
    parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
    parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
    parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")
    # Misc
    parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
    parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
    parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
    parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
    parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
    parser.add_argument('--resume', type=str, default='', help="path to resume file")
    parser.add_argument('--save-results', action='store_true', help="whether to save output results")
    args = parser.parse_args()

    # MAIN
    #======================
    
    torch.manual_seed(args.seed)

    print("Parameters:")
    print("----------------------------------------------------------------------")
    print(hps)
    avg=0
    if hps.train:
        train(hps)
    else:
        results=[['No', 'Split', 'Mean F-score']]
        for i, split_filename in enumerate(hps.splits):
            f_score = eval_split(hps, split_filename, data_dir=hps.output_dir)
            results.append([i+1, split_filename, str(round(f_score * 100.0, 3))+"%"])
            avg+=round(f_score * 100.0, 3)
        print("\nFinal Results:")
        print_table(results)
        print("Average f-score:", avg/4)


    sys.exit(0)
