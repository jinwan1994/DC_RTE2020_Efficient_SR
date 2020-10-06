"""
@project: sr_evaluation
@author: wj
@file: test.py
@ide: 
@time: 2020/9/17 15:32

"""
import argparse
import torch
from utils.dataloader import TestDataset
from torch.utils.data import DataLoader
from utils.util import load_state_dict, sr_forward_psnr, sr_forward_time
from torchprofile_ import profile_macs as profile
import os
import importlib
os.environ["CUDA_VISIBLE_DEVICES"] = '7'


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_model', type=str,default='architecture')
    parser.add_argument('--model_path', type=str, default='../pretrained-model/architecture.pth', help='path of checkpoint of test model')
    parser.add_argument('--baseline_path', type=str, default='../pretrained-model/baseline.pth', help='path of checkpoint of baseline')

    parser.add_argument('--LR_path', type=str, default='../datasets/test_data/LR', help='path of the LR images')
    parser.add_argument('--HR_path', type=str, default='../datasets/test_data/HR', help='path of the HR images')
    parser.add_argument('--upscale', type=int, default=2, help='scale factor for up-sample LR image ')
    parser.add_argument('--cuda', type=bool, default=True, help='whether use cuda or not')

    parser.add_argument('--alpha', type=float, default=2, help='the weight of alpha')
    parser.add_argument('--beta', type=float, default=4, help='the weight of beta')
    parser.add_argument('--gamma', type=float, default=0.8, help='the weight of gamma')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of inference')
    parser.add_argument('--cycle_num', type=int, default=5, help='the number of repeat running model')



    return parser.parse_args()


def evaluation(opt):

    device = torch.device('cuda' if opt.cuda else 'cpu')

    alpha = opt.alpha
    beta = opt.beta
    gamma = opt.gamma
    cycle_num = opt.cycle_num

    crop_boarder = opt.upscale

    # load dataset
    #print(opt.HR_path)
    dataset = TestDataset(opt.HR_path, opt.LR_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    test_times = 0.0

    module = importlib.import_module('model.{}'.format(opt.test_model))
    test_model = module.model(opt.upscale)
    state_dict = load_state_dict(opt.model_path)
    test_model.load_state_dict(state_dict)
    test_model = test_model.to(device)
    test_model.eval()

    # load baseline
    module = importlib.import_module('model.{}'.format('baseline'))
    baseline_model = module.model(opt.upscale)
    baseline_dict = load_state_dict(opt.baseline_path)
    baseline_model.load_state_dict(baseline_dict)
    baseline_model = baseline_model.to(device)
    baseline_model.eval()

	#calc FLOPs
    width = 360
    height = 240

    inputs = torch.randn(1, 3, height, width).to(device)
    macs = profile(test_model.to('cuda'), inputs)
    print('{:.4f} G'.format(macs / 1e9))



    save_path = '../results/{}'.format(opt.test_model)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'SR'))
        os.mkdir(os.path.join(save_path, 'GT'))




    test_psnr, test_ssim = sr_forward_psnr(dataloader, test_model, device, crop_boarder, save_path)
    baseline_psnr, baseline_ssim = sr_forward_psnr(dataloader, baseline_model, device, crop_boarder)
    
    test_times = 0
    baseline_times = 0
    for index in range(cycle_num):

        test_time = sr_forward_time(dataloader, test_model, device)
        test_times += test_time
        baseline_time = sr_forward_time(dataloader, baseline_model, device)
        baseline_times += baseline_time


    avg_test_time = (test_times / cycle_num)
    avg_baseline_time = (baseline_times / cycle_num)
    avg_time_score = gamma * min((avg_baseline_time / avg_test_time)/25, 2)

    score = alpha * (test_psnr - baseline_psnr) + beta * (test_ssim - baseline_ssim) + avg_time_score

    print('model: {}'.format(opt.test_model))
    print('test model: {:.4f}, base model: {:.4f}, psnr: {:.4f}'.format(test_psnr, baseline_psnr, alpha * (test_psnr-baseline_psnr)))
    print('test model: {:.4f}, base model: {:.4f}, ssim: {:.4f}'.format(test_ssim, baseline_ssim, beta * (test_ssim-baseline_ssim)))
    print('test model: {:.4f} ms, base model: {:.4f} ms, time: {:.4f} ms'.format(avg_test_time, avg_baseline_time, avg_time_score))
    print('score: {:.4f}'.format(score))



if __name__ == '__main__':

    opt = parser_args()
    evaluation(opt)
