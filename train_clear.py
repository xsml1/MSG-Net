import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import random
import time
import numpy as np
from pathlib import Path

import utils
from lossses.losses import *
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm

from options import options
from model.teacher import TeacherNet
from model.restormer import Restormer
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim



######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def batch_SSIM(imgx, imgy, data_range):
    imgx = imgx.data.cpu().numpy().astype(np.float32)
    imgy = imgy.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(imgx.shape[0]):
        img1 = imgx[i, :, :, :]
        img2 = imgy[i, :, :, :]
        ssim1 = compare_ssim(img1[0, :, :], img2[0, :, :], data_range=data_range)
        ssim2 = compare_ssim(img1[1, :, :], img2[1, :, :], data_range=data_range)
        ssim3 = compare_ssim(img1[2, :, :], img2[2, :, :], data_range=data_range)
        SSIM = (ssim1 + ssim2 + ssim3) / 3.0
    return (SSIM / imgx.shape[0])


best_dir = os.path.join(options.output_path, 'Best')
Clear_dir = os.path.join(options.ckpt_path, 'Clear')
Degre_dir = os.path.join(options.ckpt_path, 'Degre')
utils.mkdir(best_dir)
utils.mkdir(Clear_dir)
utils.mkdir(Degre_dir)

######### Model ###########
Clear_teacher = Restormer()

# Clear_teacher.cuda()

os.environ.pop('CUDA_VISIBLE_DEVICES', None)

print(torch.cuda.device_count())
device_ids = [0, 1]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", len(device_ids), "GPUs!\n\n")
device_id = 0

# Clear_teacher = nn.DataParallel(Clear_teacher, device_ids=device_ids)
Clear_teacher = Clear_teacher.to(device_id)
new_lr = options.lr

optimizer_c = optim.AdamW(Clear_teacher.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)


######### Scheduler ###########
warmup_epochs = 3


scheduler_cosine_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, options.epochs - warmup_epochs,
                                                          eta_min=options.min_lr)
scheduler_c = GradualWarmupScheduler(optimizer_c, multiplier=1, total_epoch=warmup_epochs,
                                     after_scheduler=scheduler_cosine_c)

scheduler_c.step()

######### Resume ###########
if options.is_pretrain:
    path_chk_rest = best_dir
    utils.load_checkpoint(Clear_teacher, str(best_dir + '/clear_best.pth'))
    utils.load_checkpoint(Degre_teacher, str(best_dir + '/degre_best.pth'))
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer_e, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler_e.step()
    new_lr = scheduler_e.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

# if len(device_ids)>1:
#     model_restoration = nn.DataParallel(scheduler_e, device_ids = device_ids)
#     print("duoka")

######### Loss ###########
criterion_char = CharbonnierLoss()
VGG = VGG19_PercepLoss()
criterion_char = criterion_char.to(device_id)
VGG = VGG.to(device_id)
from dataset_new import PromptTrainDataset, TestSpecificDataset

######### DataLoaders ###########

#


train_dataset = PromptTrainDataset(options)
# train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=options.batch_size, shuffle=True, num_workers=16,
                          drop_last=False, pin_memory=True)

# val_dataset = get_validation_data(val_dir, {'patch_size':opt.TRAINING.VAL_PS})
# val_dataset = TestSpecificDataset(options)
from dataset_new import test_dataset

# from torch.utils.data import random_split
#
# # 假设原验证集 val_dataset 总共有1000个样本
# val_size = len(test_dataset)
# subset_ratio = 0.1  # 抽取10%的数据（可根据显存调整）
# subset_size = int(val_size * subset_ratio)
# remainder_size = val_size - subset_size
#
# # 随机拆分
# val_subset, _ = random_split(
#     test_dataset,
#     [subset_size, remainder_size],
#     generator=torch.Generator().manual_seed(42)  # 固定随机种子保证可复现
# )
#
# # 创建子集的 DataLoader
# val_loader_subset = DataLoader(
#     val_subset,
#     batch_size=1,  # 保持与原验证相同的batch_size
#     shuffle=False,
#     num_workers=4,
#     pin_memory=True
# )
#
val_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False,
                        pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, options.epochs + 1))
print('===> Loading datasets')

best_clear_psnr = 0
best_degre_psnr = 0
best_clear_epoch = 0
best_degre_epoch = 0
global_step = 0

for epoch in range(start_epoch, options.epochs + 1):
    step = 0
    epoch_start_time = time.time()
    epoch_clear_loss = 0
    epoch_degre_loss = 0
    train_id = 1
    train_ssim = []
    train_psnr = []
    total_loss = 0.
    Clear_teacher.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        step += 1
        start_time = time.time()

        target = data[2].to(device_id)
        input_ = data[1].to(device_id)

        # ----------------clear teacher---------------------#
        optimizer_c.zero_grad()
        restored_c = Clear_teacher(target)
        lambda_per = 100
        lambda_char = 1
        loss_char = criterion_char(restored_c, target)
        loss_per = VGG(restored_c, target)

        loss_c = loss_char * lambda_char + loss_per * lambda_per
        loss_c.backward()
        optimizer_c.step()

        # ----------------clear teacher---------------------#


        # --------------------logging-----------------------#
        epoch_clear_loss += loss_c.item()

        global_step = global_step + 1

        clear_pred = torch.clamp(restored_c, 0., 1.)
        total_loss+=loss_c.item()

        psnr_clear = batch_PSNR(clear_pred, target, 1.)
        ssim_clear = batch_SSIM(clear_pred, target, 1.)
        train_psnr.append(psnr_clear)
        train_ssim.append(ssim_clear)
        len_batch = len(train_loader)
        batch_time = time.time()
        # batch_use_time = batch_time - start_time
    train_psnr = np.array(train_psnr)
    train_ssim = np.array(train_ssim)
    psnr_value = train_psnr.mean()
    ssim_value = train_ssim.mean()
    sys.stdout.write(
        '\r[epoch:%d/%d],[CLEAR_PSNR:%f],[CLEAR_SSIM:%f],[CLEAR_loss:%f]\n'
        % (
            epoch,
            options.epochs,
            psnr_value,
            ssim_value,
            total_loss,

        )
    )
        # --------------------logging-----------------------#

    # psnr_te = 0
    # psnr_te_1 = 0
    # ssim_te_1 = 0
    #### Evaluation ####
    if epoch % options.val_epo ==0:
        torch.save({'epoch': epoch,
                    'state_dict': Clear_teacher.state_dict(),
                    'optimizer': optimizer_c.state_dict()
                    }, str(Clear_dir + '/clear_epoch{}.pth'.format(epoch + 1)))


        # Clear_teacher.eval()
        # with torch.no_grad():
        #
        #     test_ssim = []
        #     test_psnr = []
        #
        #     for i, batch2 in enumerate(tqdm(val_loader)):
        #         target = batch2[1].to(device_id)
        #         pred = Clear_teacher(target)
        #
        #         batch_psnr = batch_PSNR(pred, target, 1.)
        #         batch_ssim = batch_SSIM(pred, target, 1.)
        #         test_psnr = np.append(test_psnr, batch_psnr)
        #         test_ssim = np.append(test_ssim, batch_ssim)
        #
        #     psnr_value = test_psnr.mean()
        #     ssim_value = test_ssim.mean()
        #     if psnr_value > best_psnr:
        #         best_psnr = psnr_value
        #         Path(best_dir).mkdir(parents=True, exist_ok=True)
        #         torch.save({'epoch': epoch,
        #                     'state_dict': Clear_teacher.state_dict(),
        #                     'optimizer': optimizer_c.state_dict()
        #                     }, str(best_dir + "/clear_best.pth"))
        #     if ssim_value > best_ssim:
        #         best_ssim = ssim_value
        #         torch.save(model.state_dict(),
        #                    "/media/aa/a4b46d17-0f49-4392-98d6-49a5c9dee8e9/xujie/MFCR/saved_models_23/best/best_ssim_%d.pth" % (
        #                     e + 1))
        # if epoch % options.val_epo == 0:
        #     torch.save({'epoch': epoch,
        #                 'state_dict': Clear_teacher.state_dict(),
        #                 'optimizer': optimizer_c.state_dict()
        #                 }, str(Clear_dir + '/clear_epoch{}.pth'.format(epoch + 1)))
        #
        # Clear_teacher.eval()
        # psnr_val_clear = []
        # psnr_val_degrad = []
        # with torch.no_grad():
        #     for ii, data_val in enumerate(tqdm(val_loader_subset), 0):
        #         target = data_val[1].to(device_id)
        #         input_ = data_val[0].to(device_id)
        #
        #
        #         restored_c = Clear_teacher(target)
        #
        #         # clear_pred = torch.clamp(restored_c, 0., 1.)
        #         #
        #         #
        #         # psnr_clear = batch_PSNR(clear_pred, target, 1.)
        #         #
        #         # psnr_val_clear.append(psnr_clear)
        #
        #
        #
        # psnr_val_clear = torch.stack(psnr_val_clear).mean().item()
        # # psnr_val_degre = torch.stack(psnr_val_degre).mean().item()
        #
        #
        # #======================CLEAR================================#
        # if psnr_val_clear > best_clear_psnr:
        #     best_clear_psnr = psnr_val_clear
        #     best_clear_epoch = epoch
        #     Path(best_dir).mkdir(parents=True, exist_ok=True)
        #     torch.save({'epoch': epoch,
        #                 'state_dict': Clear_teacher.state_dict(),
        #                 'optimizer': optimizer_c.state_dict()
        #                 }, str(best_dir + "/clear_best.pth"))

        # print("[epoch %d PSNR: %.4f --- best_clear_epoch %d Best_CLEAR_PSNR %.4f]" % (epoch, psnr_val_clear, best_clear_epoch, best_clear_psnr))
        # Path(Clear_dir).mkdir(parents=True, exist_ok=True)

        # ========================CLEAR==============================#




    scheduler_c.step()


    print("------------------------------------------------------------------")