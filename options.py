import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=2)

parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=4,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise', 'dehazing', 'deraining', 'low-light', 'underwater'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')

parser.add_argument('--train_file_dir', type=str, default='/media/aa/a4b46d17-0f49-4392-98d6-49a5c9dee8e9/xujie/AIOIR/data/train/',
                    help='training images dir.')
parser.add_argument('--val_file_dir', type=str, default='/media/aa/a4b46d17-0f49-4392-98d6-49a5c9dee8e9/xujie/AIOIR/data/test/',
                    help='validation images dir.')
parser.add_argument('--output_path', type=str, default="results/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="model_weights/", help='checkpoint save path')
parser.add_argument('--ckpt_name', type=str, default="model.ckpt", help='checkpoint save path')
parser.add_argument("--wblogger", type=str,default=None,help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir", type=str,default="train_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument('--resume', type=str, default=None, help='resume model path')
# parser.add_argument('--resume', type=str, default='model_weights/last.pth', help='resume model path')
parser.add_argument("--num_gpus", type=int,default=1, help="Number of GPUs to use for training")

options = parser.parse_args()

