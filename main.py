import argparse
import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
torch.backends.cudnn.benchmark = True
from scheduler import CyclicCosineDecayLR
from config import GlobalConfig
from model_efnetb3_swin import SwinFuser
import torchvision

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='train', help='Unique experiment identifier')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=25, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=26, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to')
parser.add_argument('--gps_features', type = int, default=0, help='use more normalized GPS features')
parser.add_argument('--loss', type=str, default='ce', help='crossentropy or focal loss')
parser.add_argument('--scheduler', type=int, default=1, help='use scheduler to control the learning rate')
parser.add_argument('--load_previous_best', type=int, default=0, help='load previous best pretrained model ')
parser.add_argument('--temp_coef', type=int, default=1, help='apply temperature coefficience on the target')
parser.add_argument('--Test', type=int, default=0, help='Test')
parser.add_argument('--augmentation', type=int, default=1, help='data augmentation of camera')
parser.add_argument('--segment', type=int, default=0, help='add segmentation on 37&39 images')
parser.add_argument('--ema', type=int, default=1, help='exponential moving average')
parser.add_argument('--flip', type=int, default=0, help='flip all the data to augmentation')
args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)

writer = SummaryWriter(log_dir=args.logdir)

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

class Engine(object):
	"""Engine that runs training and inference.		
	"""
	def __init__(self,  cur_epoch=0, cur_iter=0):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.APL = [-100]
		self.bestval = -100
		self.APLft = [-100]
		if args.loss == 'ce':#crossentropy loss
			self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
		elif args.loss == 'focal':#focal loss
			self.criterion = FocalLoss()

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		model.train()
		pred_beam_all = []
		gt_pwr_all = []
		pred_pwr_all = []
		all_true_beams = []
		# Train loop
		pbar=tqdm(dataloader_train, desc='training')
		for data in pbar:
			# efficiently zero gradients
			optimizer.zero_grad(set_to_none=True)
			# create batch and move to GPU
			image_front_list = []
			image_back_list = []
			gps_list = []
			for i in range(config.seq_len):
				image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32))
				image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))

			for i in range(config.n_gps):
				gps_list.append(data['gps'][i].to(args.device, dtype=torch.float32))
    
			pred_beams = model(image_front_list + image_back_list, gps_list)
			pred_beam = torch.remainder( torch.argmax(pred_beams, dim=1)+1, 256 )
			gt_beamidx = data['beamidx'][0].to(args.device, dtype=torch.long)
			gt_beams = data['beam'][0].to(args.device, dtype=torch.float32)
			if args.temp_coef:
				loss = self.criterion(pred_beams, gt_beams)
			else:
				loss = self.criterion(pred_beams, gt_beamidx)
			all_true_beams.append(data['all_true_beams'][0])
			pred_beam_all.append(pred_beam.cpu().numpy())
			true_pwr_batch = data['beam_pwr'][0].to(args.device, dtype=torch.float32)
			gt_beamidx_shifted = torch.remainder( gt_beamidx + 1, 256)
			gt_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx_shifted]).cpu().numpy())
			pred_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy())
			loss.backward()
			loss_epoch += float(loss.item())
			pbar.set_description(str(loss.item()))
			num_batches += 1
			optimizer.step()
   			# Exponential Moving Averages
			if args.ema:
				ema.update()	# during training, after update parameters, update shadow weights

			self.cur_iter += 1

		pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
		all_true_beams = np.squeeze(np.concatenate(all_true_beams, 0))
		pred_pwr_all = np.squeeze(np.concatenate(pred_pwr_all, 0))
		gt_pwr_all = np.squeeze(np.concatenate(gt_pwr_all, 0))
		APL_score = APL(gt_pwr_all, pred_pwr_all)
		curr_acc = compute_acc(all_true_beams, pred_beam_all)
		print('Train top beam acc: ',curr_acc, ' APL score: ',APL_score)
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		writer.add_scalar('APL_score_train', APL_score, self.cur_epoch)
		for i in range(len(curr_acc)):
			writer.add_scalars('curr_acc_train', {'beam' + str(i):curr_acc[i]}, self.cur_epoch)
		writer.add_scalar('curr_loss_train', loss_epoch, self.cur_epoch)
		if APL_score > self.APLft[-1]:
			self.APLft.append(APL_score)
			print(APL_score, self.APLft[-2], 'train APL score improved!')
		else:
			print('best APL: ',self.APLft[-1], ' dB')

	def validate(self):
		if args.ema:
			ema.apply_shadow()
		model.eval()
  
		with torch.no_grad():	
			num_batches = 0
			wp_epoch = 0.
			pred_beam_all=[]
			scenario_all = []
			gt_pwr_all = []
			pred_pwr_all = []
			all_true_beams = []
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_val), 0):
				# create batch and move to GPU
				image_front_list = []
				image_back_list = []
				gps_list = []
				for i in range(config.seq_len):
					image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32))
					image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))
    
				for i in range(config.n_gps):
					gps_list.append(data['gps'][i].to(args.device, dtype=torch.float32))
				
				pred_beams = model(image_front_list + image_back_list, gps_list)
				pred_beam = torch.remainder( torch.argmax(pred_beams, dim=1)+1, 256 )
				gt_beams = data['beam'][0].to(args.device, dtype=torch.float32)
				gt_beamidx = data['beamidx'][0].to(args.device, dtype=torch.long)
				pred_beam_all.append(pred_beam.cpu().numpy())
				all_true_beams.append(data['all_true_beams'][0])
				if args.temp_coef:
					loss = self.criterion(pred_beams, gt_beams)
				else:
					loss = self.criterion(pred_beams, gt_beamidx)
				wp_epoch += float(loss.item())
				num_batches += 1
				true_pwr_batch = data['beam_pwr'][0].to(args.device, dtype=torch.float32)
				gt_beamidx_shifted = torch.remainder( gt_beamidx + 1, 256)
				gt_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),gt_beamidx_shifted]).cpu().numpy())
				pred_pwr_all.append((true_pwr_batch[np.arange(pred_beam.shape[0]),pred_beam]).cpu().numpy())
				scenario_all.append(data['scenario'])
			
			all_true_beams = np.concatenate(all_true_beams,0) # (n_samples,256)
			pred_beam_all=np.squeeze(np.concatenate(pred_beam_all,0))
			scenario_all = np.squeeze(np.concatenate(scenario_all,0))
			pred_pwr_all = np.squeeze(np.concatenate(pred_pwr_all, 0)) # (n_samples,1)
			gt_pwr_all = np.squeeze(np.concatenate(gt_pwr_all, 0)) # (n_samples,1)
			#calculate accuracy and APL score according to different scenarios
			scenarios = ['scenario36', 'scenario37', 'scenario38', 'scenario39']
			for s in scenarios:
				beam_scenario_index = np.array(scenario_all) == s
				pred_pwr_s = pred_pwr_all[beam_scenario_index]
				gt_pwr_s = gt_pwr_all[beam_scenario_index]
				if np.sum(beam_scenario_index) > 0:
					curr_acc_s = compute_acc(all_true_beams[beam_scenario_index],pred_beam_all[beam_scenario_index])
					APL_score_s = APL(gt_pwr_s,pred_pwr_s)
					
					print(s, ' curr_acc: ', curr_acc_s, ' APL_score: ', APL_score_s)
					for i in range(len(curr_acc_s)):
						writer.add_scalars('curr_acc_val', {s + 'beam' + str(i):curr_acc_s[i]}, self.cur_epoch)
					writer.add_scalars('APL_score_val', {s:APL_score_s}, self.cur_epoch)

			curr_acc = compute_acc(all_true_beams, pred_beam_all)
   
			APL_score_val = APL(gt_pwr_all, pred_pwr_all)
			wp_loss = wp_epoch / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:d}, Batch {batch_num:d}:' + f' Wp: {wp_loss:3.3f}')
			print('Val top beam acc: ',curr_acc, 'APL score: ', APL_score_val)
			writer.add_scalars('APL_score_val', {'scenario_all':APL_score_val}, self.cur_epoch)
			writer.add_scalar('curr_loss_val', wp_loss, self.cur_epoch)

			self.val_loss.append(wp_loss)
			self.APL.append(float(APL_score_val))
			self.cur_epoch += 1

		if args.ema:
			ema.restore()	# after eval, restore model parameter


	def test(self):
		model.eval()
		with torch.no_grad():
			pred_beam_all=[]
			# Validation loop
			for batch_num, data in enumerate(tqdm(dataloader_test), 0):
				# create batch and move to GPU
				image_front_list = []
				image_back_list = []
				gps_list = []
				for i in range(config.seq_len):
					image_front_list.append(data['front_images'][i].to(args.device, dtype=torch.float32))
					image_back_list.append(data['back_images'][i].to(args.device, dtype=torch.float32))
    
				for i in range(config.n_gps):
					gps_list.append(data['gps'][i].to(args.device, dtype=torch.float32))
				
				pred_beams = model(image_front_list + image_back_list, gps_list)
				pred_beam = torch.argmax(pred_beams, dim=1)
				pred_beam_all.append(pred_beam.cpu().numpy())

			pred_beam_all = np.squeeze(np.concatenate(pred_beam_all, 0))
			df_out = pd.DataFrame()
			df_out['prediction'] = pred_beam_all
			df_out.to_csv('Beamwise_prediction.csv', index=False)

	def save(self):
		save_best = False
  
		if self.APL[-1] >= self.bestval:
			self.bestval = self.APL[-1]
			self.bestval_epoch = self.cur_epoch - 1
			save_best = True
		print(f'best APL = {self.bestval:.4f} dB @ epoch = {self.bestval_epoch}')

		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'iter': self.cur_iter,
			'bestval': self.bestval,
			'bestval_epoch': self.bestval_epoch,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
			'APL': self.APL,
		}

		# Save checkpoint for every epoch
		# Log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))
   
		# save the bestpretrained model
		if save_best:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'optim.pth'))
			tqdm.write('====== Overwrote best model ======>')
		elif args.load_previous_best:
			model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
			optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'optim.pth')))
			tqdm.write('====== Load the previous best model ======>')

class FocalLoss(nn.Module):
	def __init__(self, gamma=2, alpha=0.25):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
	def __call__(self, input, target):
		if len(target.shape) == 1:
			target = torch.nn.functional.one_hot(target, num_classes=256)
		loss = torchvision.ops.sigmoid_focal_loss(input, target.float(), alpha=self.alpha, gamma=self.gamma,
												  reduction='mean')
		return loss

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def compute_acc(all_beams, only_best_beam, top_k=[1, 3, 5]):
    
    """ 
    Computes top-k accuracy given prediction and ground truth labels.

    Note that it works bidirectionally. 
    <all_beams> is (N_SAMPLES, N_BEAMS) but it can represent:
        a) the ground truth beams sorted by receive power
        b) the predicted beams sorted by algorithm's confidence of being the best

    <only_best_beam> is (N_SAMPLES, 1) and can represent (RESPECTIVELY!):
        a) the predicted optimal beam index
        b) the ground truth optimal beam index

    For the competition, we will be using the function with inputs described in (a).

    """
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)

    n_test_samples = len(only_best_beam)
    if len(all_beams) != n_test_samples:
        raise Exception(
            'Number of predicted beams does not match number of labels.')

    # For each test sample, count times where true beam is in k top guesses
    for samp_idx in range(len(only_best_beam)):
        for k_idx in range(n_top_k):
            hit = np.any(all_beams[samp_idx, :top_k[k_idx]] == only_best_beam[samp_idx])
            total_hits[k_idx] += 1 if hit else 0

    # Average the number of correct guesses (over the total samples)
    return np.round(total_hits / len(only_best_beam)*100, 4)


def APL(true_best_pwr, est_best_pwr):
    """
    Average Power Loss: average of the power wasted by using the predicted beam
    instead of the ground truth optimum beam.
    """
    
    return np.mean(10 * np.log10(est_best_pwr / true_best_pwr))


# Config
config = GlobalConfig()
config.gps_features = args.gps_features
config.segment = args.segment

import random
import numpy as np
seed = 100
random.seed(seed)
np.random.seed(seed) # numpy
torch.manual_seed(seed) # torch+CPU
# torch.cuda.manual_seed(seed) # torch+GPU
torch.use_deterministic_algorithms(False)
g = torch.Generator()
g.manual_seed(seed)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============ DATASET
from load_data import MATTIA_Data

if not args.Test:
	train_val_split = 0.9
	train_root = config.root + '/Multi_Modal/'
	train_root_csv = 'deepsense_challenge2023_trainset.csv'
	development_set = MATTIA_Data(root=train_root, root_csv=train_root_csv, config=config, test=False, augment=args.augmentation)
	if args.flip:
		development_set_flipped = MATTIA_Data(root=train_root, root_csv=train_root_csv, config=config, test=False, augment=args.augmentation, flip=True)
		development_set = ConcatDataset([development_set, development_set_flipped])
	train_dim = int(train_val_split * len(development_set))
	train_set, val_set = torch.utils.data.random_split(development_set,[train_dim,len(development_set) - train_dim])
	dataloader_train = DataLoader(train_set,batch_size=args.batch_size,shuffle=True, num_workers=8, pin_memory=True,
									worker_init_fn=seed_worker, generator=g)
	dataloader_val = DataLoader(val_set,batch_size=args.batch_size,shuffle=True, num_workers=8, pin_memory=False,
									worker_init_fn=seed_worker, generator=g)
else:
	test_root = config.root + '/Multi_Modal_Test/'
	test_root_csv = 'challenge.csv'
	test_set = MATTIA_Data(root=test_root, root_csv=test_root_csv, config=config, test=True)
	print('test_set:', len(test_set))
	dataloader_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)
    

# Model
model = SwinFuser(config,args.device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

if args.scheduler:#Cyclic Cosine Decay Learning Rate
	scheduler = CyclicCosineDecayLR(optimizer,
	                                init_decay_epochs=15,
	                                min_decay_lr=1e-6,
	                                restart_interval = 7,
	                                restart_lr= 1e-4,
	                                warmup_epochs=7,
	                                warmup_start_lr=0.0001)
 
trainer = Engine()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('======Total trainable parameters: ', params)

# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('======Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
	print('======Loading checkpoint from ' + args.logdir)
	with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
		log_table = json.load(f)

	# Load variables
	trainer.cur_epoch = log_table['epoch']
	if 'iter' in log_table: trainer.cur_iter = log_table['iter']
	trainer.bestval = log_table['bestval']
	trainer.train_loss = log_table['train_loss']
	trainer.val_loss = log_table['val_loss']
	trainer.APL = log_table['APL']

	## Only for testing
	print('======loading best_model')
	model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
	optimizer.load_state_dict(torch.load(os.path.join(args.logdir,'optim.pth')))


ema = EMA(model, 0.999)

if args.ema:
	ema.register()

# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)
if args.Test:
	trainer.test()
	print('Test finished!')
else:
	for epoch in range(trainer.cur_epoch, args.epochs):
		print('\nepoch:',epoch)
		trainer.train()
		trainer.validate()
		trainer.save()
  
		if args.scheduler:
			print('lr', scheduler.get_lr())
			scheduler.step()

