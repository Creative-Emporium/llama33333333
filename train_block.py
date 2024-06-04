
from argparse import ArgumentParser
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import time
import os
from datetime import date
from tensorboardX import SummaryWriter
import shutil
import json
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, model_parallel_is_initialized

from llama.model import ModelArgs, Transformer, TransformerBlock
from trainer.optim import ExpScheduler
from trainer.utils import run_train_epoch, run_validation_epoch, print_performances, reportScalars



class NoiseDataset (torch.utils.data.Dataset):
	def __init__(self, size, shape, std=1, device='cuda', seed=None):
		self.size = size
		self.shape = shape
		self.std = std
		self.device = device
		self.generator = torch.Generator()
		self.seed = seed

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		if idx == 0 and self.seed is not None:
			self.generator.manual_seed(self.seed)

		noise = torch.randn(*self.shape, generator=self.generator) * self.std
		return noise.to(self.device)


class BlockCouple (nn.Module):
	def __init__ (self, teacher, student, freqs_cis):
		super().__init__()

		self.teacher = teacher
		self.student = student
		self.freqs_cis = freqs_cis


	def forward (self, input):
		n_seq = input.shape[1]

		freqs_cis = self.freqs_cis[:n_seq].to(input.device)

		mask = torch.full((n_seq, n_seq), float('-inf'), device=input.device)
		mask = torch.triu(mask, diagonal=1)
		mask = torch.hstack([torch.zeros((n_seq, 0), device=input.device), mask]).type_as(input)

		target = self.teacher(input, 0, freqs_cis, mask)
		pred = self.student(input, 0, freqs_cis, mask)

		loss = F.mse_loss(pred, target)

		error = loss.sqrt().cpu().item()
		accuracy = -math.log(error)
		metrics = dict(error=error, accuracy=accuracy)

		return loss, metrics


def load_llama (config):
	n_seq = config.data.n_seq
	batch_size = config.data.batch_size

	pretrain_model = config.model.args.pretrain_model

	params = json.loads(open(os.path.join(pretrain_model.root_dir, 'params.json'), 'r').read())
	model_args = ModelArgs(max_seq_len=n_seq, max_batch_size=batch_size, **params)
	model_args_relu = ModelArgs(max_seq_len=n_seq, max_batch_size=batch_size, **params, activation='relu')

	if not torch.distributed.is_initialized():
		torch.distributed.init_process_group('nccl')
	if not model_parallel_is_initialized():
		model_parallel_size = int(os.environ.get('WORLD_SIZE', 1))
		initialize_model_parallel(model_parallel_size)

	model = Transformer(model_args)

	torch.cuda.set_device(int(os.environ.get('RANK', 1)))

	checkpoint = torch.load(os.path.join(pretrain_model.root_dir, pretrain_model.checkpoint), map_location='cpu')
	model.load_state_dict(checkpoint, strict=False)

	return model, model_args_relu


def set_block_grad (block, config=None):
	block.requires_grad_(False)

	if config is not None:
		if config.ffn_norm:
			block.ffn_norm.requires_grad_(True)
		if 'feed_forward' in config:
			for name in config.feed_forward:
				getattr(block.feed_forward, name).requires_grad_(True)
		if 'attention' in config:
			for name in config.attention:
				getattr(block.attention, name).requires_grad_(True)



def main ():
	parser = ArgumentParser()
	parser.add_argument('config', help='config file')
	args = parser.parse_args()

	config = OmegaConf.load(args.config)

	if 'env' in config:
		for k, v in config.env.items():
			os.environ[k] = str(v)

	train_dir = os.path.join('./train', config.id.format(
		filename=os.path.splitext((os.path.basename(args.config)))[0],
		date=date.today().strftime('%Y%m%d')))
	train_dir = train_dir[:-6] if train_dir.endswith('.local') else train_dir
	os.makedirs(train_dir, exist_ok=True)

	device = config.trainer.device

	train_dataset = NoiseDataset(size=config.data.epoch_size, shape=(512, 4096))
	val_dataset = NoiseDataset(size=128, shape=(512, 4096), seed=1)

	train_data = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=config.data.batch_size,
		shuffle=True,
	)
	val_data = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=config.data.batch_size,
		shuffle=False,
	)

	layer_index = config.model.args.layer_index

	llama, model_args_relu = load_llama(config)
	teacher = llama.layers[layer_index]

	student = TransformerBlock(layer_index, model_args_relu)
	student.load_state_dict(teacher.state_dict())

	set_block_grad(teacher)
	set_block_grad(student, config.model.args.train_components)

	net = BlockCouple(teacher, student, llama.freqs_cis).to(device)

	weight_size = sum(p.numel() for p in student.parameters() if p.requires_grad)
	print(f'Trainable parameters: {weight_size:,}')

	optim = torch.optim.Adam(net.parameters(), lr=config.optim.scheduler.init_lr)
	optim_scheduler = ExpScheduler(optim, config.optim.scheduler.init_lr, config.optim.scheduler.decay, n_warmup_steps=config.optim.scheduler.warmup_steps)

	tb_writer = SummaryWriter(log_dir=train_dir)

	report_step = 0
	best_acc = 0

	for epoch in range(config.trainer.epochs):
		print(f'[Epoch {epoch}]')

		# validation
		t0 = time.time()
		val_loss, val_metrics = run_validation_epoch(net, val_data)
		print_performances('Validation', val_loss, val_metrics, t0)

		scalars = {
			'val_loss': val_loss,
		}
		for k, v in val_metrics.items():
			scalars['val_' + k] = v
		reportScalars(tb_writer, scalars, report_step)

		acc = val_metrics['accuracy']
		if acc > best_acc:
			best_acc = acc
			if epoch > 0:
				model_name = f'model_{epoch}-acc{acc:.4f}.chkpt'
				shutil.move(os.path.join(train_dir, 'latest.chkpt'), os.path.join(train_dir, model_name))

		# training
		t0 = time.time()
		train_loss, train_metrics = run_train_epoch(net, train_data, optim_scheduler)
		lr = optim.param_groups[0]['lr']
		print_performances('Train', train_loss, train_metrics, t0, lr=lr)

		checkpoint = {
			'epoch': epoch,
			'model': student.state_dict(),
			'optim': optim.state_dict(),
		}
		model_name = 'latest.chkpt'
		torch.save(checkpoint, os.path.join(train_dir, model_name))

		scalars = {
			'loss': train_loss,
			'learning_rate': lr,
			**train_metrics,
		}
		report_step = optim_scheduler.n_steps * config.data.batch_size
		reportScalars(tb_writer, scalars, report_step)


if __name__ == '__main__':
	main()
