
import torch
from tqdm import tqdm
import time
import math



def stat_average (data, n_batch):
	return dict([(k, v / n_batch) for k, v in data.items()])


def run_train_epoch (net, data, optim_scheduler):
	net.train()
	total_loss, n_batch = 0, 0
	metric_data = {}

	for batch in tqdm(data, desc='  - (Training)   '):
		optim_scheduler.zero_grad()
		loss, metrics = net(batch)

		loss.backward()
		optim_scheduler.step()

		n_batch += 1
		total_loss += loss.item()

		for k, v in metrics.items():
			metric_data[k] = metric_data[k] + v if k in metric_data else v

	metrics = stat_average(metric_data, n_batch)

	return total_loss / n_batch, metrics


def run_validation_epoch (net, data):
	net.eval()
	total_loss, n_batch = 0, 0
	metric_data = {}

	with torch.no_grad():
		for batch in tqdm(data, desc='  - (Validation) '):
			loss, metrics = net(batch)

			n_batch += 1
			total_loss += loss.item()

			for k, v in metrics.items():
				metric_data[k] = metric_data[k] + v if k in metric_data else v

	metrics = stat_average(metric_data, n_batch)

	return total_loss / n_batch, metrics


def print_metric (metric):
	if type(metric) == float or type(metric) == torch.Tensor:
		return f'accuracy: {metric*100:3.3f}%'
	elif type(metric) == dict:
		items = map(lambda item: f'{item[0]}: {item[1]:.4f}',
			filter(lambda item: type(item[1]) in [int, float],
				metric.items()))
		return ', '.join(items)
	else:
		return str(metric)


def print_performances(header, loss, metric, start_time, lr=math.nan):
	print('  - {header:12} loss: {loss: .4e}, {metric}, lr: {lr:.4e}, elapse: {elapse:3.2f} min'
		.format(header=f"({header})", loss=loss, metric=print_metric(metric), elapse=(time.time()-start_time)/60, lr=lr))


def reportScalars (tb_writer, scalars, step):
	for k, v in scalars.items():
		if type(v) == dict:
			for kk, vv in v.items():
				tb_writer.add_scalar(f'{k}/{kk}', vv, step)
		else:
			tb_writer.add_scalar(k, v, step)
