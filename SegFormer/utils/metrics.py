import torch


def _fast_hist(true, pred, num_classes, ignore_index=None):
	true = true.view(-1)
	pred = pred.view(-1)

	if ignore_index is not None:
		mask = true != ignore_index
		true = true[mask]
		pred = pred[mask]

	hist = torch.bincount(
		num_classes * true + pred,
		minlength=num_classes ** 2,
	).reshape(num_classes, num_classes)

	return hist


def update_confusion_matrix(conf_matrix, true, pred, num_classes, ignore_index=None):
	conf_matrix += _fast_hist(true, pred, num_classes, ignore_index)
	return conf_matrix


def compute_metrics(conf_matrix):
	conf_matrix = conf_matrix.float()
	diag = torch.diag(conf_matrix)
	denom = conf_matrix.sum(1) + conf_matrix.sum(0) - diag

	iou = torch.where(denom > 0, diag / denom, torch.zeros_like(denom))
	miou = iou.mean().item()
	acc = diag.sum().item() / conf_matrix.sum().item() if conf_matrix.sum() > 0 else 0.0
	per_class_acc = torch.where(
		conf_matrix.sum(1) > 0,
		diag / conf_matrix.sum(1),
		torch.zeros_like(diag),
	)

	return {
		"pixel_acc": acc,
		"miou": miou,
		"iou": iou.tolist(),
		"per_class_acc": per_class_acc.tolist(),
	}
