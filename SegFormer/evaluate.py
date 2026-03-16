import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from models.segformer_model import SegFormerChange
from utils.dataset import ChangeDataset
from utils.metrics import update_confusion_matrix, compute_metrics


def load_checkpoint(model, ckpt_path):
	state = torch.load(ckpt_path, map_location="cpu")
	if isinstance(state, dict) and "model_state" in state:
		model.load_state_dict(state["model_state"], strict=False)
	else:
		model.load_state_dict(state, strict=False)


def evaluate(split, ckpt_path, batch_size, img_size):
	dataset = ChangeDataset(
		config.get_split_dir(split),
		img_size=img_size,
		num_classes=config.NUM_CLASSES,
		label_values=config.LABEL_VALUES,
	)
	loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=config.NUM_WORKERS,
		pin_memory=config.PIN_MEMORY,
	)

	model = SegFormerChange(num_classes=config.NUM_CLASSES)
	load_checkpoint(model, ckpt_path)
	model.to(config.DEVICE)
	model.eval()

	conf_matrix = torch.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=torch.long)

	with torch.no_grad():
		for imgs, labels in loader:
			imgs = imgs.to(config.DEVICE)
			labels = labels.to(config.DEVICE)

			outputs = model(imgs)
			outputs = F.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)
			preds = torch.argmax(outputs, dim=1)

			conf_matrix = update_confusion_matrix(conf_matrix, labels.cpu(), preds.cpu(), config.NUM_CLASSES)

	return compute_metrics(conf_matrix)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--split", default="val", choices=["train", "val", "test"])
	parser.add_argument("--ckpt", default=str(config.CHECKPOINT_PATH))
	parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
	parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
	parser.add_argument("--out", default="")
	args = parser.parse_args()

	metrics = evaluate(args.split, args.ckpt, args.batch_size, args.img_size)
	print(json.dumps(metrics, indent=2))

	if args.out:
		out_path = Path(args.out)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		with open(out_path, "w", encoding="utf-8") as f:
			json.dump(metrics, f, indent=2)


if __name__ == "__main__":
	main()
