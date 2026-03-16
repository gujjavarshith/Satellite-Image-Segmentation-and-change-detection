import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import config
from models.segformer_model import SegFormerChange
from utils.dataset import ChangeDataset
from utils.visualtization import save_prediction_triplet, save_prediction_with_legend


def load_checkpoint(model, ckpt_path):
	state = torch.load(ckpt_path, map_location="cpu")
	if isinstance(state, dict) and "model_state" in state:
		model.load_state_dict(state["model_state"], strict=False)
	else:
		model.load_state_dict(state, strict=False)


def predict_single(model, im1_path, im2_path, img_size):
	im1 = cv2.imread(im1_path)
	im2 = cv2.imread(im2_path)

	im1 = cv2.resize(im1, (img_size, img_size))
	im2 = cv2.resize(im2, (img_size, img_size))

	img = np.concatenate([im1 / 255.0, im2 / 255.0], axis=2)
	img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0)
	img = img.to(config.DEVICE)

	with torch.no_grad():
		logits = model(img)
		logits = F.interpolate(logits, size=(img_size, img_size), mode="bilinear", align_corners=False)
		pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype("uint8")

	return im1, im2, pred


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt", default=str(config.CHECKPOINT_PATH))
	parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
	parser.add_argument("--im1", default="")
	parser.add_argument("--im2", default="")
	parser.add_argument("--split", default="val", choices=["train", "val", "test"])
	parser.add_argument("--index", type=int, default=0)
	parser.add_argument("--num_images", type=int, default=30)
	parser.add_argument("--with_legend", action="store_true")
	parser.add_argument("--out", default="outputs/prediction.png")
	args = parser.parse_args()

	model = SegFormerChange(num_classes=config.NUM_CLASSES)
	load_checkpoint(model, args.ckpt)
	model.to(config.DEVICE)
	model.eval()

	if args.im1 and args.im2:
		im1, im2, pred = predict_single(model, args.im1, args.im2, args.img_size)
		out_path = Path(args.out)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		if args.with_legend:
			save_prediction_with_legend(
				im1,
				im2,
				pred,
				str(out_path),
				num_classes=config.NUM_CLASSES,
				class_names=config.CLASS_NAMES,
			)
		else:
			save_prediction_triplet(im1, im2, pred, str(out_path), num_classes=config.NUM_CLASSES)
		print(f"Saved prediction to: {out_path}")
	else:
		dataset = ChangeDataset(
			config.get_split_dir(args.split),
			img_size=args.img_size,
			return_name=True,
			num_classes=config.NUM_CLASSES,
			label_values=config.LABEL_VALUES,
		)
		count = max(1, min(args.num_images, 15))
		start_index = max(0, args.index)
		end_index = min(start_index + count, len(dataset))

		out_path = Path(args.out)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		base_stem = out_path.stem
		suffix = out_path.suffix if out_path.suffix else ".png"

		for idx in range(start_index, end_index):
			_, _, name = dataset[idx]
			im1_path = str(config.get_split_dir(args.split) / "im1" / name)
			im2_path = str(config.get_split_dir(args.split) / "im2" / name)
			im1, im2, pred = predict_single(model, im1_path, im2_path, args.img_size)

			multi_out = out_path.parent / f"{base_stem}_{idx:04d}{suffix}"
			if args.with_legend:
				save_prediction_with_legend(
					im1,
					im2,
					pred,
					str(multi_out),
					num_classes=config.NUM_CLASSES,
					class_names=config.CLASS_NAMES,
				)
			else:
				save_prediction_triplet(im1, im2, pred, str(multi_out), num_classes=config.NUM_CLASSES)
			print(f"Saved prediction: {multi_out}")


if __name__ == "__main__":
	main()