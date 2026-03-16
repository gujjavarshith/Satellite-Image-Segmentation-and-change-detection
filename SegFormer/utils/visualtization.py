import numpy as np
import cv2

import config


def get_palette(num_classes=7):
	palette = list(config.CLASS_COLORS_BGR)
	if num_classes <= len(palette):
		return palette[:num_classes]

	extra = np.random.randint(0, 255, size=(num_classes - len(palette), 3))
	palette.extend([tuple(map(int, c)) for c in extra])
	return palette


def colorize_mask(mask, num_classes=7):
	palette = get_palette(num_classes)
	h, w = mask.shape
	color = np.zeros((h, w, 3), dtype=np.uint8)
	for cls_id in range(num_classes):
		color[mask == cls_id] = palette[cls_id]
	return color


def overlay_mask(image_bgr, mask, alpha=0.5, num_classes=7):
	color_mask = colorize_mask(mask, num_classes)
	return cv2.addWeighted(image_bgr, 1 - alpha, color_mask, alpha, 0)


def save_prediction_triplet(im1, im2, pred_mask, out_path, num_classes=7):
	pred_color = colorize_mask(pred_mask, num_classes)
	combined = np.concatenate([im1, im2, pred_color], axis=1)
	cv2.imwrite(out_path, combined)


def create_legend_panel(class_names=None, num_classes=7, width=320, row_h=38):
	if class_names is None:
		class_names = list(config.CLASS_NAMES)

	class_names = class_names[:num_classes]
	height = max(80, row_h * len(class_names) + 30)
	panel = np.full((height, width, 3), 30, dtype=np.uint8)

	palette = get_palette(num_classes)
	cv2.putText(panel, "Legend", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)

	for i, class_name in enumerate(class_names):
		y = 40 + i * row_h
		color = palette[i]
		cv2.rectangle(panel, (16, y), (46, y + 22), color, -1)
		cv2.rectangle(panel, (16, y), (46, y + 22), (220, 220, 220), 1)
		label = f"{i}: {class_name}"
		cv2.putText(panel, label, (58, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

	return panel


def save_prediction_with_legend(im1, im2, pred_mask, out_path, num_classes=7, class_names=None):
	pred_color = colorize_mask(pred_mask, num_classes)
	triplet = np.concatenate([im1, im2, pred_color], axis=1)
	legend = create_legend_panel(class_names=class_names, num_classes=num_classes)
	legend = cv2.resize(legend, (legend.shape[1], triplet.shape[0]), interpolation=cv2.INTER_NEAREST)
	final = np.concatenate([triplet, legend], axis=1)
	cv2.imwrite(out_path, final)
