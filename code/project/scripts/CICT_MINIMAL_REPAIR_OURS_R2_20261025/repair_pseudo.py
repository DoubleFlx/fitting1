from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from ultralytics.utils.nms import non_max_suppression


@dataclass
class PseudoBatch:
    cls: torch.Tensor
    bboxes: torch.Tensor
    batch_idx: torch.Tensor
    stats: dict[str, Any]


def _teacher_inference_tensor(output: Any) -> torch.Tensor:
    inference = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(inference):
        raise TypeError(f"Teacher inference must be Tensor or tuple[Tensor,...], got {type(output)}")
    if inference.ndim != 3:
        raise RuntimeError(f"Teacher inference must have shape [B,C,N], got {tuple(inference.shape)}")
    if inference.shape[1] < 5:
        raise RuntimeError(f"Teacher inference channel count is invalid: {tuple(inference.shape)}")

    # Clone the teacher output before in-place NMS.
    inference = inference.detach().clone()
    if inference.is_inference():
        raise RuntimeError("Teacher prediction clone unexpectedly remained an inference tensor")
    return inference


def generate_high_confidence_pseudo(
    output: Any,
    *,
    conf: float,
    iou: float,
    max_det: int,
    imgsz: int,
    area_min: float,
    aspect_min: float,
    aspect_max: float,
) -> PseudoBatch:
    inference = _teacher_inference_tensor(output)
    if not bool(torch.isfinite(inference).all()):
        raise FloatingPointError("STOP_NONFINITE_TEACHER_OUTPUT")
    batch_size = int(inference.shape[0])
    class_scores = inference[:, 4:, :].amax(dim=1)
    pre_nms_per_image = (class_scores >= conf).sum(dim=1).detach().cpu().tolist()

    predictions = non_max_suppression(
        inference,
        conf_thres=conf,
        iou_thres=iou,
        max_det=max_det,
    )
    if len(predictions) != batch_size:
        raise RuntimeError(f"NMS returned {len(predictions)} images for batch size {batch_size}")

    cls_list: list[torch.Tensor] = []
    box_list: list[torch.Tensor] = []
    batch_idx_list: list[torch.Tensor] = []
    confidences: list[float] = []
    areas_final: list[float] = []
    post_nms_per_image: list[int] = []
    final_per_image: list[int] = []
    area_removed = 0
    aspect_removed = 0
    clipped_boxes = 0
    clipped_retained_boxes = 0
    clipped_per_image = []

    for image_index, det in enumerate(predictions):
        if det is None:
            det = inference.new_zeros((0, 6))
        post_nms_per_image.append(int(len(det)))
        if len(det) == 0:
            final_per_image.append(0)
            clipped_per_image.append(0)
            continue

        boxes = det[:, :4].clone()
        if not bool(torch.isfinite(boxes).all()):
            raise FloatingPointError("STOP_NONFINITE_PSEUDO_BEFORE_CLIP")
        was_clipped = ((boxes < 0) | (boxes > imgsz)).any(dim=1)
        n_clipped = int(was_clipped.sum().item())
        clipped_boxes += n_clipped
        clipped_per_image.append(n_clipped)
        boxes[:, 0::2].clamp_(0, imgsz)
        boxes[:, 1::2].clamp_(0, imgsz)
        confs = det[:, 4]
        classes = det[:, 5]

        boxes_norm = boxes / float(imgsz)
        widths = (boxes_norm[:, 2] - boxes_norm[:, 0]).clamp(min=0)
        heights = (boxes_norm[:, 3] - boxes_norm[:, 1]).clamp(min=0)
        areas = widths * heights

        valid_area = areas > area_min
        area_removed += int((~valid_area).sum().item())
        was_clipped = was_clipped[valid_area]
        boxes_norm = boxes_norm[valid_area]
        confs = confs[valid_area]
        classes = classes[valid_area]
        widths = widths[valid_area]
        heights = heights[valid_area]
        areas = areas[valid_area]

        if len(boxes_norm) == 0:
            final_per_image.append(0)
            continue

        aspect = widths / heights.clamp(min=1e-9)
        valid_aspect = (aspect >= aspect_min) & (aspect <= aspect_max)
        aspect_removed += int((~valid_aspect).sum().item())
        was_clipped = was_clipped[valid_aspect]
        clipped_retained_boxes += int(was_clipped.sum().item())
        boxes_norm = boxes_norm[valid_aspect]
        confs = confs[valid_aspect]
        classes = classes[valid_aspect]
        areas = areas[valid_aspect]

        if len(boxes_norm) == 0:
            final_per_image.append(0)
            continue
        if not torch.all(classes == 0):
            raise RuntimeError(f"Protocol B is single-class, but Teacher produced classes {classes.unique().tolist()}")

        x1, y1, x2, y2 = boxes_norm.unbind(dim=1)
        xywh = torch.stack(
            ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
            dim=1,
        )
        final_per_image.append(int(len(xywh)))
        cls_list.append(classes.float().unsqueeze(1))
        box_list.append(xywh.float())
        batch_idx_list.append(
            torch.full((len(xywh),), image_index, dtype=torch.long, device=xywh.device)
        )
        confidences.extend(float(value) for value in confs.detach().cpu())
        areas_final.extend(float(value) for value in areas.detach().cpu())

    device = inference.device
    if cls_list:
        cls_tensor = torch.cat(cls_list, dim=0)
        box_tensor = torch.cat(box_list, dim=0)
        batch_idx_tensor = torch.cat(batch_idx_list, dim=0)
    else:
        cls_tensor = torch.zeros((0, 1), dtype=torch.float32, device=device)
        box_tensor = torch.zeros((0, 4), dtype=torch.float32, device=device)
        batch_idx_tensor = torch.zeros((0,), dtype=torch.long, device=device)

    stats = {
        "batch_size": batch_size,
        "out_of_bounds_boxes": clipped_boxes,
        "clipped_boxes": clipped_boxes,
        "clipped_per_image": clipped_per_image,
        "clipped_boxes_removed_by_original_filters": clipped_boxes - clipped_retained_boxes,
        "pre_nms_candidates": int(sum(pre_nms_per_image)),
        "pre_nms_per_image": [int(value) for value in pre_nms_per_image],
        "post_nms_boxes": int(sum(post_nms_per_image)),
        "post_nms_per_image": post_nms_per_image,
        "area_filter_removed": int(area_removed),
        "aspect_filter_removed": int(aspect_removed),
        "final_boxes": int(len(cls_tensor)),
        "final_per_image": final_per_image,
        "positive_images": int(sum(count > 0 for count in final_per_image)),
        "empty_images": int(sum(count == 0 for count in final_per_image)),
        "confidences": confidences,
        "areas": areas_final,
    }
    return PseudoBatch(cls_tensor, box_tensor, batch_idx_tensor, stats)
