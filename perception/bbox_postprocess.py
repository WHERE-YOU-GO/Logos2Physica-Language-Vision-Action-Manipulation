from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from common.datatypes import BBox2D, Detection2D


def _iou(a: BBox2D, b: BBox2D) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = a.width() * a.height()
    area_b = b.width() * b.height()
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def clip_boxes_to_image(dets: list[Detection2D], image_shape: tuple[int, int]) -> list[Detection2D]:
    image_h, image_w = image_shape
    clipped: list[Detection2D] = []
    for det in dets:
        x1 = max(0, min(det.bbox.x1, image_w - 1))
        y1 = max(0, min(det.bbox.y1, image_h - 1))
        x2 = max(0, min(det.bbox.x2, image_w))
        y2 = max(0, min(det.bbox.y2, image_h))
        if x2 <= x1 or y2 <= y1:
            continue
        clipped.append(replace(det, bbox=BBox2D(x1=x1, y1=y1, x2=x2, y2=y2)))
    return clipped


def filter_by_score(dets: list[Detection2D], score_thresh: float) -> list[Detection2D]:
    return [det for det in dets if det.score >= float(score_thresh)]


def classwise_nms(dets: list[Detection2D], iou_thresh: float) -> list[Detection2D]:
    kept: list[Detection2D] = []
    by_label: dict[str, list[Detection2D]] = defaultdict(list)
    for det in dets:
        by_label[det.label].append(det)

    for label_dets in by_label.values():
        label_dets.sort(key=lambda det: det.score, reverse=True)
        while label_dets:
            current = label_dets.pop(0)
            kept.append(current)
            label_dets = [det for det in label_dets if _iou(current.bbox, det.bbox) < float(iou_thresh)]
    return kept


def keep_topk_per_label(dets: list[Detection2D], k: int = 3) -> list[Detection2D]:
    by_label: dict[str, list[Detection2D]] = defaultdict(list)
    for det in dets:
        by_label[det.label].append(det)

    kept: list[Detection2D] = []
    for label_dets in by_label.values():
        label_dets.sort(key=lambda det: det.score, reverse=True)
        kept.extend(label_dets[: max(0, int(k))])
    return kept


def select_best_detection(dets: list[Detection2D], expected_label: str) -> Detection2D | None:
    expected = expected_label.strip().lower()
    matches = [
        det
        for det in dets
        if det.label.strip().lower() == expected
        or str(det.extras.get("category", det.label)).strip().lower() == expected
    ]
    if not matches:
        return None
    matches.sort(key=lambda det: det.score, reverse=True)
    return matches[0]
