import json
import os
import cv2
import numpy as np
from tqdm import tqdm


def generate_masks(dataset_folder, coco_json_name):

    coco_path = os.path.join(dataset_folder, coco_json_name)
    mask_folder = os.path.join(dataset_folder, "masks")

    os.makedirs(mask_folder, exist_ok=True)

    with open(coco_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    annotations_per_image = {}

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        annotations_per_image.setdefault(img_id, []).append(ann)

    for img_id, img_info in tqdm(images.items()):

        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        mask = np.zeros((height, width), dtype=np.uint8)

        anns = annotations_per_image.get(img_id, [])

        for ann in anns:

            segmentation = ann.get("segmentation", [])
            bbox = ann.get("bbox", None)

            # ---------- POLYGON ----------
            if isinstance(segmentation, list) and len(segmentation) > 0:

                for poly in segmentation:
                    pts = np.array(poly).reshape((-1, 2))
                    pts = pts.astype(np.int32)

                    cv2.fillPoly(mask, [pts], 255)

            # ---------- BBOX -> ELLIPSE ----------
            elif bbox is not None:

                x, y, w, h = bbox

                center = (int(x + w/2), int(y + h/2))
                axes = (int(w/2), int(h/2))

                cv2.ellipse(
                    mask,
                    center,
                    axes,
                    angle=0,
                    startAngle=0,
                    endAngle=360,
                    color=255,
                    thickness=-1
                )

        name_no_ext = os.path.splitext(file_name)[0]
        mask_path = os.path.join(mask_folder, name_no_ext + ".png")

        cv2.imwrite(mask_path, mask)


if __name__ == "__main__":

    dataset_folder = "train"
    coco_json_name = "_annotations.coco.json"

    generate_masks(dataset_folder, coco_json_name)

    print("All masks generated successfully.")