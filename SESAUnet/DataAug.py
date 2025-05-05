import cv2
import numpy as np
import os

# Data Folders
image_folder = "/train/image"
mask_folder = "train/mask"
output_image_folder = "/aug/img"
output_mask_folder = "/aug/mask"


os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Augmentation Parameters
gaussian_blur_kernel = (5, 5)
illumination_change = (-25, 25)  # illumination Variation
zoom_in_factor = 1.2
zoom_out_factor = 0.8


def apply_augmentation(image, mask, technique, filename):
    augmented_image, augmented_mask = image.copy(), mask.copy()

    if technique == "gaussian_blur":
        augmented_image = cv2.GaussianBlur(image, gaussian_blur_kernel, 0)

    elif technique == "illumination":
        change = np.random.uniform(*illumination_change)
        augmented_image = cv2.convertScaleAbs(image, alpha=1, beta=change)

    elif technique == "zoom_in":
        height, width = image.shape[:2]
        zoomed_image = cv2.resize(image, None, fx=zoom_in_factor, fy=zoom_in_factor, interpolation=cv2.INTER_LINEAR)
        zoomed_mask = cv2.resize(mask, None, fx=zoom_in_factor, fy=zoom_in_factor, interpolation=cv2.INTER_NEAREST)
        augmented_image = zoomed_image[
            int((zoomed_image.shape[0] - height) / 2): int((zoomed_image.shape[0] - height) / 2) + height,
            int((zoomed_image.shape[1] - width) / 2): int((zoomed_image.shape[1] - width) / 2) + width
        ]
        augmented_mask = zoomed_mask[
            int((zoomed_mask.shape[0] - height) / 2): int((zoomed_mask.shape[0] - height) / 2) + height,
            int((zoomed_mask.shape[1] - width) / 2): int((zoomed_mask.shape[1] - width) / 2) + width
        ]

    elif technique == "zoom_out":
        height, width = image.shape[:2]
        canvas_image = np.zeros_like(image)
        canvas_mask = np.zeros_like(mask)
        zoomed_image = cv2.resize(image, None, fx=zoom_out_factor, fy=zoom_out_factor, interpolation=cv2.INTER_LINEAR)
        zoomed_mask = cv2.resize(mask, None, fx=zoom_out_factor, fy=zoom_out_factor, interpolation=cv2.INTER_NEAREST)
        start_x = (width - zoomed_image.shape[1]) // 2
        start_y = (height - zoomed_image.shape[0]) // 2
        canvas_image[start_y:start_y + zoomed_image.shape[0], start_x:start_x + zoomed_image.shape[1]] = zoomed_image
        canvas_mask[start_y:start_y + zoomed_mask.shape[0], start_x:start_x + zoomed_mask.shape[1]] = zoomed_mask
        augmented_image, augmented_mask = canvas_image, canvas_mask

    # results save
    cv2.imwrite(os.path.join(output_image_folder, f"{filename}.jpg"), augmented_image)
    cv2.imwrite(os.path.join(output_mask_folder, f"{filename}.jpg"), augmented_mask)

# Loop 
techniques = ["gaussian_blur", "illumination", "zoom_in", "zoom_out"]
technique_count = len(techniques)
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])


start_index = 1051

for i, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    mask_path = os.path.join(mask_folder, image_file)

    
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Masque en niveaux de gris

   
    technique = techniques[i % technique_count]
    apply_augmentation(image, mask, technique, start_index)

    start_index += 1

print("Augmentation terminée avec succès!")
