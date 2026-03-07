import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm


def gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def brightness_scale(image, factor):
    img = image.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)


def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def high_contrast(image, factor=1.5):
    img = image.astype(np.float32)
    mean = np.mean(img)
    img = (img - mean) * factor + mean
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_perturbation(image, perturbation):
    
    if perturbation == "gaussian_10":
        return gaussian_noise(image, 10)

    if perturbation == "gaussian_20":
        return gaussian_noise(image, 20)

    if perturbation == "gaussian_30":
        return gaussian_noise(image, 30)

    if perturbation == "overexposure_40":
        return brightness_scale(image, 1.4)

    if perturbation == "overexposure_60":
        return brightness_scale(image, 1.6)

    if perturbation == "underexposure_40":
        return brightness_scale(image, 0.6)

    if perturbation == "underexposure_60":
        return brightness_scale(image, 0.4)

    if perturbation == "gamma_05":
        return gamma_correction(image, 0.5)

    if perturbation == "gamma_15":
        return gamma_correction(image, 1.5)

    if perturbation == "high_contrast":
        return high_contrast(image, 1.5)

    return image


def generate_perturbed_datasets(test_set_path, output_path):

    image_dir = os.path.join(test_set_path, "image")
    mask_dir = os.path.join(test_set_path, "mask")

    perturbations = [
        "gaussian_10",
        "gaussian_20",
        "gaussian_30",
        "overexposure_40",
        "overexposure_60",
        "underexposure_40",
        "underexposure_60",
        "gamma_05",
        "gamma_15",
        "high_contrast"
    ]

    image_files = sorted(os.listdir(image_dir))

    for perturb in perturbations:

        out_img = os.path.join(output_path, perturb, "image")
        out_mask = os.path.join(output_path, perturb, "mask")

        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_mask, exist_ok=True)

        print(f"\nProcessing {perturb}...")

        for img_name in tqdm(image_files):

            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)

            image = cv2.imread(img_path)

            perturbed = apply_perturbation(image, perturb)

            cv2.imwrite(os.path.join(out_img, img_name), perturbed)

            if os.path.exists(mask_path):
                shutil.copy(mask_path, os.path.join(out_mask, img_name))


if __name__ == "__main__":

    test_set_path = "test"
    output_path = "perturbed_test_sets"

    generate_perturbed_datasets(test_set_path, output_path)

    print("\nAll perturbation datasets generated successfully.")