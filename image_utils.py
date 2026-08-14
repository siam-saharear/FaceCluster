import os
import cv2


def scrape_image_paths(*args):
    folder_path = ""
    for arg in args:
        folder_path = os.path.join(folder_path, arg)
    files = os.listdir(folder_path)
    image_paths = []
    for file in files:
        file_extention = str((file.split("/")[-1]).split(".")[-1])
        if file_extention.lower() == "png" or file_extention.lower() == "jpg":
            image_paths.append(os.path.join(folder_path, file))
    return image_paths


def image_resizer(image, new_h=300):
    h,w = image.shape[:2]
    aspect_ratio = w/h
    new_w = int(aspect_ratio*new_h)
    resized = cv2.resize(image, (new_w, new_h),)
    return resized
