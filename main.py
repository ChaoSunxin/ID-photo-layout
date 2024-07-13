import os
from enum import Enum

import cv2
import numpy as np
from PIL import Image
from rembg import remove


def resize_and_crop(image: Image, target_width, target_height):
    """Resize and crop an image to fit the desired dimensions."""
    img_width, img_height = image.size
    img_ratio = img_width / img_height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        new_height = target_height
        new_width = int(new_height * img_ratio)
    else:
        new_width = target_width
        new_height = int(new_width / img_ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image


def auto_crop(image):
    """Automatically crop image to fit face and shoulders within a specified ratio."""
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        raise ValueError("No faces detected!")
    (x, y, w, h) = faces[0]
    margin_h = int(0.6 * h)
    margin_w = int(0.4 * w)
    # Margin on top as 30% of face height
    # Margin on sides as 20% of face width
    bottom_margin = int(0.5 * h)  # Add extra space equivalent to half of face height
    # Correcting the margins to stay within the image boundaries
    start_x = max(0, x - margin_w)

    start_y = max(0, y - margin_h)
    end_x = min(open_cv_image.shape[1], x + w + margin_w)
    end_y = min(open_cv_image.shape[0], y + h + bottom_margin)
    # # Draw rectangle around the face (shoulders included)
    # cv2. rectangle(open_cv_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2) # Red color rectangli
    # # Save the image with the rectangle
    # marked_image = cv2. cvtColor (open_cv_1mage, cv2. COLOR_BGR2RGB)
    # marked_pil_image = Image. fromarray (marked_image)
    # marked_pil_image. save ("marked_image.jpg")
    # Crop the image
    cropped_image = open_cv_image[start_y:end_y, start_x:end_x]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_image)


def replace_bg(image: Image, bg_image: Image = None, bg_color: str = "#fff"):
    """Replace background of the image with a bg_image or bg_color"""

    # alpha_matting=true 蒙板扣图
    fg_image = remove(image, alpha_natting=False)

    if bg_image:
        width, height = fg_image.size
        bg_image = bg_image.resize((width, height), Image.Resampling.LANCZOS)
        bg_image = bg_image.convert("RGBA")
        combined = Image.alpha_composite(bg_image, fg_image).convert("RGB")
    elif bg_color:
        width, height = fg_image.size
        bg_image = Image.new("RGBA", (width, height), bg_color)
        combined = Image.alpha_composite(bg_image, fg_image).convert("RGB")
    else:
        raise ValueError("Either bg_image or bg_color must be provided.")
    return combined


def auto_rmebg_crop(photo_path, bg_color="#fff"):
    # 自动扣图并更换背景
    result_image = replace_bg(image=Image.open(photo_path), bg_image=None, bg_color=bg_color)
    # 自动识别头部并进行裁剪
    try:
        result_image = auto_crop(result_image)
    except ValueError as eroor:
        print("file %s, error%s", (photo_path, eroor))
        return
    return result_image


def inch_layout(image: Image, photo_size: tuple[int], page_size: tuple[int]):
    photo_width, photo_height = photo_size
    page_width, page_height = page_size
    # 创建一个新的白色背景的图片
    output_image = Image.new("RGB", page_size, "white")

    # 调整大小和裁剪
    crop_img = resize_and_crop(image, photo_width, photo_height)

    # 计算每行和列可以容纳的照片数量
    photos_per_row = page_width // (photo_width)
    photos_per_column = page_height // (photo_height)

    gap_x = min((page_width - photo_width * photos_per_column) // photos_per_column, 2)

    offset_x = (page_width - (photo_width + gap_x) * (photos_per_column)) // 2
    offset_y = (page_height - (photo_height + gap_x) * photos_per_row) // 2

    # 将证件照放置到背景图片中
    for row in range(photos_per_column):
        for col in range(photos_per_row):
            x = col * (photo_width + gap_x) + offset_x
            y = row * (photo_height + gap_x) + offset_y
            output_image.paste(crop_img, (x, y))
    return output_image


# 像素宽度 = 英寸宽度 × DPI
# 1 英寸 = 2.54 厘米。

DPI = 300
inch_of_cm = 2.54


def cm_to_inches(cm):
    return cm / inch_of_cm


def inch_to_pixel(cm, dpi):
    return int(cm_to_inches(cm) * dpi)


class InchToPixel(Enum):
    ONE_INCH = (inch_to_pixel(2.5, DPI), inch_to_pixel(3.5, DPI))
    SMALL_ONE_INCH = (inch_to_pixel(2.5, DPI), inch_to_pixel(3.3, DPI))
    SMALL_TWO_INCH = (inch_to_pixel(4.8, DPI), inch_to_pixel(3.3, DPI))
    TWO_INCH = (inch_to_pixel(3.5, DPI), inch_to_pixel(5.3, DPI))
    THREE_INCH = (inch_to_pixel(6.2, DPI), inch_to_pixel(8.9, DPI))
    FIVE_INCH = (inch_to_pixel(8.9, DPI), inch_to_pixel(12.7, DPI))
    SIX_INCH = (inch_to_pixel(10.2, DPI), inch_to_pixel(15.2, DPI))

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height


if __name__ == "__main__":
    img = auto_rmebg_crop(os.path.join(os.path.dirname(__file__), "training-originals/0009_007.jpg"), "#438edb")
    inch_layout_img = inch_layout(img, InchToPixel.ONE_INCH.value, InchToPixel.SIX_INCH.value)
    inch_layout_img.save(os.path.join(os.path.dirname(__file__), "test-output/0009_007.jpg"))
