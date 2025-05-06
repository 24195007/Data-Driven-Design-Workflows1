import os
from PIL import Image

def convert_webp_to_jpg(folder_path, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join("converted_jpg")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".webp"):
            webp_path = os.path.join(folder_path, filename)
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, jpg_filename)

            try:
                with Image.open(webp_path) as img:
                    rgb_img = img.convert("RGB")  # 防止透明通道出错
                    rgb_img.save(jpg_path, "JPEG")
                    print(f"✅ Converted: {filename} → {jpg_filename}")
            except Exception as e:
                print(f"❌ Failed to convert {filename}: {e}")

# 示例使用：将当前目录下的 webp 图转为 jpg
if __name__ == "__main__":
    source_folder = "archdaily_images"  # 修改为你的路径
    convert_webp_to_jpg(source_folder)
