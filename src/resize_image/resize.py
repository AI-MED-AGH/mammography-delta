# Resize images with padding
from PIL import Image, ImageOps

def resize_with_padding(img_path, target_size):
    img = Image.open(img_path).convert('RGB')
    result = ImageOps.pad(img, (target_size, target_size), method=Image.Resampling.LANCZOS, color='black', centering=(0.5, 0.5))
    return result

# res = resize_with_padding("../../images/1472.png", 224)
# res.show()
# res = np.array(res)
# res.shape