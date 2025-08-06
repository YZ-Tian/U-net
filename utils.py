from PIL import Image

#每张图片尺寸不一样，需要统一为正方形尺寸
# 1. 对每张图像，计算其宽和高的最大值（最长边）
# 2. 创建一个边长为最长边的正方形画布（背景填充指定值）
# 3. 将原图居中贴到正方形画布上（保持原图大小）
# 4. 将正方形图像缩放到模型所需的输入尺寸（如 256×256）
def keep_image_size_open(path,size=(256,256)):
    img=Image.open(path)
    temp=max(img.size)
    mask=Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img)   # 把图片粘到mask的左上角
    mask=mask.resize(size)
    return mask
    