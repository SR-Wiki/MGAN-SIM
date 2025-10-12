import numpy as np
from PIL import Image
import tifffile

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])


def load_img(filepath):
    img = Image.open(filepath).convert('L')
    img = img.resize((512, 512), Image.BICUBIC)
    if np.random.random() < 0.5:
        img = np.fliplr(img)

    if np.random.random() > 0.5:
        img = np.flipud(img)

    ran = np.random.random()
    if ran < 0.5:
        img = np.rot90(img, 1)

    if ran > 0.5:
        img = np.rot90(img, -1)

    return img

def load_img1(filepath):

    img = Image.open(filepath).convert('RGB')
    #img = img.resize((1536,1664), Image.BICUBIC)
    return img
def load_predictimg():
    for i in range(1, 10):
        batch_image = 'datasets/SIM147_CCP128/predict'
        img = tifffile.imread('%s/%d.tif' % (batch_image, i))

        imgs_Bs = []
        img_res = [256, 512]
        imsize = (1, img_res[0], img_res[1], 1)
        imgs_B = np.zeros(imsize)

        img_B = scipy.misc.imresize(img, img_res)
        imgs_Bs.append(img_B)
        imgs_Bs = np.array(imgs_Bs) / 255.
        imgs_B[:, :, :, 0] = imgs_Bs

        d, fake_A = gan.predict(imgs_B)
        save_ = 255 * np.array(fake_A)
    return save_
def save_img(image_tensor: object, filename: object) -> object:
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))
