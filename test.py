from PIL import Image, ImageSequence
Image.MAX_IMAGE_PIXELS = None


filepath = "/Users/sam/Downloads/COOK 1.tif"

img = Image.open(filepath)
pages = [frame.copy() for frame in ImageSequence.Iterator(img)]
smallest_layer_pil = pages[-1]

print("Smallest layer size (PIL):", smallest_layer_pil.size)
print("Number of available layers:", len(pages))