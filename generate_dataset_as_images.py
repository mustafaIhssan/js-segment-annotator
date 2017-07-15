import numpy as np
from PIL import Image

# Decode
encoded = np.array(Image.open('data/annotations/1.png'))
annotation = np.bitwise_or(np.bitwise_or(
    encoded[:, :, 0].astype(np.uint32),
    encoded[:, :, 1].astype(np.uint32) << 8),
    encoded[:, :, 2].astype(np.uint32) << 16)

print(np.unique(annotation))

# Encode
Image.fromarray(np.stack([
    np.bitwise_and(annotation, 255),
    np.bitwise_and(annotation >> 8, 255),
    np.bitwise_and(annotation >> 16, 255),
    ], axis=2).astype(np.uint8)).save('encoded.png')