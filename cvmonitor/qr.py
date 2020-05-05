import os
from uuid import uuid4

import cv2
import matplotlib.pyplot as plt
import numpy as np
import qrcode
from matplotlib.backends.backend_pdf import PdfPages
from pylab import imshow, show # noqa F401
from pyzbar import pyzbar

np.set_printoptions(precision=3)


def generate_pdf(pdf_file, title, ncols, nrows):
    if not nrows:
        nrows = int(os.environ.get("CVMONITOR_QR_PDF_ROWS", 6))
    if not ncols:
        ncols = int(os.environ.get("CVMONITOR_QR_PDF_COLS", 4))
    with PdfPages(pdf_file) as pdf:
        index = 0
        fig, axarr = plt.subplots(nrows, ncols, figsize=[8, 11])
        fig.tight_layout(pad=4, h_pad=3, w_pad=2)
        for y in range(nrows):
            for x in range(ncols):
                index += 1
                qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
                uuid = uuid4().hex
                text = f"cvmonitors-{title}-{uuid}"
                qr.add_data(text)
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                axarr[y, x].set_title(f"{title}\n{uuid[:16]}", fontsize=8)
                axarr[y, x].set_xticks([])
                axarr[y, x].set_yticks([])
                axarr[y, x].set_yticklabels([])
                axarr[y, x].set_xticklabels([])
                axarr[y, x].imshow(img, cmap="gray")

        pdf.savefig(fig)


def read_codes(image):
    """
    read barcodes in an image
    """
    decodedObjects = pyzbar.decode(image)
    codes = []
    for obj in decodedObjects:
        try:
            codes.append(
                {
                    "data": obj.data.decode(),
                    "top": obj.rect.top,
                    "left": obj.rect.left,
                    "bottom": obj.rect.top + obj.rect.height,
                    "right": obj.rect.left + obj.rect.width,
                    "type": obj.type,
                }
            )
        except Exception:
            continue
    return codes


def find_qrcode(image, prefix):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    decodedObjects = pyzbar.decode(image)
    if not decodedObjects:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
        image = clahe.apply(image)
        decodedObjects = pyzbar.decode(image)

    detected_qrcode = None
    for obj in decodedObjects:
        text = obj.data.decode()
        if text.startswith(prefix):
            detected_qrcode = obj
            break
    return detected_qrcode
