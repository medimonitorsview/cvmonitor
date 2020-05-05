import base64
import copy
import datetime
import io
import logging
import os
import random
import time

import cv2
import imageio
import numpy as np
import qrcode
import ujson as json
from flask import Blueprint, abort, request
from pylab import imshow, show  # noqa F401

from .image_align import align_by_qrcode, get_oriented_image
from .qr import generate_pdf, read_codes
from .utils import draw_segments
from .device_fields import Cleaner, get_fields_info

np.set_printoptions(precision=3)


class ResultLogger():

    def __init__(self):
        self.basedir = './log/'
        self.index = 0

    def log_ocr(self, image, segments, server_ocr, imageId=None, monitorId=None):
        self.index += 1
        folder_name = self.basedir + time.strftime("%Y_%m_%d_%H")
        os.makedirs(folder_name, exist_ok=True)
        fname = f'{folder_name}/{self.index:09}'
        if imageId and monitorId:
            fname = f'{folder_name}/{monitorId}_{imageId}'
        json.dump({'segments': segments, 'server_ocr': server_ocr}, open(f'{fname}.json', 'w'))
        if image is not None:
            with open(f'{fname}.jpg', 'wb') as f:
                f.write(image)


class ComputerVision:
    def __init__(self):
        self.blueprint = Blueprint("cv", __name__)
        self.qrDecoder = cv2.QRCodeDetector()
        self.devices = get_fields_info()
        self.cleaner = Cleaner(get_fields_info())
        self.resultsLogger = ResultLogger()

        @self.blueprint.route("/ping/")
        def ping():
            return "pong cv"

        @self.blueprint.route("/detect_codes", methods=["POST"])
        def detect_codes():
            """
            Get QR or barcodes in an image
            ---
            description: Get QR codes and barcodes in an image
            requestBody:
                content:
                    image/png:
                      schema:
                        type: string
                        format: binary
            responses:
             '200':
                dsecription: array of detections
                content:
                    application/json:
                        schema:
                            type: array
                            items:
                                type: object
                                properties:
                                    data:
                                        type: string
                                    top:
                                        type: number
                                    left:
                                        type: number
                                    bottom:
                                        type: number
                                    right:
                                        type: number
            """
            image = np.asarray(imageio.imread(request.data))
            codes = read_codes(image)
            return json.dumps(codes), 200, {"content-type": "application/json"}

        @self.blueprint.route("/align_image", methods=["POST"])
        def align_image():
            """
            Given a jpeg image with that containes the  QR code, use that QR code to align the image
            ---
            description: Gets a jpeg and returns a jpeg
            requestBody:
                content:
                    image/png:
                      schema:
                        type: string
                        format: binary
            responses:
              '200':
                descritption: jpeg image
                content:
                    image/png:
                      schema:
                        type: string
                        format: binary

            """
            use_exif = os.environ.get("CVMONITOR_ORIENT_BY_EXIF", "TRUE") == "TRUE"
            use_qr = os.environ.get("CVMONITOR_ORIENT_BY_QR", "FALSE") == "TRUE"
            qrprefix = str(os.environ.get("CVMONITOR_QR_PREFIX", "cvmonitor"))
            qrsize = int(os.environ.get("CVMONITOR_QR_TARGET_SIZE", 100))
            boundery = float(os.environ.get("CVMONITOR_QR_BOUNDERY_SIZE", 50))
            align_image_by_qr = (
                os.environ.get("CVMONITOR_SKIP_ALIGN", "TRUE") == "FALSE"
            )
            save_before_align = os.environ.get("CVMONITOR_SAVE_BEFORE_ALIGN") == "TRUE"
            save_after_align = os.environ.get("CVMONITOR_SAVE_AFTER_ALIGN") == "TRUE"

            imdata = io.BytesIO(request.data)
            image, detected_qrcode, _ = get_oriented_image(
                imdata, use_exif=use_exif, use_qr=use_qr
            )

            headers = {"content-type": "image/jpeg"}
            if save_before_align:
                imdata.seek(0)
                with open("original_image.jpg", "wb") as f:
                    f.write(imdata)

            if detected_qrcode is None:
                if align_image_by_qr:
                    abort(
                        400,
                        "Could not align the image by qr code, no such code detected",
                    )
            else:
                headers["X-MONITOR-ID"] = detected_qrcode.data.decode()

            if align_image_by_qr:
                logging.debug("Trying to align image by qr code")
                image, _ = align_by_qrcode(
                    image, detected_qrcode, qrsize, boundery, qrprefix
                )

            if save_after_align:
                imageio.imwrite("aligned_image.jpg", image)

            b = io.BytesIO()
            imageio.imwrite(b, image, format="jpeg")
            b.seek(0)
            return b.read(), 200, headers

        @self.blueprint.route("/run_ocr", methods=["POST"])
        def run_ocr():
            """
            Process ocr results
            ---
            description: Process ocr results
            requestBody:
                content:
                    application/json:
                      schema:
                        type: object
                        properties:
                            image:
                                type: string
                                contentEncoding: base64
                                contentMediaType: image/jpeg
                            segments:
                                type: array
                                items:
                                    type: object
                                    properties:
                                        top:
                                            type: number
                                            format: integer
                                        left:
                                            type: number
                                            format: integer
                                        bottom:
                                            type: number
                                            format: integer
                                        right:
                                            type: number
                                            format: integer
            responses:
              '200':
                description: ocr results
                content:
                    application/json:
                      schema:
                        type: array
                        items:
                            type: object
                            properties:
                                segment_name:
                                    type: string
                                value:
                                    type: string
            """
            data = request.json
            monitorId = data.get("monitorId")
            imageId = data.get("imageId")
            logging.debug(f'id: {data.get("monitorId")}:{data.get("imageId")}')

            segments = copy.deepcopy(data.get("segments", []))
            logging.debug(f'Recived segments {segments}')
            cleaned_segments = self.cleaner.clean_segments(copy.deepcopy(segments), monitorId, imageId)
            logging.debug(f'Cleaned segments {cleaned_segments}')
            try:
                imageId = int(imageId)
            except Exception:
                imageId = None
            # Randomly log some images:
            rnum = random.randint(0, 10)
            if (imageId and ((imageId % 20) == 0 or (imageId % 20) == 1)) or rnum == 5:
                if 'image' in data:
                    del data['image']
                self.resultsLogger.log_ocr(None, None, {'cleaned': cleaned_segments,
                                                        "process_time":  datetime.datetime.isoformat(datetime.datetime.now())}, imageId, monitorId)
            return json.dumps(cleaned_segments), 200, {"content-type": "application/json"}

        @self.blueprint.route("/show_ocr/", methods=["POST"])
        def show_ocr():
            """
            Run ocr on an image
            ---
            description: run ocr on image
            requestBody:
                content:
                    application/json:
                      schema:
                        type: object
                        properties:
                            image:
                                type: string
                                contentEncoding: base64
                                contentMediaType: image/jpeg
                            segments:
                                type: array
                                items:
                                    type: object
                                    properties:
                                        top:
                                            type: number
                                            format: integer
                                        left:
                                            type: number
                                            format: integer
                                        bottom:
                                            type: number
                                            format: integer
                                        right:
                                            type: number
                                            format: integer
            responses:
              '200':
                description: ocr results
                content:
                    application/json:
                      schema:
                        type: array
                        items:
                            type: object
                            properties:
                                segment_name:
                                    type: string
                                value:
                                    type: string
            """
            data = request.json
            assert "image" in data
            image = np.asarray(
                imageio.imread(base64.decodebytes(data["image"].encode()))
            )
            # Suggest segments
            if data.get("segments"):
                image = draw_segments(image, data.get('segments'))

            headers = {"content-type": "image/jpeg"}
            b = io.BytesIO()
            imageio.imwrite(b, image, format="jpeg")
            b.seek(0)
            return b.read(), 200, headers

        @self.blueprint.route("/qr/<title>", methods=["GET"])
        def qr(title):
            """
            Generate pdf of qr codes, after the /qr/ put title for
            each qr.
            The data in the qr code will be cvmonitor-title-16_random_characters
            ---
            description: get pdf of qr codes
            get:
            parameters:
            - in: path
              name: title
              schema:
                  type: string
                  required: true
                  default: cvmonitor
            - in: query
              name: width
              schema:
                  type: number
                  required: false
            - in: query
              name: height
              schema:
                  type: number
                  required: false
            responses:
              '200':
                descritption: pdf of results
                content:
                    application/pdf:
            """
            try:
                width = int(request.args.get("width"))
            except Exception:
                width = None
            try:
                height = int(request.args.get("height"))
            except Exception:
                height = None

            headers = {
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="random-qr.pdf"',
            }
            pdf_buffer = io.BytesIO()
            generate_pdf(pdf_buffer, title, width, height)
            pdf_buffer.seek(0)
            return pdf_buffer.read(), 200, headers

        @self.blueprint.route("/measurements/<device>", methods=["GET"])
        def get_measurements(device):
            return json.dumps([x for x in get_fields_info([device]).keys()]), 200, {'content-type': 'application/json'}

        @self.blueprint.route("/qr_display/<monitorId>", methods=["GET"])
        def qr_display(monitorId):
            """
            Generate image of qr
            ---
            description: get pdf of qr codes
            get:
            parameters:
            - in: path
              name: monitorId
              schema:
                  type: string
                  required: true
                  default: cvmonitor
            responses:
              '200':
                descritption: qr image
                content:
                    image/png:
            """
            headers = {
                "Content-Type": "image/png",
            }
            buffer = io.BytesIO()
            qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
            qr.add_data(monitorId)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            image = np.array(img).astype(np.uint8)*255
            imageio.imwrite(buffer, image, format='png')
            buffer.seek(0)
            return buffer.read(), 200, headers
