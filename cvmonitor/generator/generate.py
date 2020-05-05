#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import copy
import datetime
import io
import os
import pickle
import random
import sys
import time
import traceback
import uuid
from typing import List

import cv2
import imageio
import matplotlib
import numpy as np
import pylab
import qrcode
import requests
from matplotlib import pyplot as plt
from pylab import imshow, show  # noqa F401
from pyzbar import pyzbar

from cvmonitor import cv
from cvmonitor.device_fields import (
    get_field_rand_value,
    get_fields_info
)

QRSIZE = 100

SEND_TO_SERVER = False
name_list = [
    "בנימין נתניהו",
    "יולי אדלשטיין",
    'ישראל כ"ץ',
    "גלעד ארדן",
    "גדעון סער",
    "מירי רגב",
    "יריב לוין",
    "יואב גלנט",
    "ניר ברקת",
    "גילה גמליאל",
    "אבי דיכטר",
    "זאב אלקין",
    "חיים כץ",
    "אלי כהן",
    "צחי הנגבי",
    "אופיר אקונ0יס",
    "יובל שטייניץ",
    "ציפי חוטובלי",
    "דודי אמסלם",
    "גדי יברקן",
    "אמיר אוחנה",
    "אופיר כץ",
    "אתי עטייה",
    "יואב קיש",
    "דוד ביטן",
    "קרן ברק",
    "שלמה קרעי",
    "מיקי זוהר",
    "יפעת שאשא-ביטון",
    "שרן השכל",
    "מיכל שיר",
    "קטי שטרית",
    "פטין מולא",
    "מאי גולן",
    "טלי פלוסקוב",
    "עוזי דיי",
    "בני גנץ",
    "יאיר לפיד",
    "משה יעלון",
    "גבי אשכנזי",
    "אבי ניסנקורן",
    "מאיר כהן",
    "מיקי חיימוביץ'",
    "עפר שלח",
    "יועז הנדל",
    "אורנה ברביבאי",
    "מיכאל ביטון",
    "חילי טרופר",
    "צבי האוזר",
    "אורית פרקש-הכהן",
    "קארין אלהרר",
    "מירב כהן",
    "יואל רזבוזוב",
    "אסף זמיר",
    "יזהר שי",
    "אלעזר שטרן",
    "מיקי לוי",
    "עומר ינקלביץ'",
    "פנינה תמנו-שטה",
    "ר'דיר מריח",
    "רם בן ברק",
    "אלון שוסטר",
    "יואב סגלוביץ'",
    "רם שפע",
    "בועז טופורובסקי",
    "אורלי פרומן",
    "איתן גינזבורג",
    "אנדריי קוז'ינוב",
    "עידן רול",
    "איימן עודה",
    "מטאנס שחאדה",
    "אחמד טיבי",
    "מנסור עבאס",
    "עאידה תומא סלימאן",
    "וליד טאהא",
    "עופר כסיף",
    "היבא יזבק",
    "אוסאמה סעדי",
    "יוסף ג'בארין",
    "סעיד אלחרומי",
    "ג'אבר עסאקלה",
    "סאמי אבו שחאדה",
    "סונדוס סאלח",
    "אימאן ח'טיב יאסי",
    "אריה דרעי",
    "יצחק כהן",
    "משולם נהרי",
    "יעקב מרגי",
    "יואב בן צור",
    "מיכאל מלכיאלי",
    "משה ארבל",
    "ינון אזולאי",
    "משה אבוטבו",
    "אביגדור ליברמן",
    "עודד פורר",
    "יבגני סובה",
    "אלי אבידר",
    "יוליה מלינובסקי",
    "חמד עמאר",
    "אלכס קושני",
    "יעקב ליצמן",
    "משה גפני",
    "מאיר פרוש",
    "אורי מקלב",
    "יעקב טסלר",
    "יעקב אשר",
    "ישראל אייכל",
    "עמיר פרץ",
    "ניצן הורוביץ",
    "תמר זנדברג",
    "איציק שמולי",
    "מרב מיכאלי",
    "יאיר גול",
    "נפתלי בנט",
    "רפי פרץ",
    "איילת שקד",
    "בצלאל סמוטריץ'",
    "מתן כהנא",
    "אופיר סופ",
    "אורלי לוי-אבקסיס",
]

devices = get_fields_info()


def get_qr_code(title):
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
    # Seedable uuid4:
    a = "%32x" % random.getrandbits(128)
    rd = a[:12] + "4" + a[13:16] + "a" + a[17:]
    uuid4 = uuid.UUID(rd)
    text = f"cvmonitors-{title}-{uuid4.hex[:16]}"
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    qrsize = random.randint(120, 180)
    img = cv2.resize(np.array(img, dtype=np.uint8) * 255, (qrsize, qrsize))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img, text


def create_segments(
    device_type, fontScale, thickness, image_size=[1000, 1200], x_start=300, y_start=200
):
    size = (
        np.array(
            cv2.getTextSize(
                text=str("COFFE"),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=fontScale,
                thickness=thickness,
            )[0]
        )
        + 10
    )
    # size=np.array(cv2.getTextSize(text=str('CO'), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=fontScale,thickness=thickness)[0])+10
    y_step = size[1] + 50
    x_step = size[0] + 50
    x = x_start
    y = y_start
    segments = []
    fields = get_fields_info([device_type])
    for m in fields:
        if x + x_step >= image_size[1] - 50:
            x = x_start
            y += y_step
        segments.append(
            {
                "top": max(y - 20, 0),
                "left": max(x - 20, 0),
                "bottom": int(min(y + size[1], image_size[0] - 10)),
                "right": int(min(x + size[0], image_size[1] - 10)),
                "name": m,
            }
        )
        if x + x_step < image_size[1] - 50:
            x += x_step
    return segments


def fill_segments(segments, device_type):
    values = []
    colors = []
    fields = get_fields_info([device_type])
    for s in segments:
        values.append(
            {"name": s["name"], "value": get_field_rand_value(fields[s["name"]])}
        )
        colors.append(
            (random.randint(20, 255), random.randint(20, 255), random.randint(20, 255))
        )
    return values, colors


def change_values(values, device_type):
    fields = get_fields_info([device_type])
    for i in range(len(values)):
        values[i]["value"] = get_field_rand_value(
            fields[values[i]["name"]], values[i]["value"]
        )
    return values


def rotate_image(image, angle):
    new_image = np.zeros(
        [int(image.shape[0] * 1.5), int(image.shape[1] * 1.5), 3], dtype=image.dtype
    )

    new_image[
        new_image.shape[0] // 4: image.shape[0] + new_image.shape[0] // 4,
        new_image.shape[1] // 4: image.shape[1] + new_image.shape[1] // 4,
        :,
    ] = image
    shp = np.array(new_image.shape[1::-1])
    sz = tuple([int(x) for x in (shp)])
    image_center = tuple([int(x) for x in (shp // 2)])
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    result = cv2.warpAffine(new_image, rot_mat, sz, flags=cv2.INTER_LINEAR)
    return result


def generate_picture(
    qrcode, image_size, segments, values, colors, fontScale, thickness
):
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8) + 40
    image[12: qrcode.shape[0] + 12, 18: qrcode.shape[1] + 18, :] = qrcode
    for s, v, color, f, t in zip(segments, values, colors, fontScale, thickness):
        size = cv2.getTextSize(
            text=str(v["value"]),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=f,
            thickness=t,
        )
        image = cv2.putText(
            img=image,
            text=str(v["value"]),
            org=(s["left"], s["top"] + size[0][1] + 20),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=f,
            color=color,
            thickness=t,
        )
        cv2.putText(
            img=image,
            text=str(v["name"]),
            org=(s["left"], s["top"] - 10),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=0.6,
            color=(0, 0, 0),
            thickness=1,
        )
    return image


class Device:
    def __init__(self, device_type, patient, room_number):
        self.index = 0
        self.monitor_id = None
        self.device_type = device_type
        self.image_size = [1000, 1200]
        self.segments = create_segments(device_type, 7, 8, self.image_size)
        self.fontScale = [random.randint(2, 7) for _ in self.segments]
        self.thickness = [random.randint(3, 8) for _ in self.segments]
        self.draw_segments = copy.deepcopy(self.segments)
        self.values, self.colors = fill_segments(self.segments, device_type)
        qrcode, self.qrtext = get_qr_code(device_type)
        self.qrcode = rotate_image(qrcode, float(random.randint(-10, 10)))
        self.patient = patient
        self.room_number = room_number

    def picture(self):
        return generate_picture(
            self.qrcode,
            self.image_size,
            self.draw_segments,
            self.values,
            self.colors,
            self.fontScale,
            self.thickness,
        )

    def change_values(self):
        self.values = change_values(self.values, self.device_type)


def fill_rooms(device_count) -> List[Device]:
    active_devices: List[Device] = []
    names = copy.deepcopy(name_list)
    random.shuffle(names)
    for i in range(device_count):
        room = random.randint(1, 400)
        patient = names[i]
        active_devices.append(Device("ivac", patient, room))
        active_devices.append(Device("monitor", patient, room))
        active_devices.append(Device("respirator", patient, room))
    return active_devices


def order_points(pts):
    """
    order such that [top-left,top-right,bottom-right,bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_qrcode(image, prefix):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    decodedObjects = pyzbar.decode(image)
    if decodedObjects is None:
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


def update_segments(image, segments, qrprefix="cvmonitor"):
    detected_qrcode = find_qrcode(image, qrprefix)
    if detected_qrcode is None:
        raise RuntimeError("Could not detect QR")
    warped, M = cv.align_by_qrcode(image, detected_qrcode, qrsize=QRSIZE, boundery=50)
    segments = copy.deepcopy(segments)
    for i, s in enumerate(segments):
        V = np.array(
            [
                [s["left"], s["top"], 1],
                [s["left"], s["bottom"], 1],
                [s["right"], s["top"], 1],
                [s["right"], s["bottom"], 1],
            ]
        )
        U = M @ V.transpose()
        for r in range(4):
            U[:, r] /= U[-1, r]
        s["left"] = int(min(U[0, 0:2]))
        s["right"] = min(int(max(U[0, 2:])), warped.shape[1])
        s["top"] = int(min(U[1, 0], U[1, 2]))
        s["bottom"] = min(int(max(U[1, 1], U[1, 3])), warped.shape[0])
    return warped, segments


def draw_segements(image, segments, colors):
    for s, c in zip(segments, colors):
        c = np.median(
            image, axis=[0, 1]
        )  # change rectangle               color as background
        image = cv2.rectangle(
            image,
            (int(s["left"]), int(s["top"])),
            (int(s["right"]), int(s["bottom"])),
            c,
            1,
        )
    return image


# model_ocr = monitor_ocr.build_model()


def send_picture(url: str, device: Device):
    image = device.picture()
    print(device.values)
    image = draw_segements(image, device.draw_segments, device.colors)

    if SEND_TO_SERVER:
        b = io.BytesIO()
        imageio.imwrite(b, image, format="jpeg")
        # imageio.imwrite(device.qrtext+f'.{device.index}.jpg', image, format='jpeg')
        b.seek(0)
        headers = {"Content-Type": "image/jpeg", "X-IMAGE-ID": str(device.index)}
        if device.monitor_id is not None:
            headers["X-MONITOR-ID"] = str(device.monitor_id)
        else:
            headers["X-MONITOR-ID"] = str(device.qrtext)
        timestamp = str(datetime.datetime.utcnow().isoformat())
        headers["X-TIMESTEMP"] = timestamp
        monitor_data = {
            "monitorId": str(device.monitor_id),
            "imageId": str(device.index),
            "timestamp": str(datetime.datetime.utcnow().isoformat()),
            "segments": copy.deepcopy(device.segments),
        }
        for s, v in zip(monitor_data["segments"], device.values):
            s["value"] = v["value"]
            s["source"] = "device"
            s["scoure"] = 1.0
            s["level"] = "basement"
        res = requests.post(
            url + f"/monitor_data/{device.monitor_id}", json=monitor_data
        )
        res = requests.post(url + f"/monitor_image", data=b, headers=headers)
        res_data = res.json()
        print(res_data)
        if "nextImageId" in res_data:
            device.index = res_data["nextImageId"]
        if "monitorId" in res_data:
            device.monitor_id = res_data["monitorId"]
    else:
        detected_qrcode = find_qrcode(image, "")
        if detected_qrcode is None:
            raise RuntimeError("Could not detect QR")
        image, M = cv.align_by_qrcode(
            image, detected_qrcode, qrsize=QRSIZE, boundery=50
        )
        # FIXME: assume that image is in RGB (not BGR). if not - should fix code in monitor_ocr.detect()
        # texts = model_ocr.ocr(expected_boxes, image, threshold=0.2)
        # print(texts)


def send_all_pictures(url, active_devices: List[Device]):
    device_indxes = list(range(len(active_devices)))
    random.shuffle(device_indxes)
    for di in device_indxes:
        device = active_devices[di]
        try:
            send_picture(url, device)
        except Exception:
            print(f"error while sending picture")
            traceback.print_exc()


def add_devices(url: str, active_devices: List[Device]):

    for di in range(len(active_devices)):
        device = active_devices[di]
        device_json = {
            "monitorId": device.qrtext,
            "patientId": device.patient,
            "imageId": 0,
            "roomId": device.room_number,
            "deviceCategory": device.device_type,
            "segments": device.segments,
        }
        requests.post(url + f"/monitor/{device.qrtext}", json=device_json)


def generate_data(url):
    if os.path.exists("devices.pkl"):
        active_devices = pickle.load(open("devices.pkl", "rb"))
    else:
        active_devices = fill_rooms(5)
        pickle.dump(active_devices, open("devices.pkl", "wb"))

    for d in active_devices:
        picture = d.picture()
        picture, new_segments = update_segments(
            picture, d.segments, qrprefix="cvmonitor"
        )
        d.segments = new_segments
        picture = draw_segements(picture, new_segments, d.colors)
        try:
            cv2.imwrite(d.qrtext + ".jpg", picture)
        except Exception:
            cv2.imwrite(d.qrtext + ".jpg", d.picture())
            print(d.qrtext)
            exit(-1)
    send_all_pictures(url, active_devices)
    if not SEND_TO_SERVER:
        exit(0)
    add_devices(url, active_devices)
    while True:
        send_all_pictures(url, active_devices)
        for d in active_devices:
            d.change_values()
        time.sleep(0.01)


def simulate_monitor(url):
    matplotlib.use("tkAgg")
    # matplotlib.use("Qt4agg")
    devices: List[Device] = fill_rooms(1)
    device = devices[random.randint(0, 2)]
    image = device.picture()
    fig = pylab.figure(figsize=[12, 10])
    plt.gca().set_title(device.qrtext)
    pylab.ioff()
    print(device.values)
    print(f"Device: {device.qrtext}")

    def press(event):
        print(event.key)
        if event.key == "q":
            exit(-1)

    fig.canvas.mpl_connect("key_press_event", press)
    while True:
        image = device.picture()
        pylab.imshow(image)
        plt.pause(0.05)
        if random.randint(0, 1) == 0:
            device.change_values()

        print(".", end="")


def delete_all(url):
    monitors = requests.get(f"{url}/monitor/list").json()
    for monitor in monitors:
        print(f"deleteing {monitor}")
        print(requests.delete(f"{url}/monitor/{monitor}").json())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--no_send", action="store_true", help="dont send to server just create images"
    )
    parser.add_argument(
        "--send", action="store_true", help="dont send to server just create images"
    )
    parser.add_argument("--sim", action="store_true", help="Simulate a device")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--url", default=None, type=str, help="Server url to use")
    parser.add_argument(
        "--delete_all", action="store_true", help="Delete all monitors from server"
    )
    args = parser.parse_args()
    if args.no_send != args.send:
        if args.no_send:
            SEND_TO_SERVER = False
        if args.send:
            SEND_TO_SERVER = True
    if not args.send and not args.no_send and not args.sim and not args.delete_all:
        print(parser.format_help())
        exit(-1)
    random.seed(args.seed)
    if args.delete_all:
        sys.exit(delete_all(args.url) or 0)
    if args.url is None:
        print("Please set url for the server")
        print(parser.format_help())
        exit(-1)
    if args.sim:
        simulate_monitor(args.url)
    else:
        generate_data(args.url)
