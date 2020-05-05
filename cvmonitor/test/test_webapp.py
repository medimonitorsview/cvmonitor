import base64
import os

import imageio
import numpy as np
from flask import url_for
from pylab import imshow, show  # noqa F401


def test_ping(client):

    res = client.get(url_for("ping"))
    assert res.data == b"pong"


def test_cv_ping(client):
    res = client.get(url_for("cv.ping"))
    assert res.data == b"pong cv"


def test_codes(client):
    image = open(os.path.dirname(__file__) + "/data/barcode.png", "rb").read()
    assert len(image) > 0
    res = client.post(
        url_for("cv.detect_codes"),
        data=image,
        headers={"content-type": "application/png"},
    )
    assert res.json[0]["data"] == "Foramenifera"


def test_qr_image(client):
    qr_res = client.get(
        url_for("cv.qr_display", monitorId='zoo-monitor'),
    )
    data = qr_res.data
    res = client.post(
        url_for("cv.detect_codes"),
        data=data,
        headers={"content-type": "application/png"},
    )
    assert res.json[0]["data"] == "zoo-monitor"


def test_align(client):
    image = open(os.path.dirname(__file__) + "/data/qrcode.png", "rb").read()
    assert len(image) > 0
    os.environ["CVMONITOR_QR_PREFIX"] = "http"
    res = client.post(
        url_for("cv.align_image"),
        data=image,
        headers={"content-type": "application/png"},
    )
    res_image = np.asarray(imageio.imread(res.data))
    assert res_image.shape[0] > 0


def test_align_whole_image(client):
    image = open(os.path.dirname(__file__) + "/data/barcode_monitor.jpg", "rb").read()
    assert len(image) > 0
    os.environ["CVMONITOR_QR_PREFIX"] = "http"
    res = client.post(
        url_for("cv.align_image"),
        data=image,
        headers={"content-type": "application/png"},
    )
    res_image = np.asarray(imageio.imread(res.data))
    assert res_image.shape[0] > 0


def test_exif_align(client):
    src_image = open(os.path.dirname(__file__) + "/data/sample.jpeg", "rb").read()
    assert len(src_image) > 0
    os.environ["CVMONITOR_QR_PREFIX"] = ""
    res = client.post(
        url_for("cv.align_image"),
        data=src_image,
        headers={"content-type": "application/png"},
    )
    res_image = np.asarray(imageio.imread(res.data))
    up_image = imageio.imread(os.path.dirname(__file__) + "/data/sample_up.jpg")
    assert np.median(np.abs(res_image - up_image)) < 2.0


def test_get_measurements(client):
    assert "HR" in client.get(url_for("cv.get_measurements", device="monitor")).json
    assert not client.get(url_for("cv.get_measurements", device="unknown")).json
    assert (
        "Rate" in client.get(url_for("cv.get_measurements", device="respirator")).json
    )


def test_client_ocr(client):
    image = open(os.path.dirname(__file__) + "/data/11.jpg", "rb").read()
    bbox_list = np.load(open(os.path.dirname(__file__) + "/data/11_recs.npy", "rb"))[:-1]
    image_buffer = base64.encodebytes(image).decode()
    os.environ["CVMONITOR_SERVER_OCR"] = "FALSE"
    segments = []
    devices_names = ["HR", "RR", "SpO2", "IBP"]
    values = [" 52", "(15)", "93a", "115 /45"]
    for i, b in enumerate(bbox_list):
        segments.append(
            {
                "left": int(b[0]),
                "top": int(b[1]),
                "right": int(b[2]),
                "bottom": int(b[3]),
                "name": str(devices_names[i]),
                "value": values[i]
            }
        )
        segments.append({
            "left": 100,
            "top": 100,
            "right": 200,
            "bottom": 200,
            "name": None,
            "value": None
        })
        segments.append({
            "left": 100,
            "top": 100,
            "right": 200,
            "bottom": 200,
            "name": None,
            "value": "123"
        })
    data = {"image": image_buffer, "segments": segments}
    res = client.post(url_for("cv.run_ocr"), json=data)
    results = res.json
    print(res.json)
    assert res.json
    assert len(results) == 5
    for r, e in zip(
        results,
        [
            {"name": devices_names[0], "value": "52"},
            {"name": devices_names[1], "value": "15"},
            {"name": devices_names[2], "value": "93"},
            {"name": "IBP-Systole", "value": "115"},
            {"name": "IBP-Diastole", "value": "45"},
        ],
    ):
        assert e.items() <= r.items()
