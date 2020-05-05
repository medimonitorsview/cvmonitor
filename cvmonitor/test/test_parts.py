import os
import time

import imageio
import numpy as np
from pylab import imshow, show  # noqa F401

from .. import image_align, qr
from ..aug_clean import MonitorValues
from ..device_fields import Cleaner, PostProcessor, get_fields_info
from ..image_align import align_by_qrcode
from ..qr import find_qrcode


def test_pdf_generate():
    qr.generate_pdf("test.pdf", "something", 4, 6)


def test_find_qrcode():
    image = imageio.imread(os.path.dirname(__file__) + "/data/barcode_monitor.jpg")
    qrcode = qr.find_qrcode(image, "")
    assert qrcode.data.decode().startswith("http")


def test_align_image1():
    image = imageio.imread(os.path.dirname(__file__) + "/data/barcode_monitor.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_image2():
    image = imageio.imread(os.path.dirname(__file__) + "/data/test.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_rotate():
    image = imageio.imread(os.path.dirname(__file__) + "/data/rotated.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_flipped():
    image = imageio.imread(os.path.dirname(__file__) + "/data/flipped.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)

    assert warpped.shape[0] > 0


def test_align_90deg_large():
    image = imageio.imread(os.path.dirname(__file__) + "/data/90_deg_rotate.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_90deg_small():
    image = imageio.imread(os.path.dirname(__file__) + "/data/90_deg_rotate_small.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_align_another():
    image = imageio.imread(os.path.dirname(__file__) + "/data/another.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert warpped.shape[0] > 0


def test_printed_qr():
    image = imageio.imread(os.path.dirname(__file__) + "/data/printed.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)

    assert warpped.shape[0] > 0


def test_generated_qr():
    image = imageio.imread(os.path.dirname(__file__) + "/data/generated_from_pdf.jpg")
    qrcode = find_qrcode(image, "")
    warpped, M = align_by_qrcode(image, qrcode)
    assert "dba7b418e0ef450c" in qrcode.data.decode()
    assert warpped.shape[0] > 0


def test_orient_by_qr():
    res = []
    rwarpped = []
    print()
    for i in [1, 2, 3, 4]:
        im_file = open(os.path.dirname(__file__) + f"/data/bad{i}.jpg", "rb")
        image, _, _ = image_align.get_oriented_image(im_file, use_qr=True)
        res.append(image)
        # Detect QR again, so it will be eaactly the same
        qrcode = find_qrcode(image, "")
        print(qrcode)
        warpped, M = align_by_qrcode(image, qrcode)
        rwarpped.append(warpped)
        # The original rotaion was lossy...
        assert np.median(res[0] - res[-1]) < 2.0


def test_cleaner_1():
    cleaner = Cleaner(get_fields_info())
    assert cleaner.clean_segments([{"name": "Medication Name", "value": "simv+"}], "aaa", "1") == {"name": "Medication Name", "value": "simv+"}


def test_cleaner_ibp():
    cleaner = Cleaner(get_fields_info())
    res = cleaner.clean_segments([{"name": "IBP", "value": "120/80"}], "aaa", "1")
    assert res == [{"name": "IBP-Systole", "value": "120"}, {"name": "IBP-Diastole", "value": "80"}]

    res = cleaner.clean_segments([{"name": "IBP", "value": "120/80"}, {"name": "IBP", "value": None}], "aaa", "1")
    assert res == [{"name": "IBP-Systole", "value": "120"}, {"name": "IBP-Diastole", "value": "80"}]


def test_cleaner_nibp():
    cleaner = Cleaner(get_fields_info())
    res = cleaner.clean_segments([{"name": "NIBP", "value": "120/80"}], "aaa", "1")
    assert res == [{"name": "NIBP-Systole", "value": "120"}, {"name": "NIBP-Diastole", "value": "80"}]

    res = cleaner.clean_segments([{"name": "NIBP", "value": "120/80"}, {"name": "NIBP", "value": None}], "aaa", "1")
    assert res == [{"name": "NIBP-Systole", "value": "120"}, {"name": "NIBP-Diastole", "value": "80"}]


def test_get_clean_value_int():
    mv = MonitorValues(get_fields_info())
    assert mv.get_clean_value({"name": "HR", "value": ["100"]}) == ["100"]


def test_get_clean_value_float():
    mv = MonitorValues(get_fields_info())
    assert mv.get_clean_value({"name": "Temp", "value": ["37.8"]}) == ["37.8"]
    assert mv.get_clean_value({"name": "Temp", "value": ["378"]}) == ["37.8"]
    assert mv.get_clean_value({"name": "Temp", "value": ["370"]}) == ["37.0"]


def test_pp_1():
    pp = PostProcessor(get_fields_info())
    names = ["HR", "RR", "SpO2", "IBP"]
    values = [
        [" 52", "(15)", "93a", "115 /45"],
        [" 2", "(15)", "93a", None],
        ["a52", "(5)", "93a", "115/ 45"],
        ["a52", "(15)", "93a", "115/ 45"],
    ]
    clean = ['52', '15', '93', '115/45']
    for i in range(4):
        segments = [{'name': n, 'value': v} for n, v in zip(names, values[i])]
        cleaned = pp(segments, 'mon1', str(i))
    assert [c['value'] for c in cleaned] == clean


def test_pp_0():
    pp = PostProcessor(get_fields_info(), 1)
    names = ["HR", "RR", "SpO2", "IBP"]
    values = [
        [" 52", "(15)", "90a", "115 /45"],
        [" 2", "(15)", "90a", None],
        ["a52", "(5)", "90a", "115/ 45"],
        ["a52", "(15)", "93a", "115/ 45"],
    ]
    clean = ['52', '15', '93', '115/45']
    for i in range(4):
        segments = [{'name': n, 'value': v} for n, v in zip(names, values[i])]
        cleaned = pp(segments, 'mon1', str(i))
    assert [c['value'] for c in cleaned] == clean


def test_aug_clean_simple():
    mv = MonitorValues(get_fields_info())
    res = mv.get_clean_value(
        augs_dict={
            "name": "Temp",
            "value": ["37", "37.2"]
        },
    )
    assert res == ["37.2"]


def test_aug_cleaning():
    mv = MonitorValues(get_fields_info())

    augs_dict = {"name": "HR", "value": ["120", "12"]}
    assert mv.get_latest_valid_value(augs_dict) == "120"

    augs_dict = {"name": "HR", "value": ["1200", "1200"]}
    assert mv.get_latest_valid_value(augs_dict) == "120"

    time.sleep(2)
    assert mv.get_latest_valid_value(augs_dict, window_size=2) is None

    augs_dict = {"name": "Peep", "value": ["45", "4.5", ".5", "5"]}
    assert mv.get_latest_valid_value(augs_dict) == "4.5"

    # augs_dict = {"name": "Tenp", "value": ["372"]}
    # assert mv.get_latest_valid_value(augs_dict) is None

    augs_dict = {"name": "Temp", "value": ["yui36ga"]}
    assert mv.get_latest_valid_value(augs_dict) == "36"

    augs_dict = {"name": "Temp", "value": ["372"]}
    assert mv.get_latest_valid_value(augs_dict) == "37.2"

    augs_dict = {"name": "NIBP", "value": ["10u0 / 110k"]}
    assert mv.get_latest_valid_value(augs_dict) == "100/110"

    augs_dict = {"name": "SpO2", "value": ["37.2"]}
    assert mv.get_latest_valid_value(augs_dict) is None

    augs_dict = {"name": "SpO2", "value": ["100", "10"]}
    assert mv.get_latest_valid_value(augs_dict) == "100"

    augs_dict = {"name": "SpO2", "value": ["37.2", "10"]}
    assert mv.get_latest_valid_value(augs_dict) == "100"

    augs_dict = {"name": "SpO2", "value": ["NNATO372", "10"]}
    assert mv.get_latest_valid_value(augs_dict) == "100"
