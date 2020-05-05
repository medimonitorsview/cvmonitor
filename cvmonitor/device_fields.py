import random
import re
import copy
from collections import Counter, defaultdict
from cvmonitor.aug_clean import MonitorValues
from .utils import is_int


def get_fields_info(device_types=['respirator', 'ivac', 'monitor']):

    ivac = {
        'Medication Name': {'max_len': 10, 'dtype': str},
        'Volume Left to Infuse':  {'max_len': 3, 'min': 10, 'max': None, 'dtype': int},
        'Volume to Insert':  {'max_len': 3, 'min': 10, 'max': None, 'dtype': int},
        'Infusion Rate':  {'max_len': 4, 'min': 0, 'max': None, 'dtype': float, 'num_digits_after_point': 1},
    }
    respirator = {
        'Ventilation Mode': {'max_len': 10, 'dtype': str},
        'Tidal Volume': {'max_len': 3, 'min': 350, 'max': 600, 'dtype': int, "min_range": 200, "max_range": 1000},
        'Expiratory Tidal Volume': {'max_len': 3, 'min': None, 'max': None, 'dtype': int},
        'Rate': {'max_len': 2, 'min': 10, 'max': 40, 'dtype': int, "min_range": 0, "max_range": 99},
        'Total Rate': {'max_len': 2, 'min': 10, 'max': 40, 'dtype': int, "min_range": 0, "max_range": 99},
        'Peep': {'max_len': 2, 'min': None, 'max': None, 'dtype': int, "min_range": 0, "max_range": 99},
        'Ppeak': {'max_len': 2, 'min': None, 'max': 40, 'dtype': int},
        'FIO2': {'max_len': 3, 'min': None, 'max': None, 'dtype': int},
        # FIXME: assume that operator selects only X.X part, without the digit 1
        'I:E Ratio': {'max_len': 2, 'min': None, 'max': None, 'dtype': float, 'num_digits_after_point': 1},
        'Inspiratory time': {'max_len': 2, 'min': None, 'max': None, 'dtype': float, 'num_digits_after_point': 1},
    }
    monitor = {
        'HR': {'max_len': 3, 'min': 45, 'max': 120, 'dtype': int, "max_range": 200, "min_range": 10},
        'SpO2': {'max_len': 3, 'min': 90, 'max': None, 'dtype': int, "min_range": 20, "max_range": 100},
        'RR': {'max_len': 2, 'min': 8, 'max': 26, 'dtype': int, "min_range": 0, "max_range": 99},
        'IBP': {'max_len': 7, 'regex': '.*?([1-2]{0,1}[0-9]{1,2}) *([/17]) *([1-2]{0,1}[0-9]{1,2}).*', 'sub': '\\1/\\3'},
        'NIBP': {'max_len': 7, 'regex': '.*?([1-2]{0,1}[0-9]{1,2}) *([/17]) *([1-2]{0,1}[0-9]{1,2}).*', 'sub': '\\1/\\3'},
        'IBP-Mean': {'max_len': 3, 'dtype': int, "min_range": 10, "max_range": 500},
        'NIBP-Mean': {'max_len': 3, 'dtype': int, "min_range": 10, "max_range": 500},
        'IBP-Systole': {'max_len': 3, 'min': 80, 'max': 180, 'dtype': int, "min_range": 40, "max_range": 299},  # left blood pressure
        'IBP-Diastole': {'max_len': 3, 'min': 40, 'max': 100, 'dtype': int, "min_range": 40, "max_range": 299},  # right blood pressure
        'NIBP-Systole': {'max_len': 3, 'min': 80, 'max': 180, 'dtype': int, "min_range": 40, "max_range": 299},  # left blood pressure
        'NIBP-Diastole': {'max_len': 3, 'min': 40, 'max': 100, 'dtype': int, "min_range": 40, "max_range": 299},  # right blood pressure
        'Temp': {'min_len': 2, 'max_len': 3, 'min': 35.0, 'max': 38.0, 'dtype': float, 'num_digits_after_point': 1, "min_range": 33, "max_range": 45},
        'etCO2': {'max_len': 2, 'min': 24, 'max': 44, 'dtype': int, "min_range": 0, "max_range": 999},
    }
    # for annotations only, currently not found in android app
    # 'hr_saturation': {'max_len': 3, 'min': 45, 'max': 120, 'dtype': int},
    devices = {}

    if 'respirator' in device_types:
        devices.update(respirator)
    if 'monitor' in device_types:
        devices.update(monitor)
    if 'ivac' in device_types:
        devices.update(ivac)
    return devices


def get_field_rand_value(field_info, current=None):
    if field_info.get('dtype') in [float, int]:
        max_val = field_info.get('max') or int(0.99999 * (10**(field_info['max_len'])))
        min_val = field_info.get('min') or 0
        if current is None:
            base = random.randint(min_val,  max_val)
            if 'num_digits_after_point' in field_info:
                divisor = 10**field_info['num_digits_after_point']
                base = float(base) + random.randint(0, divisor) * (1/divisor)
        else:
            base = field_info['dtype'](current) + random.randint(-3, 3)
            if random.randint(0, 1) == 0:
                base += 0.1
            base = max(min_val, min(base, max_val))
            base = field_info['dtype'](base)
    if field_info.get('dtype') in [str]:
        base = random.choice(['wine', 'beer', 'coffee', 'soda', 'water'])
    if 'dtype' not in field_info:
        base = '100/100'  # rstr.xeger(field_info['regex'])
    max_len = field_info.get('max_len', 20)
    base = str(base)
    if '.' in base:
        max_len += 1
    if len(base) > max_len:
        base = base[:max_len]
    return base


def cleanup_field(field_info, field_value):
    res = ''
    if field_info.get('regex'):
        regex_res = re.match(field_info.get('regex'), field_value or '')
        if not regex_res:
            return ''
        if field_info.get('sub'):
            return re.sub(field_info.get('regex'), field_info.get('sub'), field_value or '')
        return ''.join(regex_res.groups())
    if field_info.get('dtype') == str:
        res = field_value
    if field_info.get('dtype') == int:
        for c in field_value:
            if is_int(c):
                res += c
                if len(res) >= field_info.get('max_len', 10):
                    break
    if field_info.get('dtype') == float:
        for c in field_value:
            if is_int(c) or c == '.':
                res += c
                if len(res) >= field_info.get('max_len', 10)+1:
                    break
    return res


# Currently unused, replaced with Cleaner
class PostProcessor():

    """
    Cleaning ocr data
    """
    def __init__(self, devices, average_length=3):
        self.monitors = {}
        self.devices = devices
        self.average_length = average_length

    def clean(self, segments):
        cleaned_segments = {}
        for s in segments:
            if s.get('name') and s.get('name') in self.devices and s.get('value') and s.get('level') != 'crop':
                cleaned = copy.deepcopy(s)
                cleaned['value'] = cleanup_field(self.devices[s['name']], s['value'])
                cleaned_segments[s['name']] = cleaned
        for s in segments:
            if s.get('name') and s.get('name') in self.devices and s.get('value'):
                if s.get('name') in cleaned_segments and not cleaned_segments[s.get('name')].get('value'):
                    cleaned = copy.deepcopy(s)
                    cleaned['value'] = cleanup_field(self.devices[s['name']], s['value'])
                    cleaned_segments[s['name']] = cleaned

        for s in segments:
            if s.get('name') == 'tracker':
                cleaned_segments['tracker'] = s
        cleaned_segments = [s for s in cleaned_segments.values()]
        return cleaned_segments

    def average(self, segments, monitorId, imageId):
        if monitorId not in self.monitors:
            self.monitors[monitorId] = [(imageId, segments)]
            return segments
        if len(self.monitors[monitorId]) <= self.average_length:
            self.monitors[monitorId].append((imageId, segments))

        if len(self.monitors[monitorId]) > self.average_length:
            self.monitors[monitorId].pop(0)

        if len(self.monitors[monitorId]) < self.average_length:
            return segments

        values = {}
        segments_dict = {}
        for pId, pSegments in self.monitors[monitorId]:
            for s in pSegments:
                if s.get('name'):
                    segments_dict[s.get('name')] = s
                    if s.get('name') not in values:
                        values[s['name']] = []
                    values[s['name']].append(s.get('value'))

        for k, v in values.items():
            if v:
                segments_dict[k]['value'] = Counter(v).most_common(1)[0][0]
            else:
                segments_dict[k]['value'] = None
        return [s for s in segments_dict.values()]

    def __call__(self, segments, monitorId, imageId):
        tracker = None
        for s in segments:
            if s.get('name') == 'tracker':
                tracker = copy.deepcopy(s)
        resseg = self.clean(segments)  # self.average(self.clean(segments),monitorId,imageId)
        for s in segments:
            if s.get('name') == 'tracker':
                s.update(tracker)
        return resseg


class Cleaner:

    def __init__(self, sensors):
        self.sensors = sensors
        self.monitors = defaultdict(lambda: MonitorValues(sensors))

    def sysdis(self, segments_dict):
        if "IBP" in segments_dict and "IBP-Systole" not in segments_dict and "IBP-Diastole" not in segments_dict:
            values = segments_dict["IBP"]["value"]
            if all(['/' in v for v in values]):
                systole = copy.deepcopy(segments_dict["IBP"])
                systole["value"] = [x.split("/")[0] for x in values]
                diastole = copy.deepcopy(segments_dict["IBP"])
                diastole["value"] = [x.split("/")[-1] for x in values]
                systole["name"] = "IBP-Systole"
                diastole["name"] = "IBP-Diastole"
                segments_dict["IBP-Systole"] = systole
                segments_dict["IBP-Diastole"] = diastole
                del segments_dict["IBP"]

        if "NIBP" in segments_dict and "NIBP-Systole" not in segments_dict and "NIBP-Diastole" not in segments_dict:
            values = segments_dict["NIBP"]["value"]
            if all(['/' in v for v in values]):
                systole = copy.deepcopy(segments_dict["NIBP"])
                systole["value"] = [x.split("/")[0] for x in values]
                diastole = copy.deepcopy(segments_dict["NIBP"])
                diastole["value"] = [x.split("/")[-1] for x in values]
                systole["name"] = "NIBP-Systole"
                diastole["name"] = "NIBP-Diastole"
                segments_dict["NIBP-Systole"] = systole
                segments_dict["NIBP-Diastole"] = diastole
                del segments_dict["NIBP"]

    def clean_segments(self, segments, monitorId, imageId):
        if monitorId is None:
            monitorId = "unkown monitor"
        mv: MonitorValues = self.monitors[monitorId]
        segments_dict = {}
        for segment in segments:
            name = segment.get("name")
            if name is None or not name:
                continue
            if name not in segments_dict:
                segments_dict[name] = copy.deepcopy(segment)
                segments_dict[name]["value"] = list()
            cleaned_field = segment.get("value")
            if name in self.sensors and segment.get("value"):
                cleaned_field = cleanup_field(self.sensors[name], segment.get("value"))
            segments_dict[name]["value"].append(cleaned_field)
            if segment.get("level") == "crop":
                segments_dict[name].update(dict(top=segment["top"], left=segment["left"], bottom=segment["bottom"], right=segment["right"]))

        # Split systole-diastole
        self.sysdis(segments_dict)
        for name, aug_dict in segments_dict.items():
            if name in self.sensors and self.sensors[name].get("dtype") in [float, int]:
                res = mv.get_latest_valid_value(aug_dict)
                if res:
                    segments_dict[name]["value"] = res
            if isinstance(segments_dict[name]["value"], list):
                segments_dict[name]["value"] = Counter(segments_dict[name]["value"]).most_common(1)[0][0]
            if "clean_value" in segments_dict[name]:
                segments_dict[name].pop("clean_value")
        return [v for k, v in segments_dict.items()]
