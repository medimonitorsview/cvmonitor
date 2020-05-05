import numpy as np
import re
import datetime


class SensorValue:

    def __init__(self, name):
        self.sensor_name = name
        self.last_valid_time = datetime.datetime(1970, 1, 1)
        self.last_valid_value = None


class MonitorValues:
    """
    Clean *numeric* monitor values.
    """

    def __init__(self, sensors):
        self.sensors = sensors
        self.values_dict = {name: SensorValue(name) for name in sensors.keys()}

    def get_latest_valid_value(self, augs_dict, window_size=10):
        """
        Returns a single valid value from the OCR readings. If fails to clean the current readings, take the last valid reading
        within the stated time frame, or None if there isn't one.
        Input:
            augs_dict: {
                         name: "Temp",
                         value: [37, 37.2]
                        },
            window_size: 10 (seconds)
        Returns:
            value: 37.2
        """
        value = self.get_value_from_augs(augs_dict)
        sensor_name = augs_dict["name"]

        # TODO Do we want to throw an exception in that case? This requires defining all possible sensors beforehand
        if sensor_name not in self.values_dict:
            self.values_dict[sensor_name] = SensorValue(sensor_name)

        if value is None:
            if datetime.datetime.now() - self.values_dict[sensor_name].last_valid_time <= datetime.timedelta(0, window_size):
                value = self.values_dict[sensor_name].last_valid_value
        else:
            self.values_dict[sensor_name].last_valid_time = datetime.datetime.now()
            self.values_dict[sensor_name].last_valid_value = value

        return value

    def get_value_from_augs(self, augs_dict):
        """
        Returns a single valid value from the OCR readings, or None if fails to clean.
        Input:
            augs_dict: {
                         name: "Temp",
                         value: [37, 37.2]
                        }
        Returns:
            result: [37.2]
        """
        # TODO Add error logging?
        try:
            augs_dict["clean_value"] = self.get_clean_value(augs_dict)
        except Exception:
            return None

        if is_error(augs_dict, self.sensors):
            return None
        else:
            return augs_dict["clean_value"][0]

    def remove_sensor_overlap(self, values, name):
        global overlaps
        result = set()
        vl = list(values)
        for i, cur_v in enumerate(vl):
            found_overlap = False
            for sensor_name, sensor_value in self.values_dict.items():
                if sensor_name != name and str(sensor_value.last_valid_value) == cur_v:
                    print("found overlap in {}:{} with previous {}:{}".format(name, cur_v, sensor_name, sensor_value.last_valid_value))
                    found_overlap = True
                    overlaps += 1
            if not found_overlap:
                result.add(cur_v)
        return result

    def remove_invalid_ranges(self, values, name):
        result = set()
        for v in values:
            clean_dict = {'name': name, 'clean_value': [v]}
            if is_valid_ranges(clean_dict, self.sensors):
                result.add(v)
            elif name in self.values_dict and v in str(self.values_dict[name].last_valid_value):
                result.add(str(self.values_dict[name].last_valid_value))
        return result

    def get_clean_value(self, augs_dict):
        """
        Clean the OCR readings for a specific value according to several common patterns.
        Input:
            augs_dict: {
                         name: "Temp",
                         value: [37, 37.2]
                        }
        Returns:
            result: [37.2]
        """

        values = augs_dict["value"]
        # remove invalid characters
        result = list(filter(lambda x: x != '', set([re.sub('[^0-9.:/]', '', v) for v in values])))
        # fix common temperature error ([372] => [37.2])
        if augs_dict["name"] == "Temp":
            result = fix_temp(result)
        else:
            # fix [48, 4.8] => [4.8]
            if "." in str(result) and len(set([x.replace('.', '') for x in result])) == 1:
                result = set([x for x in result if "." in x])

        # if one of the values isn't within the ranges, remove it, unless it's a substring of a previous valid value and then replace it
        result = self.remove_invalid_ranges(result, augs_dict["name"])

        # if one value is a substring of the other, take the longer one ([48, 8] => [48], [48, .8] => 4.8)
        result = remove_substrings(result, augs_dict["name"])

        # if a value is the same as a different sensor value in the last window, remove it
        result = self.remove_sensor_overlap(result, augs_dict["name"])

        # make sure it's a set
        result = list(set(result))

        return result


def fix_temp(result):
    for idx, v in enumerate(result):
        v = re.sub('[:/]', '', v)
        if "." not in v and int(v) > 350 and int(v) < 420:
            v = v[:2] + "." + v[-1]
        result[idx] = v

    return result


def remove_substrings(values, name):
    result = set()
    substr_found = False
    for i, v in enumerate(values):
        for j, m in enumerate(values):
            if i != j and v in m:
                result.add(m)
                substr_found = True
            elif i != j and v.replace(".", "") in m:
                if name == "Peep":
                    result.add(m.replace(v.replace(".", ""), v))
                    substr_found = True
    # if there are several "mother" strings, take the longest one ([45, 4.5, .5, 5] => [45, 4.5, .5] => [4.5])
    if substr_found and len(result) > 1:
        ls = list(result)
        result = [ls[np.argmax([len(x) for x in ls])]]
    if not substr_found:
        result = values
    return result


def check_value(name, value, sensors):
    """
    Checks if a value is within the valid range according to its name
    """
    min_th = sensors.get(name, {}).get("min_range", None)
    max_th = sensors.get(name, {}).get("max_range", None)
    try:
        numeric_value = float(value)
        if min_th is not None and numeric_value < min_th:
            return False
        if max_th is not None and numeric_value > max_th:
            return False
    except Exception:
        return False
    return True


def is_error(cleaned_augs_dict, sensors):
    """
    Checks if the OCR readings of a specific value contain an error. More than one valid augmentation value is considered an error.
    """
    if len(cleaned_augs_dict["clean_value"]) != 1:
        return True
    return not is_valid_ranges(cleaned_augs_dict, sensors)


def is_valid_ranges(cleaned_augs_dict, sensors):
    """
    Checks if cleaned_augs_dict contains valid values; NIBP requires two checks, for every side of the slash (120/80)
    """
    if cleaned_augs_dict["name"] != 'NIBP':
        return check_value(cleaned_augs_dict["name"], cleaned_augs_dict["clean_value"][0], sensors)
    else:
        name_s, name_d = 'NIBPs', 'NIBPd'
        try:
            value_s, value_d = cleaned_augs_dict["clean_value"][0].split('/')
        except Exception:
            return False
        return check_value(name_s, value_s, sensors) and check_value(name_d, value_d, sensors)


