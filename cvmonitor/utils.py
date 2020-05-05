import cv2


def is_int(val):
    try:
        int(val)
    except ValueError:
        return False
    return True


def draw_segments(image, segments):
    for s in segments:
        location = [int(s['left']), int(s['top'])], [int(s['right']), int(s['bottom'])]
        color = [0, 255, 0]
        cv2.rectangle(image, tuple(location[0]), tuple(location[1]), color, 1)
        cv2.putText(
            img=image, text=str(f"{s.get('name','?')}: {s.get('value','?')}"), org=(location[0][0], location[1][1] + 10),
            fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=color, thickness=1
        )
    return image
