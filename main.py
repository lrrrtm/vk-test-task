import argparse
import subprocess
import time

import cv2
import numpy as np


TIMEOUT_SECONDS = 10.0


def run_adb(*args: str, serial: str | None = None, timeout: float = 10.0) -> bytes:
    command = ["adb"]
    if serial:
        command += ["-s", serial]
    try:
        result = subprocess.run(
            [*command, *args], capture_output=True, timeout=timeout, check=False
        )
    except FileNotFoundError as error:
        raise RuntimeError("adb не найден. Добавьте adb в PATH.") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Команда adb превысила таймаут.") from error

    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(message or "Команда adb завершилась с ошибкой.")
    return result.stdout


def get_device() -> str:
    lines = run_adb("devices").decode(errors="replace").splitlines()[1:]
    devices = [line.split()[0] for line in lines if line.rstrip().endswith("\tdevice")]
    if not devices:
        raise RuntimeError("Подключенное и авторизованное устройство не найдено.")
    if len(devices) > 1:
        raise RuntimeError("Подключено несколько устройств. Оставьте подключенным одно.")
    return devices[0]


def launch_app(package: str, serial: str) -> None:
    output = run_adb(
        "shell",
        "monkey",
        "-p",
        package,
        "-c",
        "android.intent.category.LAUNCHER",
        "1",
        serial=serial,
    ).decode(errors="replace")
    if "No activities found" in output or "monkey aborted" in output.lower():
        raise RuntimeError(f"Не удалось запустить пакет {package}")


def take_screenshot(serial: str, timeout: float) -> np.ndarray:
    png = run_adb("exec-out", "screencap", "-p", serial=serial, timeout=timeout)
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Не удалось декодировать скриншот устройства")
    return image


def find_green_button(image: np.ndarray) -> tuple[int, int] | None:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 70, 50), (90, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    height, width = image.shape[:2]
    screen_area = height * width
    candidates: list[tuple[float, int, int]] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        box_area = w * h
        if (
            area >= screen_area * 0.0005
            and box_area <= screen_area * 0.4
            and w >= 24
            and h >= 24
            and area / box_area >= 0.55
        ):
            candidates.append((area, x + w // 2, y + h // 2))

    if not candidates:
        return None
    _, x, y = max(candidates)
    return x, y


def find_and_tap(serial: str) -> tuple[int, int] | None:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while (remaining := deadline - time.monotonic()) > 0:
        point = find_green_button(take_screenshot(serial, remaining))
        if point:
            run_adb("shell", "input", "tap", *map(str, point), serial=serial)
            return point
        time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", help="имя пакета, например com.vkontakte.android")
    args = parser.parse_args()

    try:
        serial = get_device()
        launch_app(args.package, serial)
        point = find_and_tap(serial)
    except RuntimeError as error:
        print(f"Ошибка: {error}")
        return 1

    if point is None:
        print("Зеленая кнопка не найдена")
        return 2
    print(f"Зеленая кнопка нажата: x={point[0]}, y={point[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
