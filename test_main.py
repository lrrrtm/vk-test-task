import unittest

import cv2
import numpy as np

from main import find_green_button


class FindGreenButtonTest(unittest.TestCase):
    def test_finds_center_of_green_rectangle(self) -> None:
        image = np.zeros((500, 300, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 200), (250, 280), (0, 200, 0), -1)

        self.assertEqual(find_green_button(image), (150, 240))

    def test_returns_none_without_green(self) -> None:
        image = np.zeros((500, 300, 3), dtype=np.uint8)

        self.assertIsNone(find_green_button(image))


if __name__ == "__main__":
    unittest.main()
