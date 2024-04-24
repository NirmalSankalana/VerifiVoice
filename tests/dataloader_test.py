import unittest
from verifyvoice import DataLoader


class TestDataLoader(unittest.TestCase):

    def setUp(self) -> None:
        self.load_audio = DataLoader.load_audio

    def test_load_audio(self):
        sample1 = "../samples/dr-uthaya-e1.mp3"
        s1 = self.load_audio(sample1, 160)
        self.assertEqual(len(s1), 10)


def __main__():
    unittest.main()
