"""Run: python -m unittest discover -s tests -v"""
import unittest
from pathlib import Path
import cv2
import numpy as np
from modules import lab01,lab02,lab03,lab04,lab05,lab07

class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset=Path(__file__).resolve().parents[1]/'static/assets/sample-flower.png'
        cls.image=cv2.imread(str(asset))
        assert cls.image is not None

    def test_lab_01_images(self):
        r=lab01.process(self.image,{'scale':'75','rotation':'180'})
        self.assertGreaterEqual(len(r['outputs']),7)

    def test_lab_02_edges(self):
        r=lab02.process(self.image,{'canny_low':'70','canny_high':'200'})
        self.assertEqual(len(r['outputs']),4)

    def test_lab_03_histogram(self):
        r=lab03.process(self.image,{'clip':'3'})
        self.assertEqual(len(r['outputs']),6)

    def test_lab_04_sift(self):
        r=lab04.process(self.image,{'detector':'SIFT'})
        self.assertGreater(len(r['outputs']),1)

    def test_lab_05_compression(self):
        r=lab05.process(self.image,{'components':'12','mode':'compression'})
        self.assertEqual(len(r['outputs']),3)

    def test_lab_07_segmentation(self):
        r=lab07.process(self.image,{'seg_method':'compare'})
        self.assertEqual(len(r['outputs']),7)

if __name__ == '__main__':unittest.main()
