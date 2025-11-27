"""A basic test for serl_launcher."""

import unittest
from serl_launcher.common import typing


class SerlLauncherTest(unittest.TestCase):

  def test_basic(self):
    print(dir(typing))
    self.assertTrue(hasattr(typing, "PRNGKey"))


if __name__ == "__main__":
  unittest.main()
