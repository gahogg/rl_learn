from setuptools import setup

setup(
  name='rl_learn',
  version='0.1.5',
  description='A package to accompany the book "Reinforcement Learning: A Python Introduction"',
  long_description="rl-learn is a Python package to accompany \"Reinforcement Learning: A Python Introduction\", which is a book that teaches fundamental reinforcement learning techniques in Python. It closely follows Sutton and Barto's famous \"Reinforcement Learning: An Introduction\".",
  author='Gregory Hogg',
  author_email='gahogg@uwaterloo.ca',
  url='https://github.com/gahogg/rl_learn.git',
  license='MIT',
  packages=['rl_learn', 'rl_learn/bandits'],
  zip_safe=False)
