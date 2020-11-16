from setuptools import setup

setup(
  name='rl_learn',
  version='1.0.2',
  description='A package to accompany the book "Reinforcement Learning: A Python Introduction"',
  long_description="rl-learn is a Python package that teaches fundamental reinforcement learning techniques in Python. It closely follows Sutton and Barto's famous \"Reinforcement Learning: An Introduction\".\nCheck out the GitHub for example code: https://github.com/gahogg/rl_learn",
  author='Gregory Hogg',
  author_email='gahogg@uwaterloo.ca',
  url='https://github.com/gahogg/rl_learn.git',
  license='MIT',
  packages=['rl_learn', 'rl_learn/bandits'],
  install_requires=["numpy>=1.18.1,<=1.18.1", "matplotlib>=3.1.3,<=3.1.3"],
  zip_safe=False)
