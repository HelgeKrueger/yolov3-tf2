from setuptools import setup

with open('requirements-gpu.txt') as f:
    requirements = f.read().splitlines()
    
requirements = [r for r in requirements if r not in ['', '-e .']]

setup(name='yolov3_tf2',
      version='0.1',
      url='https://github.com/zzh8829/yolov3-tf2',
      author='Zihao Zhang',
      author_email='zzh8829@gmail.com',
      packages=['yolov3_tf2'],
      install_requires=requirements)
