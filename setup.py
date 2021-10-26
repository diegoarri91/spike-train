from setuptools import setup

setup(name='spike-train',
      version='0.1',
      description='Python package that implements various spike trains operations.',
      url='https://github.com/diegoarri91/spike-train',
      author='Diego M. Arribas',
      author_email='diegoarri91@gmail.com',
      license='MIT',
      packages=['spiketrain'],
      install_requires=['matplotlib', 'numpy', 'scipy']
      )
