from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='bathfitting',
    version='0.1.0',
    description='Bath fitting via semi-definite relaxation package',
    long_description=readme,
    author='Leonardo Zepeda-Nunez',
    author_email='zepedanunez@wisc.edu',
    url='https://github.com/Forgotten/BathFitting',
    #license=license,
    install_requires=['numpy', 'scipy', 'cvxpy']
    #packages=find_packages(exclude=('tests', 'docs'))
)