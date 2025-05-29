from setuptools import setup, find_packages

setup(
    name = 'scpilot',
    version = '0.1',
    author = 'Jialiang Wang',
    author_email = '18846091447@163.com',
    license = 'MIT',
    packages = find_packages(include=['scpilot', 'scpilot.*']),
    zip_safe = False,
)


# python setup.py develop