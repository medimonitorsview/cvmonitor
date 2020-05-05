from setuptools import setup, find_packages


setup(
    name="cvmonitor",
    use_scm_version=True,
    packages=find_packages(),
    install_requires=open("requirements.txt").readlines(),
    entry_points={
        'console_scripts':
            [
                'cvmonitor=cvmonitor.server:main'
            ],
    },
)
