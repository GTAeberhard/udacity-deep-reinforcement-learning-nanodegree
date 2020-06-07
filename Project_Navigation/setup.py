#!/usr/bin/env python

from setuptools import setup, Command, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info


def download_banana_environment():
    import io
    import os
    import requests
    import stat
    import zipfile

    def download_and_extract_zip(url, target_folder):
        print("Downloading and extracting {}...".format(url))
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(target_folder)

    def set_execute_permissions(file):
        st = os.stat(file)
        os.chmod(file, st.st_mode | stat.S_IEXEC)

    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip",
                             "environment")
    assert(os.path.isdir("environment/Banana_Linux"))
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip",
                             "environment")
    assert(os.path.isdir("environment/Banana_Linux_NoVis"))
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip",
                             "environment")
    os.rename("environment/VisualBanana_Linux", "environment/Banana_Linux_Pixels")
    assert(os.path.isdir("environment/Banana_Linux_Pixels"))

    set_execute_permissions("environment/Banana_Linux/Banana.x86")
    set_execute_permissions("environment/Banana_Linux/Banana.x86_64")
    set_execute_permissions("environment/Banana_Linux_NoVis/Banana.x86")
    set_execute_permissions("environment/Banana_Linux_NoVis/Banana.x86_64")
    set_execute_permissions("environment/Banana_Linux_Pixels/Banana.x86")
    set_execute_permissions("environment/Banana_Linux_Pixels/Banana.x86_64")


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        import requests
        download_banana_environment()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        import requests
        download_banana_environment()


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        import requests
        download_banana_environment()


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="unityagents",
    version="0.4.0",
    description="Unity Machine Learning Agents",
    license="Apache License 2.0",
    author="Unity Technologies",
    author_email="ML-Agents@unity3d.com",
    url="https://github.com/Unity-Technologies/ml-agents",
    packages=find_packages(),
    install_requires=required,
    setup_requires="requests",
    cmdclass={
        "install": CustomInstallCommand,
        "egg_info": CustomEggInfoCommand,
        "develop": CustomDevelopCommand
    },
    long_description=("Unity Machine Learning Agents allows researchers and developers "
                      "to transform games and simulations created using the Unity Editor into environments "
                      "where intelligent agents can be trained using reinforcement learning, evolutionary "
                      "strategies, or other machine learning methods through a simple to use Python API.")
)
