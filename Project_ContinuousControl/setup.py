#!/usr/bin/env python

from setuptools import setup, Command, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info


def download_reacher_environment():
    import io
    import os
    import requests
    import shutil
    import stat
    import zipfile

    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

    def download_and_extract_zip(url, target_folder):
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(target_folder)

    def set_execute_permissions(file):
        st = os.stat(file)
        os.chmod(file, st.st_mode | stat.S_IEXEC)

    if not os.path.isdir(os.path.join(CURRENT_PATH, "environment/ReacherSingleAgent_Linux")):
        download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip",
                                 "environment")
        shutil.move(os.path.join(CURRENT_PATH, "environment/Reacher_Linux"),
                    os.path.join(CURRENT_PATH, "environment/ReacherSingleAgent_Linux"))
        assert(os.path.isdir(os.path.join(CURRENT_PATH, "environment/ReacherSingleAgent_Linux")))
        set_execute_permissions(os.path.join(CURRENT_PATH, "environment/ReacherSingleAgent_Linux/Reacher.x86"))
        set_execute_permissions(os.path.join(CURRENT_PATH, "environment/ReacherSingleAgent_Linux/Reacher.x86_64"))

    if not os.path.isdir("environment/ReacherMultiAgent_Linux"):
        download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip",
                                 "environment")
        shutil.move(os.path.join(CURRENT_PATH, "environment/Reacher_Linux"),
                    os.path.join(CURRENT_PATH, "environment/ReacherMultiAgent_Linux"))
        assert(os.path.isdir(os.path.join(CURRENT_PATH, "environment/ReacherMultiAgent_Linux")))
        set_execute_permissions(os.path.join(CURRENT_PATH, "environment/ReacherMultiAgent_Linux/Reacher.x86"))
        set_execute_permissions(os.path.join(CURRENT_PATH, "environment/ReacherMultiAgent_Linux/Reacher.x86_64"))

    if not os.path.isdir(os.path.join(CURRENT_PATH, "environment/ReacherSingleAgent_Linux_NoVis")):
        download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip",
                                 "environment")
        shutil.move(os.path.join(CURRENT_PATH, "environment/Reacher_Linux_NoVis"),
                    os.path.join(CURRENT_PATH, "environment/ReacherSingleAgent_Linux_NoVis"))
        assert(os.path.isdir(os.path.join(CURRENT_PATH, "environment/ReacherSingleAgent_Linux_NoVis")))
        set_execute_permissions(os.path.join(CURRENT_PATH, "environment/ReacherSingleAgent_Linux_NoVis/Reacher.x86"))
        set_execute_permissions(os.path.join(CURRENT_PATH, "environment/ReacherSingleAgent_Linux_NoVis/Reacher.x86_64"))

    if not os.path.isdir(os.path.join(CURRENT_PATH, "environment/ReacherMultiAgent_Linux_NoVis")):
        download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip",
                                 "environment")
        shutil.move(os.path.join(CURRENT_PATH, "environment/Reacher_Linux_NoVis"),
                    os.path.join(CURRENT_PATH, "environment/ReacherMultiAgent_Linux_NoVis"))
        assert(os.path.isdir(os.path.join(CURRENT_PATH, "environment/ReacherMultiAgent_Linux_NoVis")))
        set_execute_permissions(os.path.join(CURRENT_PATH, "environment/ReacherMultiAgent_Linux_NoVis/Reacher.x86"))
        set_execute_permissions(os.path.join(CURRENT_PATH, "environment/ReacherMultiAgent_Linux_NoVis/Reacher.x86_64"))


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        import requests
        download_reacher_environment()


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        import requests
        download_reacher_environment()


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        import requests
        download_reacher_environment()


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
