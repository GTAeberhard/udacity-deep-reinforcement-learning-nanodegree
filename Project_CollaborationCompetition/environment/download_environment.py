"""
Script which downloads the necessary Unity environments from the Udacity AWS servers
which are required for working with this project.

Simply run this script from the command line with a Python interpreter and the
necessary files will be downloaded to the 'environment' folder.
"""
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

if not os.path.isdir(os.path.join(CURRENT_PATH, "Tennis_Linux")):
    print("Downloading Tennis_Linux environment...")
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip",
                             CURRENT_PATH)
    assert(os.path.isdir(os.path.join(CURRENT_PATH, "Tennis_Linux")))
    set_execute_permissions(os.path.join(CURRENT_PATH, "Tennis_Linux/Tennis.x86"))
    set_execute_permissions(os.path.join(CURRENT_PATH, "Tennis_Linux/Tennis.x86_64"))

if not os.path.isdir(os.path.join(CURRENT_PATH, "Tennis_Linux_NoVis")):
    print("Downloading Tennis_Linux_NoVis environment...")
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip",
                             CURRENT_PATH)
    assert(os.path.isdir(os.path.join(CURRENT_PATH, "Tennis_Linux_NoVis")))
    set_execute_permissions(os.path.join(CURRENT_PATH, "Tennis_Linux_NoVis/Tennis.x86"))
    set_execute_permissions(os.path.join(CURRENT_PATH, "Tennis_Linux_NoVis/Tennis.x86_64"))
