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

if not os.path.isdir(os.path.join(CURRENT_PATH, "Banana_Linux")):
    print("Downloading Banana_Linux environment...")
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip",
                             CURRENT_PATH)
    assert(os.path.isdir(os.path.join(CURRENT_PATH, "Banana_Linux")))
    set_execute_permissions(os.path.join(CURRENT_PATH, "Banana_Linux/Banana.x86"))
    set_execute_permissions(os.path.join(CURRENT_PATH, "Banana_Linux/Banana.x86_64"))

if not os.path.isdir(os.path.join(CURRENT_PATH, "Banana_Linux_NoVis")):
    print("Downloading Banana_Linux_NoVis environment...")
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip",
                             CURRENT_PATH)
    assert(os.path.isdir(os.path.join(CURRENT_PATH, "Banana_Linux_NoVis")))
    set_execute_permissions(os.path.join(CURRENT_PATH, "Banana_Linux_NoVis/Banana.x86"))
    set_execute_permissions(os.path.join(CURRENT_PATH, "Banana_Linux_NoVis/Banana.x86_64"))

if not os.path.isdir(os.path.join(CURRENT_PATH, "Banana_Linux_Pixels")):
    print("Downloading Banana_Linux_Pixels environment...")
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip",
                             CURRENT_PATH)
    shutil.move(os.path.join(CURRENT_PATH, "VisualBanana_Linux"),
                os.path.join(CURRENT_PATH, "Banana_Linux_Pixels"))
    assert(os.path.isdir(os.path.join(CURRENT_PATH, "Banana_Linux_Pixels")))
    set_execute_permissions(os.path.join(CURRENT_PATH, "Banana_Linux_Pixels/Banana.x86"))
    set_execute_permissions(os.path.join(CURRENT_PATH, "Banana_Linux_Pixels/Banana.x86_64"))
