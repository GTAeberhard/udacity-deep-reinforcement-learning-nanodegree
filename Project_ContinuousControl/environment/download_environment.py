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

if not os.path.isdir(os.path.join(CURRENT_PATH, "ReacherSingleAgent_Linux")):
    print("Downloading ReacherSingleAgent_Linux environment...")
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip",
                             CURRENT_PATH)
    shutil.move(os.path.join(CURRENT_PATH, "Reacher_Linux"),
                os.path.join(CURRENT_PATH, "ReacherSingleAgent_Linux"))
    assert(os.path.isdir(os.path.join(CURRENT_PATH, "ReacherSingleAgent_Linux")))
    set_execute_permissions(os.path.join(CURRENT_PATH, "ReacherSingleAgent_Linux/Reacher.x86"))
    set_execute_permissions(os.path.join(CURRENT_PATH, "ReacherSingleAgent_Linux/Reacher.x86_64"))

if not os.path.isdir(os.path.join(CURRENT_PATH, "ReacherMultiAgent_Linux")):
    print("Downloading ReacherMultiAgent_Linux environment...")
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip",
                             CURRENT_PATH)
    shutil.move(os.path.join(CURRENT_PATH, "Reacher_Linux"),
                os.path.join(CURRENT_PATH, "ReacherMultiAgent_Linux"))
    assert(os.path.isdir(os.path.join(CURRENT_PATH, "ReacherMultiAgent_Linux")))
    set_execute_permissions(os.path.join(CURRENT_PATH, "ReacherMultiAgent_Linux/Reacher.x86"))
    set_execute_permissions(os.path.join(CURRENT_PATH, "ReacherMultiAgent_Linux/Reacher.x86_64"))

if not os.path.isdir(os.path.join(CURRENT_PATH, "ReacherSingleAgent_Linux_NoVis")):
    print("Downloading ReacherSingleAgent_Linux_NoVis environment...")
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip",
                             CURRENT_PATH)
    shutil.move(os.path.join(CURRENT_PATH, "Reacher_Linux_NoVis"),
                os.path.join(CURRENT_PATH, "ReacherSingleAgent_Linux_NoVis"))
    assert(os.path.isdir(os.path.join(CURRENT_PATH, "ReacherSingleAgent_Linux_NoVis")))
    set_execute_permissions(os.path.join(CURRENT_PATH, "ReacherSingleAgent_Linux_NoVis/Reacher.x86"))
    set_execute_permissions(os.path.join(CURRENT_PATH, "ReacherSingleAgent_Linux_NoVis/Reacher.x86_64"))

if not os.path.isdir(os.path.join(CURRENT_PATH, "ReacherMultiAgent_Linux_NoVis")):
    print("Downloading ReacherMultiAgent_Linux_NoVis environment...")
    download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip",
                             CURRENT_PATH)
    shutil.move(os.path.join(CURRENT_PATH, "Reacher_Linux_NoVis"),
                os.path.join(CURRENT_PATH, "ReacherMultiAgent_Linux_NoVis"))
    assert(os.path.isdir(os.path.join(CURRENT_PATH, "ReacherMultiAgent_Linux_NoVis")))
    set_execute_permissions(os.path.join(CURRENT_PATH, "ReacherMultiAgent_Linux_NoVis/Reacher.x86"))
    set_execute_permissions(os.path.join(CURRENT_PATH, "ReacherMultiAgent_Linux_NoVis/Reacher.x86_64"))
