import io
import os
import requests
import stat
import zipfile


def download_and_extract_zip(url):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

def set_execute_permissions(file):
    st = os.stat(file)
    os.chmod(file, st.st_mode | stat.S_IEXEC)

download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip")
assert(os.path.isdir("Banana_Linux"))
download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip")
assert(os.path.isdir("Banana_Linux_NoVis"))

set_execute_permissions("Banana_Linux/Banana.x86")
set_execute_permissions("Banana_Linux/Banana.x86_64")
set_execute_permissions("Banana_Linux_NoVis/Banana.x86")
set_execute_permissions("Banana_Linux_NoVis/Banana.x86_64")
