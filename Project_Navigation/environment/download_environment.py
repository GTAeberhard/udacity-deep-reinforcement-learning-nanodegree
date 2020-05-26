import io
import requests
import zipfile

from os import path


def download_and_extract_zip(url):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip")
assert(path.isdir("Banana_Linux"))
download_and_extract_zip("https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip")
assert(path.isdir("Banana_Linux_NoVis"))
