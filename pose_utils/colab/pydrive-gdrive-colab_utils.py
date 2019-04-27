#!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

def upload_file(filename):
  try:
    upload = drive.CreateFile({"title": filename})
    upload.SetContentFile(filename)
    upload.Upload()
    print("Uploaded file {}.".format(filename))
  except Exception as e:
    print(f"Error found trying to upload {filename}\n Downloading to local machine....")
    files.download(filename)
       
def download_file(ID, name):
  download = drive.CreateFile({'id': ID})
  download.GetContentFile(name)
  print('downloaded file  {}'.format(download.get('id')))
  
def delete_file(filename):
  delete = drive.CreateFile({"title": filename})
  delete.Delete()
  
#download and resume  last trained model
download_file("1EfeDp8vEJLogWIwhWows8Mbuh03hQBIH", "recent-model.pth")
#  download dataset.
download_file("1NC0VFXVyU5_imbnXUY0Cm_6vvshShU-w", "speed.zip")
# download utils
download_file("1uccfOd5zjXL9-ug2mZZ4AMPuYtAf9MT2", "utils.zip")
# Unzip data and utils
!unzip -q speed.zip
!unzip -q utils.zip