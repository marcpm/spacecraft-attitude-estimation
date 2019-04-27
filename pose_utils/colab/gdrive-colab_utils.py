from google.colab import drive
from google.colab import files
import shutil

drive.mount('/gdrive')

def upload_file_mounted(filename):
  """Uploads model to mounted googledrive, if googleauth expires and gdrive is unmounted 
      saves model to local machine. 
      Unlike PyDrive, Updating is handled overwriting, without any delete functions  """  
  try:
    dest_path = "../gdrive/My Drive/_colab/kelvins/testing/" 
    shutil.copy2(filename, dest_path)
    print (f"File {filename} upload successful!")
  except FileNotFoundError:
    print(f"Disk unmounted. Upload failed. {filename}\n Downloading to local machine....")
    files.download(filename)

print("Downloading datasets and utils....")
# download dataset 
shutil.copy2("../gdrive/My Drive/_colab/kelvins/data/speed.zip","../content/" )
# download utils
shutil.copy2("../gdrive/My Drive/_colab/kelvins/pose_utils/pose_utils.zip", "../content/pose_utils.zip")

print("Downloads complete! \n Unzipping data ... ")
!unzip -q speed.zip
!unzip -q pose_utils.zip
print("\n Unzipping complete!")