import os
import subprocess

def download_radioml2016():
    data_dir = 'data/raw/RadioML2016.10a'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    cmd = (
        '/opt/anaconda3/bin/kaggle datasets download '
        '-d nolasthitnotomorrow/radioml2016-deepsigcom '
        f'-p {data_dir} --unzip'
    )
    print(f"Running command:\n{cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    if result.returncode == 0:
        print("Download and unzip successful!")
    else:
        print("Something went wrong during download.")

if __name__ == "__main__":
    download_radioml2016()
