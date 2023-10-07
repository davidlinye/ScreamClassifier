import librosa
import librosa.display
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px

df=pd.DataFrame(np.load('./resources/working_data/data_with_vggish.npy',allow_pickle = True),columns=['video_id','start_time','mid_ts','label','audio','vggish'])
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import image
import os
import PIL.Image

df['magnitude_spectrogram']=''
df['power_spectrogram']=''

window_size = 1024
window = np.hanning(window_size)
mx=len(df)
for index,row in df.iterrows():
    #print(f"working on row {index} of {mx}")
    file_name1 = './resources/working_data/'+f'block_{index}_magnitude.png'
    file_name2 = './resources/working_data/'+f'block_{index}_power.png'
    # sf.write(file_name, row['audio'], 44100)
    S = np.abs(librosa.stft(row['audio']))
    #librosa.display.specshow(librosa.amplitude_to_db(S**2,ref=np.max),y_axis='log')
    out1 = 2 * S / np.sum(window)
    out2 = 2 * (S**2) / np.sum(window)
    fig1 = plt.Figure()
    canvas1 = FigureCanvas(fig1)
    ax1 = fig1.add_subplot(111)
    p1 = librosa.display.specshow(librosa.amplitude_to_db(out1, ref=np.max), ax=ax1, y_axis='log', x_axis='time')
    fig1.savefig(file_name1)

    fig2 = plt.Figure()
    canvas2 = FigureCanvas(fig2)
    ax2 = fig2.add_subplot(111)
    p2 = librosa.display.specshow(librosa.amplitude_to_db(out2, ref=np.max), ax=ax2, y_axis='log', x_axis='time')
    fig2.savefig(file_name2)

    magnitude_rgba = PIL.Image.open(file_name1)
    power_rgba = PIL.Image.open(file_name2)
    magnitude = magnitude_rgba.convert('RGB')
    power = power_rgba.convert('RGB')

    df['magnitude_spectrogram'][index] = np.asarray(magnitude)
    df['power_spectrogram'][index] = np.asarray(power)


    if os.path.exists(file_name1):
        os.remove(file_name1)
    if os.path.exists(file_name2):
        os.remove(file_name2)
    # if index == 1:
    #     break

out = df.to_numpy()
np.save('../resources/working_data/data_with_vggish_and_spectrogram_images.npy', out)