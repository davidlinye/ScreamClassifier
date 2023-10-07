import pandas as pd
import torch
import numpy as np
import soundfile as sf
import os

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()
df=pd.DataFrame(np.load('./resources/working_data/data.npy',allow_pickle = True),columns=['video_id','start_time','mid_ts','label','audio'])

df['vggish']=''
for index,row in df.iterrows():
    file_name = './resources/working_data/'+f'block_{index}.wav'
    sf.write(file_name, row['audio'], 44100)
    vgg = model.forward(file_name)
    df['vggish'][index] = vgg
    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        print("The file does not exist")
    
    if index == 1:
        break

out = df.to_numpy()
np.save('../resources/working_data/data_with_vggish.npy', out)