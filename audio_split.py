from pydub import AudioSegment
from pydub.utils import make_chunks
import os

def audio_split(input_audio, size):
    '''
    receive the input audio and slice length
    split the audio
    and put the segmentation in one folder
    '''

    audio = AudioSegment.from_file(input_audio, "wav")

    chunks = make_chunks(audio, size)

    dir_name = input_audio+"_segmentation"
    os.mkdir(dir_name)

    for i, chunk in enumerate(chunks):
        seg_name = "seg-{0}.wav".format(i+1)
        print(seg_name)
        chunk.export(dir_name+"/"+seg_name, format="wav")    

if __name__=='__main__':
    print("Please input the file name and length of slice you want to split (unit:ms):")
    input_audio, size = input().split()
    audio_split(input_audio=input_audio, size=int(size))

