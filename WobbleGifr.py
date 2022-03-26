import os
import matplotlib
import moviepy.editor as mp
import librosa
import time
import math
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

class Wobbley:
    '''Class of functions for time synchronizing and trimming video files based on cross correlaiton of their audio.'''
    
    def __init__(self):
        '''Initialize VideoSynchTrimmingClass'''
        pass

    def get_clip_list(self, base_path, file_type):
        '''Return a list of all video files in the base_path folder that match the given file type.'''

        # change directory to folder containing videos
        os.chdir(base_path)

        # create general search from file type to use in glob search, including cases for upper and lowercase file types
        file_extension_upper = '*' + file_type.upper()
        file_extension_lower = '*' + file_type.lower()
    
        # make list of all files with file type
        clip_list = glob(file_extension_upper) + glob(file_extension_lower) #if two capitalization standards are used, the videos may not be in original order

        return clip_list

    def get_files(self, base_path, clip_list):
        '''Get video files from clip_list, return a list of lists containing the video file name, file, and length in frames. 
        Also return a list containing the audio sample rates of each video.'''
        
        # create empty list for storing audio and video files, will contain sublists formatted like [video_file_name,video_file,vid_length] 
        file_list = []

        # create empty list to hold audio sample rate, so we can verify samplerate is the same across all audio files
        sample_rate_list = []

        # iterate through clip_list, open video files and audio files, and store in file_list
        for clip in clip_list:
            audio_name = clip.split(".")[0] + '.wav'

            # open video file
            video_file = mp.VideoFileClip(os.path.join(base_path,clip), audio=True)

            # get length of video clip
            vid_length = math.floor(video_file.duration) #use floor to convert to integer and ensure in bounds value for later range function

            # create .wav file of clip audio
            video_file.audio.write_audiofile(os.path.join(base_path,audio_name))

            # extract raw audio from Wav file
            audio_signal, audio_rate = librosa.load(audio_name, sr = None)
            sample_rate_list.append(audio_rate)
            
            # save video file name, file, and duration in list
            file_list.append([clip, video_file, vid_length, audio_name, audio_signal])


        return file_list, sample_rate_list

    def get_fps_list(self, file_list):
        '''Retrieve frames per second of each video clip in file_list'''
        return [file[1].fps for file in file_list]

    def check_rates(self, rate_list):
        '''Check if audio sample rates or audio frame rates are equal, throw an exception if not (or if no rates are given).'''
        if len(rate_list) == 0:
            raise Exception("no rates given")
        else:
            if rate_list.count(rate_list[0]) == len(rate_list):
                print("all rates are equal to", rate_list[0])
                return rate_list[0]
            else:
                raise Exception("rates are not equal")

    def normalize_audio(self, audio_file):
        '''Perform z-score normalization on an audio file and return the normalized audio file - this is best practice for correlating.'''
        return ((audio_file - np.mean(audio_file))/np.std(audio_file - np.mean(audio_file)))

    def cross_correlate(self, audio1, audio2):
        '''Take two audio files, sync them using cross correlation, and trim them to the same length.
        Inputs are two WAV files to be synced. Return the lag expressed in terms of the audio sample rate of the clips'''

        # compute cross correlation with scipy correlate function, which gives the correlation of every different lag value
        # mode='full' makes sure every lag value possible between the two signals is used, and method='fft' uses the fast fourier transform to speed the process up 
        corr = signal.correlate(audio1, audio2, mode='full', method='fft')
        # lags gives the amount of time shift used at each index, corresponding to the index of the correlate output list
        lags = signal.correlation_lags(audio1.size, audio2.size, mode="full")
        # lag is the time shift used at the point of maximum correlation - this is the key value used for shifting our audio/video
        lag = lags[np.argmax(corr)]
    
        print("lag:", lag)

        return lag

    def find_lags(self, file_list, sample_rate):
        '''Take a file list containing video and audio files, as well as the sample rate of the audio, cross correlate the audio files, and output a lag list.
        The lag list is normalized so that the lag of the latest video to start in time is 0, and all other lags are positive'''
        
        lag_list = [self.cross_correlate(file_list[0][4],file[4])/sample_rate for file in file_list] # cross correlates all audio to the first audio file in the list
        #also divides by the audio sample rate in order to get the lag in seconds

        #now that we have our lag array, we subtract every value in the array from the max value
        #this creates a normalized lag array where the latest video has lag of 0
        #the max value lag represents the latest video - thanks Oliver for figuring this out
        norm_lag_list = [(max(lag_list) - value) for value in lag_list]
   
        print("original lag list: ", lag_list, "normalized lag list: ", norm_lag_list)
        # plot lags before and after to make visualization that this is doing what we want
        return norm_lag_list

    def trim_videos(self, file_list, lag_list):
        # this takes a list of video files and a list of lags, and shortens the beginning of the video by the lags, and trims the ends so they're all the same length
        front_trimmed_videos = []

        # for each video in the list, create a new video trimmed from the begining by the lag value for that video, and add it to the empty list
        for i in range(len(file_list)):
            print(file_list[i][1])
            front_trimmed_video = file_list[i][1].subclip(lag_list[i],file_list[i][1].duration)
            front_trimmed_videos.append([file_list[i][0], front_trimmed_video])
        
        print(front_trimmed_videos)

        # now we find the duration of each video and add it to a list to find the shortest video duration
        min_duration = min([video[1].duration for video in front_trimmed_videos])

        # create list to store names of final videos
        fully_trimmed_videos = []
        # trim all videos to length of shortest video, and give it a new name
        for video in front_trimmed_videos:
            fully_trimmed_video = video[1].subclip(0,min_duration)
            video_name = "synced_" + video[0]
            fully_trimmed_videos.append([video_name, fully_trimmed_video, min_duration]) 

        return fully_trimmed_videos # return video list

    def pick_random_frame(self, file_list, fps):
        '''Pick a random frame number from a valid range of frames from video in file_list.'''

        # converts duration into frames, picks a random frame, and converts back to seconds
        return np.random.choice((range(math.floor(file_list[0][2]*fps))))/fps

    def pick_frame(self, file_list, fps):
        '''Allow user to pick frame number from valid range to construct gif from.'''

        pass

    def get_frames(self, file_list, frame_number):
        '''Get frames from videos in file list corresponding to the frame number.'''
        
        return [[file[0], file[1].get_frame(frame_number)] for file in file_list]

    def display_images(self, image_list):
        '''Display images from list in one plot.'''

        # make figure
        fig = plt.figure(figsize=(10,7)) 

        for i in range(len(image_list)):
            fig.add_subplot(len(image_list), 1, i+1) #adds the i+1th subplot to a structure of 1 column with as many rows are in image_list
            plt.imshow(image_list[i][1]) #plot the image
            plt.title(image_list[i][0]) #add title

        fig.tight_layout()
        #plt.savefig('imageplots.png', bbox_inches='tight')
        plt.show()

    def construct_gif(self, image_list):
        '''Make gif from images in image list.'''

        # get just a list of images for the final step
        images = [image[1] for image in image_list]
        
        #we also want to add the middle images in reverse to the gif, so that it wobbles smoothly back and forth
        images_reverse = [image_list[i][1] for i in range(1, (len(image_list)-1))]
        
        for im in images_reverse:
            images.append(im)
        
        clip = mp.ImageSequenceClip(images, fps = 15)

        clip.write_gif('wobble.gif')
        clip.close()


def main():
    '''Create a wobble gif from the videos in the base path folder'''

    # start timer to measure performance
    start_timer = time.time()

    # instantiate class
    wobble = Wobbley()
    wobble # this may be unnecessary?

    # set the base path and file type
    base = '/Users/Philip/Documents/Projects/wobbleGifr'  # this is the folder containing your videos to sync
    file_type = "MP4"  # should work with or without a period at the front, and in upper or lower case
    
    # create list of video clip in base path folder
    clip_list = wobble.get_clip_list(base, file_type)

    # get the files and store in list
    files, sr = wobble.get_files(base, clip_list)

    # find the frames per second of each video
    fps_list = wobble.get_fps_list(files)
    
    # check that our frame rates and audio sample rates are all equal
    fps = wobble.check_rates(fps_list)
    wobble.check_rates(sr)
    
    # find the lags
    lag_list = wobble.find_lags(files, wobble.check_rates(sr))
    
    # use lags to trim the videos
    trimmed_videos = wobble.trim_videos(files, lag_list)

    # choose a random frame to get from 
    random_frame = wobble.pick_random_frame(trimmed_videos, fps)

    # get frames from the trimmed videos
    frame_list = wobble.get_frames(trimmed_videos, random_frame)

    # plot images that will be used in gif
    #wobble.display_images(frame_list)

    # construct the gif and save it to folder
    wobble.construct_gif(frame_list)

    # end performance timer
    end_timer = time.time()
    
    #calculate and display elapsed processing time
    elapsed_time = end_timer - start_timer
    print("elapsed processing time in seconds:", elapsed_time)


if __name__ == "__main__":
    main()