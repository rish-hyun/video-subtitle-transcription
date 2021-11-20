import cv2
import easyocr
import requests
import numpy as np
import pandas as pd
from time import sleep
from os.path import exists
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


COLUMNS = ['frame', 'ssim', 'mse', 'text', 'trans']


class SubtitleTranscription:

    def __init__(self, video_file) -> None:
        self.bbox = [0, 413, 640, 67]
        self.video_file = video_file
        self.df_file = video_file.replace('mp4', 'csv')

        self.vidcap = self.load_video(video_file)
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        self.w_frame = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h_frame = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

        self.df = self.load_df(self.df_file)

    # -------------------------------------------------------------------------------

    @staticmethod
    def load_df(file_name):
        if exists(file_name):
            return pd.read_csv(file_name)
        else:
            return pd.DataFrame(columns=COLUMNS)

    # -------------------------------------------------------------------------------

    def __del__(self):
        self.vidcap.release()
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------------

    def load_video(self, video_file):
        return cv2.VideoCapture(video_file)

    # -------------------------------------------------------------------------------

    def play_video(self, frame, delay=1):
        cv2.imshow(self.video_file, frame)
        cv2.waitKey(delay)

    # -------------------------------------------------------------------------------

    def save_df(self):
        self.df.to_csv(self.df_file, index=False, encoding='utf-8-sig')

    # -------------------------------------------------------------------------------

    def counter(self, frame_count, limit=None):
        if limit is None:
            limit = self.frame_count
        print(f'{round((frame_count/self.frame_count)*100, 2)}', end='\r')

    # -------------------------------------------------------------------------------

    def get_frames(self, raw=False):
        [x, y, w, h] = self.bbox
        self.vidcap = self.load_video(self.video_file)
        success, frame = self.vidcap.read()
        while success:
            if not raw:
                frame = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            yield frame
            success, frame = self.vidcap.read()

    # -------------------------------------------------------------------------------

    @staticmethod
    def get_contours(img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(img, kernel, iterations=6)
        thresh = cv2.threshold(dilate, 180, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            return max(contour_sizes, key=lambda x: x[0])[1]
        return None

    # -------------------------------------------------------------------------------

    def frame_scores(self, prev_frame, curr_frame):
        contour = self.get_contours(curr_frame)
        if contour is None:
            contour = self.get_contours(prev_frame)

        if contour is not None:
            [x, y, w, h] = cv2.boundingRect(contour)
            prev_frame = prev_frame[y:y+h, x:x+w]
            curr_frame = curr_frame[y:y+h, x:x+w]

        return {
            'ssim': round(ssim(prev_frame, curr_frame)*100, 2),
            'mse': round(mean_squared_error(prev_frame, curr_frame), 2),
            'contour': contour is not None
        }

    # -------------------------------------------------------------------------------

    def translate(self, text):
        api_token = "api_HEtcUvCvjXPLClCVJbaEwhaqbBmGoDYxFu"
        headers = {"Authorization": f"Bearer {api_token}"}
        model_id = 'Helsinki-NLP/opus-mt-zh-en'
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        payload = {"inputs": text}

        response = requests.post(url, headers=headers, json=payload).json()
        if isinstance(response, dict):
            # print(response)
            sleep(response['estimated_time'])
            return self.translate(text)
        else:
            return response[0]['translation_text']

    # -------------------------------------------------------------------------------

    def extract_video_info(self):
        frame_count = 1
        prev_frame = None
        reader = easyocr.Reader(['ch_sim'])
        trans_dict = self.df[['text', 'trans']].set_index('text').to_dict()['trans']

        for frame in self.get_frames():
            scores = self.frame_scores(prev_frame, frame) if prev_frame is not None else {}
    
            if frame_count not in self.df['frame'].values:
                text = ''
                if scores.get('contour', True):
                    text = ' '.join(reader.readtext(frame, detail=0)).strip()
                    if trans_dict.get(text) is None and len(text)>0:
                        trans_dict[text] = self.translate(text)                    

                self.df = self.df.append({
                    'frame': frame_count,
                    'mse': scores.get('mse'),
                    'ssim': scores.get('ssim'),
                    'text': text,
                    'trans': trans_dict.get(text)
                }, ignore_index=True)
                self.save_df()

            self.counter(frame_count)
            prev_frame = frame
            frame_count += 1

    # -------------------------------------------------------------------------------

    def transcript_text(self, file_name):
        [x, y, w, h] = self.bbox
        # y -= h

        thick = 1
        font_size = 0.9
        font_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(file_name, fourcc, self.fps, (self.w_frame, self.h_frame))

        frame_count = 1
        frame_text = zip(self.get_frames(raw=True), self.df['trans'].values)
        for frame,text in frame_text:
            if isinstance(text, str):
                (text_width, text_height) = cv2.getTextSize(text, font, font_size, thick)[0]
                loc_x = x + int(w/2) - int(text_width/2)
                loc_y = y + int(h/2) + int(text_height/2)
                mask = np.zeros((h, w), dtype=np.uint8)
                frame[y:y+h, x:x+w, :] = cv2.merge((mask, mask, mask))

                if text_width > w: # do something
                        ...
                frame = cv2.putText(frame, text, (loc_x, loc_y), font, font_size, font_color, thick, cv2.LINE_AA)

            self.counter(frame_count)
            # self.play_video(frame)
            output.write(frame)
            frame_count += 1

        output.release()

    # -------------------------------------------------------------------------------
