import cv2
import easyocr
import requests
import numpy as np
from time import sleep
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


class SubtitleTranscription:

    def __init__(self, video_file) -> None:
        self.bbox = [0, 413, 640, 67]
        self.video_file = video_file
        self.vidcap = self.load_video(video_file)

        self.w_frame = int(self.vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h_frame = int(self.vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    # -------------------------------------------------------------------------------

    def __del__(self):
        self.vidcap.release()
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------------

    def load_video(self, video_file):
        return cv2.VideoCapture(video_file)

    # -------------------------------------------------------------------------------

    def play_video(self, frame, speed=1):
        cv2.imshow('frame', frame)
        cv2.waitKey(speed)

    # -------------------------------------------------------------------------------

    def counter(self, frame_count, limit=None):
        if limit is None:
            limit = self.frame_count
        print(f'{round((frame_count/self.frame_count)*100, 2)}', end='\r')

    # -------------------------------------------------------------------------------

    def translate(self, text):
        api_token = "api_xxxxxxxxxxxxxxxxxxxxxxxxxxx"
        headers = {"Authorization": f"Bearer {api_token}"}
        model_id = 'Helsinki-NLP/opus-mt-zh-en'
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        payload = {"inputs": text}

        response = requests.post(url, headers=headers, json=payload).json()
        if isinstance(response, dict):
            print(response)
            sleep(response['estimated_time'])
            return self.translate(text)
        else:
            return response[0]['translation_text']

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

    def get_contours(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(img, kernel, iterations=6)
        thresh = cv2.threshold(dilate, 180, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            return max(contour_sizes, key=lambda x: x[0])[1]
        return None

    # -------------------------------------------------------------------------------

    def is_similar(self, prev_frame, curr_frame):
        contour = self.get_contours(curr_frame)

        is_empty_frame = True
        if contour is None:
            is_empty_frame = False
            contour = self.get_contours(prev_frame)

        if contour is not None:
            [x, y, w, h] = cv2.boundingRect(contour)
            prev_frame = prev_frame[y:y+h, x:x+w]
            curr_frame = curr_frame[y:y+h, x:x+w]

        mse_score = int(mean_squared_error(prev_frame, curr_frame))
        ssim_score = int(ssim(prev_frame, curr_frame)*100)

        similar = True
        if ssim_score < 70 and mse_score > 2000:
            similar = False

        return similar, [ssim_score, mse_score], is_empty_frame

    # -------------------------------------------------------------------------------

    def get_subtitle_stamp(self):
        stamp = []
        prev_frame = None
        start = frame_count = 1

        for frame in self.get_frames():
            if prev_frame is not None:
                similar, score, is_empty_frame = self.is_similar(prev_frame, frame)

                if not similar:
                    if not is_empty_frame:
                        stamp.append((start, frame_count-1))
                    start = frame_count
                    cv2.imwrite(f'temp/{frame_count}_{str(score)}.jpg', frame)

            self.counter(frame_count)
            prev_frame = frame
            frame_count += 1
        return stamp

    # -------------------------------------------------------------------------------

    def extract_subtitles(self, stamp):
        text = []
        frame_count = 1
        reader = easyocr.Reader(['ch_sim'])
        stamp_list = [int((start+end)/2) for (start, end) in stamp]

        for frame in self.get_frames():
            if frame_count in stamp_list:
                text.append(reader.readtext(frame, detail=0))
                print(text[-1])

            self.counter(frame_count)
            frame_count += 1
        return text

    # -------------------------------------------------------------------------------

    def transcript_text(self, stamp_list, text_list, trans_dict={}):
        [x, y, w, h] = self.bbox
        y -= h

        thick = 1
        font_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        file_name = self.video_file.replace('.', '_edit.')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(file_name, fourcc, self.fps, (self.w_frame, self.h_frame))

        frame_count = 1

        for frame in self.get_frames(raw=True):
            for index, (start, end) in enumerate(stamp_list):

                if frame_count in range(start, end+1):
                    text = ' '.join(text_list[index])
                    if trans_dict.get(text) is None:
                        trans_dict[text] = self.translate(text)
                    text = trans_dict[text]

                    font_size = 0.9
                    (text_width, text_height) = cv2.getTextSize(text, font, font_size, thick)[0]
                    loc_x = x + int(w/2) - int(text_width/2)
                    loc_y = y + int(h/2) + int(text_height/2)

                    if text_width > w:
                        text = 'LENGTH EXCEEDED'

                    mask = np.zeros((h, w), dtype=np.uint8)
                    frame[y:y+h, x:x+w, :] = cv2.merge((mask, mask, mask))
                    frame = cv2.putText(frame, text, (loc_x, loc_y), font, font_size, font_color, thick, cv2.LINE_AA)
                    break

            self.counter(frame_count)
            output.write(frame)
            frame_count += 1

        output.release()
        return trans_dict

    # -------------------------------------------------------------------------------
