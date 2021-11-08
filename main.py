# -*- coding: utf-8 -*-

from temp import trans_dict, stamp_list, text_list
from subtitle_transcription import SubtitleTranscription

sub_trans = SubtitleTranscription('战斗王EX_01.mp4')

stamp_list = sub_trans.get_subtitle_stamp()
# text_list = sub_trans.extract_subtitles(stamp_list)
# trans_dict = sub_trans.transcript_text(stamp_list, text_list, trans_dict)
