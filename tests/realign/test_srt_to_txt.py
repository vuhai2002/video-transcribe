from realign.srt_to_txt import parse_srt_cues, srt_to_transcript

SRT = """1
00:00:01,000 --> 00:00:03,000
Dung the.

2
00:00:03,200 --> 00:00:06,000
Toi nghi
ban sai roi?
"""

def test_parse_basic_two_cues():
    assert parse_srt_cues(SRT) == ["Dung the.", "Toi nghi ban sai roi?"]

def test_multiline_cue_joined_with_space():
    assert parse_srt_cues(SRT)[1] == "Toi nghi ban sai roi?"

def test_preserves_vietnamese_diacritics_and_punctuation():
    srt = "1\n00:00:00,000 --> 00:00:02,000\nKhong the chinh sua: Đúng chưa?\n"
    assert parse_srt_cues(srt) == ["Khong the chinh sua: Đúng chưa?"]

def test_handles_crlf_and_bom():
    srt = "﻿1\r\n00:00:00,000 --> 00:00:01,000\r\nHello\r\n"
    assert parse_srt_cues(srt) == ["Hello"]

def test_transcript_joins_cues_with_newline():
    assert srt_to_transcript(SRT) == "Dung the.\nToi nghi ban sai roi?"
