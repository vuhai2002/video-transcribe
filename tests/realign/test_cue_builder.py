import realign.config as cfg
from realign.text_metrics import char_count
from realign.cue_builder import fill_word_times, build_cues


def W(w, s, e, score=0.9):
    return {"w": w, "start": s, "end": e, "score": score}


def test_fill_word_times_interpolates_none():
    out = fill_word_times([W("a", 0.0, 0.5), W("b", None, None), W("c", 1.0, 1.5)])
    assert out[1]["start"] == 0.5 and out[1]["end"] >= 0.5


def test_sentence_split_when_each_cue_meets_min_display():
    # 2 câu, mỗi cue đủ dài (>=1.5s) -> KHÔNG gộp -> 2 cue
    cues = build_cues([W("Một.", 0.0, 2.0), W("Hai.", 2.1, 4.0)], cfg)
    assert len(cues) == 2
    assert cues[0]["text"] == "Một." and cues[1]["text"] == "Hai."


def test_long_pause_splits_into_two_cues():
    # ngừng DÀI 2.4s >= LONG_PAUSE -> tách cứng
    cues = build_cues([W("alpha", 0.0, 1.6), W("beta", 4.0, 5.6)], cfg)
    assert len(cues) == 2


def test_medium_pause_does_not_split_short_sentence():
    # ngừng vừa 0.8s (>= PAUSE_SPLIT, < LONG_PAUSE) KHÔNG tách câu ngắn
    words = [W("alpha", 0.0, 0.4), W("beta", 0.5, 0.9),
             W("gamma", 1.7, 2.1), W("delta", 2.2, 2.6)]   # gap beta->gamma = 0.8s
    cues = build_cues(words, cfg)
    assert len(cues) == 1


def test_deisolate_leading_misaligned_word():
    # tu dau "Nam" neo nham 8.3s, than o 54s (gap 44s) -> gop lai, dung timing than
    body = ["Mô", "Bổn", "Sư", "Phật."]
    words = [W("Nam", 8.3, 9.8)] + [W(w, 54.0 + i * 0.2, 54.0 + i * 0.2 + 0.15)
                                    for i, w in enumerate(body)]
    cues = build_cues(words, cfg)
    assert len(cues) == 1
    assert cues[0]["text"].replace("\n", " ").startswith("Nam Mô")
    assert cues[0]["start"] >= 50.0


def test_long_run_splits_to_keep_lines_within_cpl():
    words = [W(f"w{i:02d}", i * 1.0, i * 1.0 + 0.8) for i in range(40)]  # buộc ngắt để giữ <= 2 dòng <= CPL
    cues = build_cues(words, cfg)
    assert len(cues) >= 2
    for c in cues:
        assert all(char_count(ln) <= cfg.CPL_MAX for ln in c["text"].split("\n"))


def test_cps_floor_extends_short_cue_into_gap():
    # 1 cue ~60 ký tự cần >= 60/15 = 4.0s; audio chỉ 1s; có gap dài phía sau -> kéo dài
    text_words = ["chu" + str(i) for i in range(12)]  # ~ 12*5+11 = 71 ký tự, 1 cue
    words = [W(w, 0.0 + i * 0.08, 0.0 + i * 0.08 + 0.07) for i, w in enumerate(text_words)]
    # kết thúc ~0.96s, không câu/pause nội bộ -> 1 cue; không có cue sau -> cap = start+DUR_MAX
    cues = build_cues(words, cfg)
    assert len(cues) == 1
    dur = cues[0]["end"] - cues[0]["start"]
    chars = char_count(cues[0]["text"])
    assert dur >= chars / cfg.CPS_MAX - 1e-6   # đạt trần tốc độ đọc


def test_tiny_back_to_back_cue_merged_no_flash():
    # câu ngắn "Vâng." 0.4s, ngay sau là câu dài, KHÔNG có gap -> gộp, không để cue < 1.5s
    words = [W("Vâng.", 0.0, 0.4)] + [W(f"x{i}", 0.45 + i * 0.5, 0.45 + i * 0.5 + 0.45) for i in range(5)]
    cues = build_cues(words, cfg)
    for c in cues:
        assert (c["end"] - c["start"]) >= cfg.DUR_MIN - 1e-6   # không còn cue chớp


def test_merge_never_creates_over_cpl_line():
    # cue vụn "Vâng." ngay trước 1 câu dài, không gap -> không được gộp thành cue > CPL
    words = [W("Vâng.", 0.0, 0.05)] + [W("abc", 0.1 + i * 0.5, 0.1 + i * 0.5 + 0.45) for i in range(20)]
    cues = build_cues(words, cfg)
    for c in cues:
        assert all(char_count(ln) <= cfg.CPL_MAX for ln in c["text"].split("\n"))


def test_single_overlong_word_does_not_crash():
    cues = build_cues([W("x" * 50, 0.0, 3.0)], cfg)
    assert len(cues) == 1


def test_no_words_returns_empty():
    assert build_cues([], cfg) == []


def test_long_sentence_splits_balanced_at_comma():
    text = "Hôm nay chúng ta nghe một đoạn pháp cú ngắn thôi, rồi sau đó chúng ta nói qua chuyện khác."
    toks = text.split()
    words = [W(t, i * 0.4, i * 0.4 + 0.35) for i, t in enumerate(toks)]   # liên tục, không pause
    cues = build_cues(words, cfg)
    assert len(cues) == 2
    assert cues[0]["text"].replace("\n", " ").rstrip().endswith(",")      # ngắt ở dấu phẩy
    for c in cues:                                                        # không đuôi cụt
        assert len(c["text"].replace("\n", " ").split()) >= 3
