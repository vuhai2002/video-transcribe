from realign.pair_audio_srt import (
    normalize_key, build_pairs, format_unpaired_report,
)


def test_normalize_key_strips_diacritics_and_lowercases():
    assert normalize_key("Đạo làm con B") == "dao lam con b"


def test_normalize_key_collapses_whitespace():
    assert normalize_key("  Tình   yêu  Tổ quốc ") == "tinh yeu to quoc"


def test_build_pairs_matches_case_and_diacritic_insensitive(tmp_path):
    srt_dir = tmp_path / "srt"; srt_dir.mkdir()
    audio = tmp_path / "a1"; audio.mkdir()
    (srt_dir / "Đạo làm con B.srt").write_text("x", encoding="utf-8")
    (srt_dir / "Khong co audio.srt").write_text("x", encoding="utf-8")
    (audio / "đạo làm con b.mp3").write_bytes(b"")     # khac hoa/thuong + dau van match
    (audio / "Bai le thua.mp3").write_bytes(b"")       # khong co srt -> unpaired_audio

    pairs, unpaired_srt, unpaired_audio = build_pairs(str(srt_dir), [str(audio)])

    assert len(pairs) == 1
    assert pairs[0]["key"] == "dao lam con b"
    assert pairs[0]["srt_name"] == "Đạo làm con B.srt"
    assert pairs[0]["audio_path"].endswith("đạo làm con b.mp3")
    assert [u["srt_name"] for u in unpaired_srt] == ["Khong co audio.srt"]
    assert [u["audio_name"] for u in unpaired_audio] == ["Bai le thua.mp3"]


def test_format_unpaired_report_lists_both():
    txt = format_unpaired_report(
        [{"srt_name": "A.srt", "key": "a"}],
        [{"audio_name": "B.mp3", "key": "b"}],
    )
    assert "A.srt" in txt and "B.mp3" in txt
