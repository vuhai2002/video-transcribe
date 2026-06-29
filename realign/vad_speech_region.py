"""silero-vad: xác định khoảng có tiếng nói [first_speech, last_speech] để cắt nhạc/tụng intro-outro.

Các hàm:
- region_from_timestamps: hàm thuần, unit-testable, không cần model.
- load_vad_model: load silero-vad một lần, tái dùng cho nhiều file.
- speech_region: decode audio bằng load_audio (soundfile), chạy VAD, trả về (first, last) hoặc None.
"""
import realign.config as config


def region_from_timestamps(ts: list[dict], sr: int) -> tuple[float, float] | None:
    """Chuyển danh sách VAD timestamps [{start, end}] (đơn vị sample) -> (first_sec, last_sec).

    Trả về None nếu danh sách rỗng (không có tiếng nói nào được phát hiện).
    first_sec = start của segment đầu tiên / sr.
    last_sec  = end   của segment cuối cùng / sr.
    """
    if not ts:
        return None
    return (ts[0]["start"] / sr, ts[-1]["end"] / sr)


def load_vad_model():
    """Load silero-vad model một lần để tái dùng qua nhiều file (tránh tốc độ khởi tạo lại).

    Trả về model đã sẵn sàng cho get_speech_timestamps.
    """
    from silero_vad import load_silero_vad
    return load_silero_vad()


def speech_region(
    audio_path: str,
    sr: int = config.SAMPLE_RATE,
    model=None,
) -> tuple[float, float] | None:
    """Tìm khoảng có tiếng nói trong file audio bằng silero-vad.

    - Decode audio bằng realign.align_words.load_audio (soundfile, tránh torchcodec).
    - Chạy silero get_speech_timestamps trên waveform 1D.
    - Trả về (first_speech_sec, last_speech_sec) hoặc None nếu im lặng hoàn toàn.

    model: truyền vào để tái sử dụng; nếu None thì tự load một lần.
    """
    from silero_vad import get_speech_timestamps
    from realign.align_words import load_audio

    model = model or load_vad_model()
    wav, _ = load_audio(audio_path, sr)
    ts = get_speech_timestamps(wav.squeeze(0), model, sampling_rate=sr)
    return region_from_timestamps(ts, sr)
