"""
Convert SRT subtitle files to structured JSON format.

Usage:
    python srt_to_json.py <srt_file> [--video-name <name>] [--output <output_file>]

Example:
    python srt_to_json.py test-file.srt --video-name "Bai Giang 01.mp3"
    python srt_to_json.py test-file.srt -o output.json
"""

import re
import json
import argparse
from pathlib import Path


def parse_timestamp(timestamp: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds as float."""
    # SRT uses comma for milliseconds, normalize to dot
    timestamp = timestamp.strip().replace(",", ".")
    match = re.match(r"(\d+):(\d+):(\d+)\.(\d+)", timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    hours, minutes, seconds, millis = match.groups()
    total = (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )
    return round(total, 3)


def clean_quotes(text: str) -> str:
    """Replace straight double quotes with Unicode smart quotes to avoid
    excessive escaping in JSON output."""
    # Match pairs of straight quotes and replace with smart quotes
    result = []
    open_quote = True
    for char in text:
        if char == '"':
            result.append('\u201c' if open_quote else '\u201d')
            open_quote = not open_quote
        else:
            result.append(char)
    return ''.join(result)


def parse_srt(srt_content: str) -> list[dict]:
    """Parse SRT content into a list of segment dicts."""
    segments = []
    
    # Split by double newline to get blocks (handle both \r\n and \n)
    srt_content = srt_content.replace("\r\n", "\n")
    blocks = re.split(r"\n\n+", srt_content.strip())
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        
        # First line: segment number
        # Second line: timestamps
        # Remaining lines: text
        
        # Find the timestamp line (format: HH:MM:SS,mmm --> HH:MM:SS,mmm)
        timestamp_pattern = r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})"
        
        timestamp_line_idx = None
        for i, line in enumerate(lines):
            if re.search(timestamp_pattern, line):
                timestamp_line_idx = i
                break
        
        if timestamp_line_idx is None:
            continue
        
        match = re.search(timestamp_pattern, lines[timestamp_line_idx])
        start_time = parse_timestamp(match.group(1))
        end_time = parse_timestamp(match.group(2))
        
        # Text is everything after the timestamp line
        text_lines = lines[timestamp_line_idx + 1:]
        text = " ".join(line.strip() for line in text_lines if line.strip())
        
        # Remove HTML tags that sometimes appear in SRT
        text = re.sub(r"<[^>]+>", "", text)
        
        if text:  # Only add non-empty segments
            segments.append({
                "segment_id": len(segments),
                "start": start_time,
                "end": end_time,
                "text": clean_quotes(text),
            })
    
    return segments


def srt_to_json(
    srt_path: str,
    video_name: str | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Convert an SRT file to the structured JSON format.
    
    Args:
        srt_path: Path to the .srt file
        video_name: Name of the source video/audio file
        output_path: Optional path for the output JSON file
    
    Returns:
        The structured JSON dict
    """
    srt_file = Path(srt_path)
    
    if not srt_file.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
    
    # Read with UTF-8 (fallback to utf-8-sig for BOM)
    try:
        content = srt_file.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = srt_file.read_text(encoding="latin-1")
    
    segments = parse_srt(content)
    
    # Build full text from all segments
    full_text = " ".join(seg["text"] for seg in segments)
    
    # Use filename as video_name if not provided
    if video_name is None:
        video_name = srt_file.stem  # filename without extension
    
    result = {
        "video_name": video_name,
        "full_text": full_text,
        "total_segments": len(segments),
        "segments": segments,
    }
    
    # Determine output path
    if output_path is None:
        output_path = srt_file.with_suffix(".json")
    
    # Write JSON
    output_file = Path(output_path)
    output_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    print(f"✅ Converted successfully!")
    print(f"   Input:  {srt_file}")
    print(f"   Output: {output_file}")
    print(f"   Video:  {video_name}")
    print(f"   Segments: {len(segments)}")
    print(f"   Full text length: {len(full_text)} characters")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert SRT subtitle files to structured JSON format"
    )
    parser.add_argument("srt_file", help="Path to the .srt file")
    parser.add_argument(
        "--video-name", "-n",
        help="Name of the source video/audio file (default: SRT filename)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path (default: same name as SRT with .json extension)",
    )
    
    args = parser.parse_args()
    srt_to_json(args.srt_file, args.video_name, args.output)


if __name__ == "__main__":
    main()
