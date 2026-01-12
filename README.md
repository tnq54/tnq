# AutoDubber - Công Cụ Lòng Tiếng/Thuyết Minh Tự Động

Công cụ CLI bằng Python giúp tự động tạo thuyết minh hoặc lồng tiếng cho video, hỗ trợ nhiều ngôn ngữ và phong cách phim bộ TVB.

## Tính Năng

*   **Transcription:** Tự động tạo phụ đề từ video gốc (sử dụng OpenAI Whisper).
*   **Translation:** Dịch phụ đề sang tiếng Việt hoặc ngôn ngữ khác.
*   **Text-to-Speech (TTS):** Tạo giọng đọc AI chất lượng cao (Edge-TTS).
    *   Hỗ trợ tinh chỉnh tốc độ và cao độ để tạo hiệu ứng "TVB" (kịch tính, nhanh).
*   **Mixing:** Tự động ghép âm thanh vào video, hỗ trợ giảm âm lượng video gốc (Thuyết minh) hoặc tắt tiếng gốc (Lồng tiếng).

## Yêu cầu cài đặt

1.  Python 3.8+
2.  Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

*Lưu ý: Lần đầu chạy sẽ cần tải model Whisper (~140MB - 3GB tùy model).*

## Cách sử dụng

Chạy tool thông qua dòng lệnh:

```bash
python main.py --input <đường_dẫn_video> [options]
```

### Các tùy chọn (Options):

*   `--input`: Đường dẫn file video đầu vào (Bắt buộc).
*   `--output`: Tên file video đầu ra (Mặc định: output.mp4).
*   `--source_lang`: Ngôn ngữ gốc của video (Mặc định: auto - tự động nhận diện).
*   `--target_lang`: Ngôn ngữ muốn thuyết minh (Mặc định: vi - Tiếng Việt).
*   `--style`: Phong cách giọng đọc.
    *   `normal`: Bình thường.
    *   `tvb`: Nhanh hơn, cao hơn một chút, giống phim bộ.
*   `--mode`: Chế độ âm thanh.
    *   `voiceover`: Giữ âm thanh gốc nhưng nhỏ đi (Thuyết minh).
    *   `dubbing`: Tắt hoàn toàn âm thanh gốc (Lồng tiếng).

### Ví dụ

**Thuyết minh phim bộ Hong Kong (TVB Style):**

```bash
python main.py --input tap1.mp4 --style tvb --mode voiceover
```

**Lồng tiếng video hướng dẫn tiếng Anh sang tiếng Việt:**

```bash
python main.py --input tutorial.mp4 --source_lang en --target_lang vi --mode dubbing
```

## Cấu trúc thư mục

*   `src/`: Mã nguồn chính.
*   `main.py`: File chạy chính.
*   `requirements.txt`: Các thư viện cần thiết.
