"""Tests for caption_directory feature in image_video_dataset.py.

These tests verify:
- Backward compatibility when caption_directory is not specified
- Separate caption directory functionality
- Multi-dot extension handling
- Warning and error behaviors for caption filtering
- Duplicate basename detection
- Config schema validation
"""

import logging
import os
import tempfile
import unittest

from musubi_tuner.dataset.image_video_dataset import (
    ImageDirectoryDatasource,
    ImageJsonlDatasource,
    VideoDirectoryDatasource,
    VideoJsonlDatasource,
    _check_duplicate_basenames,
    _filter_paths_by_caption,
    logger,
)


class LogCapture(logging.Handler):
    """Handler that captures log records into a list for assertions."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def get_messages(self, level: int = None) -> list[str]:
        if level is None:
            return [r.getMessage() for r in self.records]
        return [r.getMessage() for r in self.records if r.levelno == level]

    def clear(self):
        self.records.clear()


class TestCaptionDirectoryHelpers(unittest.TestCase):
    """Tests for helper functions directly."""

    def test_check_duplicate_basenames_no_duplicates(self):
        """No error when all basenames are unique."""
        paths = ["/dir/a.png", "/dir/b.jpg", "/dir/c.webp"]
        # Should not raise
        _check_duplicate_basenames(paths, kind="image")

    def test_check_duplicate_basenames_with_duplicates(self):
        """Error when duplicate basenames exist."""
        paths = ["/dir/a.png", "/dir/a.jpg", "/dir/b.png"]
        with self.assertRaises(ValueError) as ctx:
            _check_duplicate_basenames(paths, kind="image")
        self.assertIn("Duplicate", str(ctx.exception))
        self.assertIn("a", str(ctx.exception))

    def test_filter_paths_empty_list(self):
        """Empty paths list returns empty list without error."""
        result = _filter_paths_by_caption([], ".txt", "/captions", "/images", "image")
        self.assertEqual(result, [])


class TestImageDirectoryDatasource(unittest.TestCase):
    """Tests for ImageDirectoryDatasource with caption_directory."""

    def setUp(self):
        """Set up log capture and temp directories."""
        self.log_capture = LogCapture()
        self.log_capture.setLevel(logging.DEBUG)
        logger.logger.addHandler(self.log_capture)
        self.temp_dirs = []

    def tearDown(self):
        """Clean up log capture and temp directories."""
        logger.logger.removeHandler(self.log_capture)
        # Clean up temp directories
        import shutil

        for d in self.temp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

    def create_temp_dir(self):
        """Create a temp directory and track it for cleanup."""
        d = tempfile.mkdtemp()
        self.temp_dirs.append(d)
        return d

    def create_image(self, directory: str, name: str):
        """Create a minimal PNG file."""
        # Create a 1x1 pixel PNG
        from PIL import Image

        img = Image.new("RGB", (1, 1), color="red")
        path = os.path.join(directory, name)
        img.save(path)
        return path

    def create_caption(self, directory: str, name: str, content: str = "test caption"):
        """Create a caption file."""
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def assertWarningCount(self, expected: int):
        warnings = self.log_capture.get_messages(logging.WARNING)
        self.assertEqual(len(warnings), expected, f"Expected {expected} warnings, got {len(warnings)}: {warnings}")

    def assertWarningContains(self, *substrings):
        warnings = self.log_capture.get_messages(logging.WARNING)
        combined = " ".join(warnings)
        for s in substrings:
            self.assertIn(s, combined, f"'{s}' not found in warnings: {warnings}")

    def test_backward_compat_no_caption_directory(self):
        """When caption_directory is not specified, captions come from image_directory."""
        image_dir = self.create_temp_dir()
        self.create_image(image_dir, "a.png")
        self.create_caption(image_dir, "a.txt", "caption for a")

        ds = ImageDirectoryDatasource(image_dir, caption_extension=".txt")

        self.assertEqual(len(ds.image_paths), 1)
        path, caption = ds.get_caption(0)
        self.assertEqual(caption, "caption for a")
        self.assertEqual(ds.caption_directory, image_dir)

    def test_separate_caption_directory(self):
        """Captions can come from a separate directory."""
        image_dir = self.create_temp_dir()
        caption_dir = self.create_temp_dir()

        self.create_image(image_dir, "a.png")
        self.create_caption(caption_dir, "a.txt", "caption from separate dir")

        ds = ImageDirectoryDatasource(image_dir, caption_extension=".txt", caption_directory=caption_dir)

        self.assertEqual(len(ds.image_paths), 1)
        path, caption = ds.get_caption(0)
        self.assertEqual(caption, "caption from separate dir")

    def test_multi_dot_caption_extension(self):
        """Multi-dot caption extensions like .flux2.txt work correctly."""
        image_dir = self.create_temp_dir()
        caption_dir = self.create_temp_dir()

        self.create_image(image_dir, "a.png")
        self.create_caption(caption_dir, "a.flux2.txt", "flux2 style caption")

        ds = ImageDirectoryDatasource(image_dir, caption_extension=".flux2.txt", caption_directory=caption_dir)

        self.assertEqual(len(ds.image_paths), 1)
        path, caption = ds.get_caption(0)
        self.assertEqual(caption, "flux2 style caption")

    def test_multi_dot_media_name(self):
        """Media files with multi-dot names like foo.bar.png work correctly."""
        image_dir = self.create_temp_dir()

        self.create_image(image_dir, "foo.bar.png")
        self.create_caption(image_dir, "foo.bar.txt", "caption for foo.bar")

        ds = ImageDirectoryDatasource(image_dir, caption_extension=".txt")

        self.assertEqual(len(ds.image_paths), 1)
        path, caption = ds.get_caption(0)
        self.assertEqual(caption, "caption for foo.bar")

    def test_warning_with_truncation(self):
        """Warning is emitted when some images are filtered, with truncation for large lists."""
        image_dir = self.create_temp_dir()
        caption_dir = self.create_temp_dir()

        # Create 30 images, only 5 have captions
        for i in range(30):
            self.create_image(image_dir, f"img{i:03d}.png")
        for i in range(5):
            self.create_caption(caption_dir, f"img{i:03d}.txt", f"caption {i}")

        ds = ImageDirectoryDatasource(image_dir, caption_extension=".txt", caption_directory=caption_dir)

        self.assertEqual(len(ds.image_paths), 5)
        self.assertWarningCount(1)
        self.assertWarningContains("25/30", "caption_extension=", "caption_directory=", "and 5 more")

    def test_zero_matches_hard_error(self):
        """Hard error when images exist but no captions match."""
        image_dir = self.create_temp_dir()
        caption_dir = self.create_temp_dir()

        self.create_image(image_dir, "a.png")
        self.create_image(image_dir, "b.png")
        # No caption files created

        with self.assertRaises(ValueError) as ctx:
            ImageDirectoryDatasource(image_dir, caption_extension=".txt", caption_directory=caption_dir)

        self.assertIn("No images with matching caption files found", str(ctx.exception))
        self.assertIn("Found 2 image(s)", str(ctx.exception))

    def test_zero_media_no_error(self):
        """Empty image directory doesn't cause caption-related error."""
        image_dir = self.create_temp_dir()
        caption_dir = self.create_temp_dir()

        # No images, just a caption directory (also empty)
        ds = ImageDirectoryDatasource(image_dir, caption_extension=".txt", caption_directory=caption_dir)

        self.assertEqual(len(ds.image_paths), 0)
        # No caption-related errors or warnings expected (just "found 0 images" info)

    def test_duplicate_basename_hard_error(self):
        """Hard error when duplicate basenames exist."""
        image_dir = self.create_temp_dir()

        self.create_image(image_dir, "a.png")
        self.create_image(image_dir, "a.jpg")
        self.create_caption(image_dir, "a.txt", "caption")

        with self.assertRaises(ValueError) as ctx:
            ImageDirectoryDatasource(image_dir, caption_extension=".txt")

        self.assertIn("Duplicate", str(ctx.exception))
        self.assertIn("a.png", str(ctx.exception))
        self.assertIn("a.jpg", str(ctx.exception))

    def test_invalid_caption_directory(self):
        """Error when caption_directory doesn't exist."""
        image_dir = self.create_temp_dir()
        self.create_image(image_dir, "a.png")

        with self.assertRaises(ValueError) as ctx:
            ImageDirectoryDatasource(image_dir, caption_extension=".txt", caption_directory="/nonexistent/path")

        self.assertIn("does not exist", str(ctx.exception))

    def test_caption_directory_is_file(self):
        """Error when caption_directory points to a file."""
        image_dir = self.create_temp_dir()
        self.create_image(image_dir, "a.png")
        file_path = self.create_caption(image_dir, "not_a_dir.txt", "content")

        with self.assertRaises(ValueError) as ctx:
            ImageDirectoryDatasource(image_dir, caption_extension=".txt", caption_directory=file_path)

        self.assertIn("not a directory", str(ctx.exception))

    def test_empty_caption_directory_raises(self):
        """Error when caption_directory is empty string."""
        image_dir = self.create_temp_dir()
        self.create_image(image_dir, "a.png")

        with self.assertRaises(ValueError) as ctx:
            ImageDirectoryDatasource(image_dir, caption_extension=".txt", caption_directory="")

        self.assertIn("caption_directory cannot be empty", str(ctx.exception))

    def test_whitespace_only_caption_directory_raises(self):
        """Error when caption_directory is only whitespace."""
        image_dir = self.create_temp_dir()
        self.create_image(image_dir, "a.png")

        with self.assertRaises(ValueError) as ctx:
            ImageDirectoryDatasource(image_dir, caption_extension=".txt", caption_directory="   ")

        self.assertIn("caption_directory cannot be empty", str(ctx.exception))

    def test_whitespace_caption_directory_stripped(self):
        """Whitespace in caption_directory is stripped with warning."""
        image_dir = self.create_temp_dir()
        caption_dir = self.create_temp_dir()
        self.create_image(image_dir, "a.png")
        self.create_caption(caption_dir, "a.txt", "caption")

        # Add trailing space to caption_directory
        ds = ImageDirectoryDatasource(image_dir, caption_extension=".txt", caption_directory=caption_dir + " ")

        self.assertEqual(len(ds.image_paths), 1)
        # New format: "caption_directory contains leading/trailing whitespace: '/path ' -> '/path'"
        self.assertWarningContains("caption_directory", "whitespace", "->")

    def test_empty_caption_extension(self):
        """Error when caption_extension is empty."""
        image_dir = self.create_temp_dir()

        with self.assertRaises(ValueError) as ctx:
            ImageDirectoryDatasource(image_dir, caption_extension="")

        self.assertIn("empty", str(ctx.exception))

    def test_whitespace_caption_extension_stripped(self):
        """Whitespace in caption_extension is stripped with warning."""
        image_dir = self.create_temp_dir()
        self.create_image(image_dir, "a.png")
        self.create_caption(image_dir, "a.txt", "caption")

        ds = ImageDirectoryDatasource(image_dir, caption_extension=" .txt ")

        self.assertEqual(len(ds.image_paths), 1)
        self.assertWarningContains("whitespace", "stripped")

    def test_caption_extension_no_dot_warning(self):
        """Warning when caption_extension doesn't start with '.'."""
        image_dir = self.create_temp_dir()
        self.create_image(image_dir, "a.png")
        # Create caption without dot: "atxt" instead of "a.txt"
        self.create_caption(image_dir, "atxt", "caption")

        ds = ImageDirectoryDatasource(image_dir, caption_extension="txt")

        # Should warn about missing dot
        self.assertWarningContains("does not start with '.'")

    def test_caption_extension_none_raises(self):
        """Error when caption_extension is None for directory dataset."""
        image_dir = self.create_temp_dir()

        with self.assertRaises(ValueError) as ctx:
            ImageDirectoryDatasource(image_dir, caption_extension=None)

        self.assertIn("caption_extension is required", str(ctx.exception))

    def test_nonexistent_image_directory(self):
        """Error when image_directory doesn't exist."""
        with self.assertRaises(ValueError) as ctx:
            ImageDirectoryDatasource("/nonexistent/image/path", caption_extension=".txt")

        self.assertIn("image_directory does not exist", str(ctx.exception))

    def test_image_directory_is_file(self):
        """Error when image_directory points to a file."""
        image_dir = self.create_temp_dir()
        file_path = self.create_caption(image_dir, "not_a_dir.txt", "content")

        with self.assertRaises(ValueError) as ctx:
            ImageDirectoryDatasource(file_path, caption_extension=".txt")

        self.assertIn("image_directory is not a directory", str(ctx.exception))


class TestVideoDirectoryDatasource(unittest.TestCase):
    """Tests for VideoDirectoryDatasource with caption_directory."""

    def setUp(self):
        """Set up log capture and temp directories."""
        self.log_capture = LogCapture()
        self.log_capture.setLevel(logging.DEBUG)
        logger.logger.addHandler(self.log_capture)
        self.temp_dirs = []

    def tearDown(self):
        """Clean up log capture and temp directories."""
        logger.logger.removeHandler(self.log_capture)
        import shutil

        for d in self.temp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

    def create_temp_dir(self):
        """Create a temp directory and track it for cleanup."""
        d = tempfile.mkdtemp()
        self.temp_dirs.append(d)
        return d

    def create_video(self, directory: str, name: str):
        """Create a minimal MP4 file (just a marker file for testing)."""
        path = os.path.join(directory, name)
        # Create minimal valid MP4 header (ftyp box)
        with open(path, "wb") as f:
            # Minimal ftyp box: size(4) + 'ftyp'(4) + brand(4) + version(4) = 16 bytes
            f.write(b"\x00\x00\x00\x14ftypisom\x00\x00\x00\x00")
        return path

    def create_caption(self, directory: str, name: str, content: str = "test caption"):
        """Create a caption file."""
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_video_caption_filtering(self):
        """Videos without captions are filtered out with a warning."""
        video_dir = self.create_temp_dir()
        caption_dir = self.create_temp_dir()

        self.create_video(video_dir, "v1.mp4")
        self.create_video(video_dir, "v2.mp4")
        self.create_caption(caption_dir, "v1.txt", "caption for v1")
        # No caption for v2

        ds = VideoDirectoryDatasource(video_dir, caption_extension=".txt", caption_directory=caption_dir)

        self.assertEqual(len(ds.video_paths), 1)
        self.assertIn("v1.mp4", ds.video_paths[0])

        # Check warning
        warnings = self.log_capture.get_messages(logging.WARNING)
        warning_text = " ".join(warnings)
        self.assertIn("v2", warning_text)

    def test_video_separate_caption_directory(self):
        """Video captions can come from a separate directory."""
        video_dir = self.create_temp_dir()
        caption_dir = self.create_temp_dir()

        self.create_video(video_dir, "video.mp4")
        self.create_caption(caption_dir, "video.txt", "video caption")

        ds = VideoDirectoryDatasource(video_dir, caption_extension=".txt", caption_directory=caption_dir)

        self.assertEqual(len(ds.video_paths), 1)
        path, caption = ds.get_caption(0)
        self.assertEqual(caption, "video caption")

    def test_nonexistent_video_directory(self):
        """Error when video_directory doesn't exist."""
        with self.assertRaises(ValueError) as ctx:
            VideoDirectoryDatasource("/nonexistent/video/path", caption_extension=".txt")

        self.assertIn("video_directory does not exist", str(ctx.exception))

    def test_video_directory_is_file(self):
        """Error when video_directory points to a file."""
        video_dir = self.create_temp_dir()
        file_path = self.create_caption(video_dir, "not_a_dir.txt", "content")

        with self.assertRaises(ValueError) as ctx:
            VideoDirectoryDatasource(file_path, caption_extension=".txt")

        self.assertIn("video_directory is not a directory", str(ctx.exception))


class TestJsonlDuplicateBasenames(unittest.TestCase):
    """Tests for JSONL datasource duplicate basename detection."""

    def setUp(self):
        """Set up temp directories."""
        self.temp_dirs = []

    def tearDown(self):
        """Clean up temp directories."""
        import shutil

        for d in self.temp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)

    def create_temp_dir(self):
        """Create a temp directory and track it for cleanup."""
        d = tempfile.mkdtemp()
        self.temp_dirs.append(d)
        return d

    def create_image(self, directory: str, name: str):
        """Create a minimal PNG file."""
        from PIL import Image

        img = Image.new("RGB", (1, 1), color="red")
        path = os.path.join(directory, name)
        img.save(path)
        return path

    def create_video(self, directory: str, name: str):
        """Create a minimal MP4 file."""
        path = os.path.join(directory, name)
        with open(path, "wb") as f:
            f.write(b"\x00\x00\x00\x14ftypisom\x00\x00\x00\x00")
        return path

    def test_image_jsonl_duplicate_basenames(self):
        """JSONL with duplicate image basenames raises error."""
        tmp_dir = self.create_temp_dir()

        # Create actual image files with duplicate basenames
        sub1 = os.path.join(tmp_dir, "sub1")
        sub2 = os.path.join(tmp_dir, "sub2")
        os.makedirs(sub1)
        os.makedirs(sub2)
        self.create_image(sub1, "photo.png")
        self.create_image(sub2, "photo.jpg")

        # Create JSONL with paths that have duplicate basenames
        jsonl_path = os.path.join(tmp_dir, "data.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(f'{{"image_path": "{sub1}/photo.png", "caption": "first"}}\n')
            f.write(f'{{"image_path": "{sub2}/photo.jpg", "caption": "second"}}\n')

        with self.assertRaises(ValueError) as ctx:
            ImageJsonlDatasource(jsonl_path)

        self.assertIn("Duplicate", str(ctx.exception))
        self.assertIn("photo", str(ctx.exception))

    def test_video_jsonl_duplicate_basenames(self):
        """JSONL with duplicate video basenames raises error."""
        tmp_dir = self.create_temp_dir()

        # Create actual video files with duplicate basenames
        sub1 = os.path.join(tmp_dir, "sub1")
        sub2 = os.path.join(tmp_dir, "sub2")
        os.makedirs(sub1)
        os.makedirs(sub2)
        self.create_video(sub1, "clip.mp4")
        self.create_video(sub2, "clip.webm")

        # Create JSONL with paths that have duplicate basenames
        jsonl_path = os.path.join(tmp_dir, "data.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(f'{{"video_path": "{sub1}/clip.mp4", "caption": "first"}}\n')
            f.write(f'{{"video_path": "{sub2}/clip.webm", "caption": "second"}}\n')

        with self.assertRaises(ValueError) as ctx:
            VideoJsonlDatasource(jsonl_path)

        self.assertIn("Duplicate", str(ctx.exception))
        self.assertIn("clip", str(ctx.exception))

    def test_image_jsonl_unique_basenames_ok(self):
        """JSONL with unique basenames works fine."""
        tmp_dir = self.create_temp_dir()
        self.create_image(tmp_dir, "a.png")
        self.create_image(tmp_dir, "b.png")

        jsonl_path = os.path.join(tmp_dir, "data.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(f'{{"image_path": "{tmp_dir}/a.png", "caption": "first"}}\n')
            f.write(f'{{"image_path": "{tmp_dir}/b.png", "caption": "second"}}\n')

        ds = ImageJsonlDatasource(jsonl_path)
        self.assertEqual(len(ds.data), 2)

    def test_video_jsonl_unique_basenames_ok(self):
        """JSONL with unique video basenames works fine."""
        tmp_dir = self.create_temp_dir()
        self.create_video(tmp_dir, "a.mp4")
        self.create_video(tmp_dir, "b.mp4")

        jsonl_path = os.path.join(tmp_dir, "data.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(f'{{"video_path": "{tmp_dir}/a.mp4", "caption": "first"}}\n')
            f.write(f'{{"video_path": "{tmp_dir}/b.mp4", "caption": "second"}}\n')

        ds = VideoJsonlDatasource(jsonl_path)
        self.assertEqual(len(ds.data), 2)

    def test_image_jsonl_missing_image_path(self):
        """JSONL with missing image_path raises error with line number."""
        tmp_dir = self.create_temp_dir()
        self.create_image(tmp_dir, "a.png")

        jsonl_path = os.path.join(tmp_dir, "data.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(f'{{"image_path": "{tmp_dir}/a.png", "caption": "first"}}\n')
            f.write('{"caption": "missing image_path"}\n')  # Line 2 is malformed

        with self.assertRaises(ValueError) as ctx:
            ImageJsonlDatasource(jsonl_path)

        self.assertIn("Missing 'image_path'", str(ctx.exception))
        self.assertIn("line 2", str(ctx.exception))

    def test_video_jsonl_missing_video_path(self):
        """JSONL with missing video_path raises error with line number."""
        tmp_dir = self.create_temp_dir()
        self.create_video(tmp_dir, "a.mp4")

        jsonl_path = os.path.join(tmp_dir, "data.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(f'{{"video_path": "{tmp_dir}/a.mp4", "caption": "first"}}\n')
            f.write('{"caption": "missing video_path"}\n')  # Line 2 is malformed

        with self.assertRaises(ValueError) as ctx:
            VideoJsonlDatasource(jsonl_path)

        self.assertIn("Missing 'video_path'", str(ctx.exception))
        self.assertIn("line 2", str(ctx.exception))

    def test_image_jsonl_empty_image_path(self):
        """JSONL with empty image_path raises error with line number."""
        tmp_dir = self.create_temp_dir()

        jsonl_path = os.path.join(tmp_dir, "data.jsonl")
        with open(jsonl_path, "w") as f:
            f.write('{"image_path": "", "caption": "empty path"}\n')

        with self.assertRaises(ValueError) as ctx:
            ImageJsonlDatasource(jsonl_path)

        self.assertIn("Missing 'image_path'", str(ctx.exception))
        self.assertIn("line 1", str(ctx.exception))


class TestConfigSchemaAcceptsCaptionDirectory(unittest.TestCase):
    """Tests for config schema validation."""

    def test_config_schema_accepts_caption_directory(self):
        """Schema validation accepts caption_directory field."""
        import tempfile

        from musubi_tuner.dataset.config_utils import ConfigSanitizer, load_user_config

        toml_content = """
[[datasets]]
image_directory = "/tmp/images"
caption_directory = "/tmp/captions"
cache_directory = "/tmp/cache"
caption_extension = ".txt"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_user_config(config_path)
            sanitized = ConfigSanitizer().sanitize_user_config(config)

            self.assertEqual(sanitized["datasets"][0]["caption_directory"], "/tmp/captions")
        finally:
            os.unlink(config_path)

    def test_video_config_schema_accepts_caption_directory(self):
        """Schema validation accepts caption_directory for video datasets."""
        import tempfile

        from musubi_tuner.dataset.config_utils import ConfigSanitizer, load_user_config

        toml_content = """
[[datasets]]
video_directory = "/tmp/videos"
caption_directory = "/tmp/video_captions"
cache_directory = "/tmp/cache"
caption_extension = ".txt"
target_frames = [81]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_user_config(config_path)
            sanitized = ConfigSanitizer().sanitize_user_config(config)

            self.assertEqual(sanitized["datasets"][0]["caption_directory"], "/tmp/video_captions")
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()
