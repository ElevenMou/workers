"""Constants shared across API routers."""

# Use string paths so the API process does not eagerly import heavy worker deps.
ANALYZE_TASK_PATH = "tasks.analyze_video.analyze_video_task"
GENERATE_TASK_PATH = "tasks.generate_clip.generate_clip_task"
CUSTOM_CLIP_TASK_PATH = "tasks.custom_clip.custom_clip_task"
PUBLISH_TASK_PATH = "tasks.publishing.publish.publish_clip_task"

CAPTION_FONT_CASES = ["as_typed", "uppercase", "lowercase"]
CAPTION_POSITIONS = ["top", "middle", "bottom", "custom"]
CAPTION_LINES_PER_PAGE_OPTIONS = [1, 2, 3, 4]
CAPTION_STYLES = ["grouped", "karaoke", "highlight", "highlight_box"]
CAPTION_ANIMATIONS = ["none", "fade", "pop", "slide_up", "bounce", "glow", "karaoke", "typewriter"]

CANVAS_ASPECT_RATIO_OPTIONS = ["9:16", "1:1", "4:5", "16:9"]
VIDEO_SCALE_MODE_OPTIONS = ["fit", "fill"]
RECOMMENDED_CANVAS_ASPECT_RATIO = "9:16"
