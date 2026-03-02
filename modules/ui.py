import os
import queue
import threading
import time
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from modules.gpu_processing import gpu_cvt_color, gpu_resize, gpu_flip
from PIL import Image, ImageOps
import json
import modules.globals
import modules.metadata
from modules.face_analyser import get_one_face
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)
from modules.gettext import LanguageManager
from modules.ui_tooltip import ToolTip
from modules.mapping_list import MAPPING_LIST
from modules.ui_mapping_list import MappingListWidget
import platform
from modules.camera import get_available_cameras

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

# Monkey-patch CustomTkinter DropdownMenu for Tk 9.0 compatibility.
# Tk 9.0 returns "" from Menu.index("end") on an empty menu, causing TclError
# in DropdownMenu._add_menu_commands when it calls self.delete(0, "end").
import tkinter as _tk

if _tk.TkVersion >= 9.0:
    from customtkinter.windows.widgets.core_widget_classes.dropdown_menu import (
        DropdownMenu as _DropdownMenu,
    )

    _orig_add_menu_commands = _DropdownMenu._add_menu_commands

    def _patched_add_menu_commands(self):
        try:
            _orig_add_menu_commands(self)
        except _tk.TclError:
            # Empty menu — just add commands without deleting first
            import sys

            if sys.platform.startswith("linux"):
                for value in self._values:
                    self.add_command(
                        label="  " + value.ljust(self._min_character_width) + "  ",
                        command=lambda v=value: self._button_callback(v),
                        compound="left",
                    )
            else:
                for value in self._values:
                    self.add_command(
                        label=value.ljust(self._min_character_width),
                        command=lambda v=value: self._button_callback(v),
                        compound="left",
                    )

    _DropdownMenu._add_menu_commands = _patched_add_menu_commands

# Re-export moved functions for backward compatibility
from modules.ui_analysis import analyze_target, check_and_ignore_nsfw  # noqa: F401
from modules.ui_webcam import (  # noqa: F401
    webcam_preview,
    create_webcam_preview,
    _capture_thread_func,
    _processing_thread_func,
    DETECT_EVERY_N,
)
from modules.ui_mapper import (  # noqa: F401
    create_source_target_popup,
    create_source_target_popup_for_webcam,
    update_webcam_source,
    update_webcam_target,
    update_popup_source,
    clear_source_target_images,
    refresh_data,
    close_mapper_window,
    update_pop_status,
    update_pop_live_status,
    POPUP,
    POPUP_LIVE,
    source_label_dict,
    source_label_dict_live,
    target_label_dict_live,
    popup_status_label,
    popup_status_label_live,
    POPUP_WIDTH,
    POPUP_HEIGHT,
    POPUP_SCROLL_WIDTH,
    POPUP_SCROLL_HEIGHT,
    POPUP_LIVE_WIDTH,
    POPUP_LIVE_HEIGHT,
    POPUP_LIVE_SCROLL_WIDTH,
    POPUP_LIVE_SCROLL_HEIGHT,
    MAPPER_PREVIEW_MAX_HEIGHT,
    MAPPER_PREVIEW_MAX_WIDTH,
    DEFAULT_BUTTON_WIDTH,
    DEFAULT_BUTTON_HEIGHT,
)


ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 1000

SIDEBAR_THUMB_SIZE = (120, 120)

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 960
PREVIEW_DEFAULT_HEIGHT = 540

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

_ = lambda x: x  # replaced by LanguageManager in init()
preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
_download_progress_bar = None
_last_progress_update = 0.0

img_ft, vid_ft = modules.globals.file_types

# Debounce timer for responsive image scaling
_resize_timer_id = None

# Selected camera index, updated by camera detection and dropdown selection
_selected_camera_index = 0

# Embedded preview state
_embedded_label: "ctk.CTkLabel | None" = None          # CTkLabel inside embedded frame
_popout_label: "ctk.CTkLabel | None" = None            # CTkLabel inside PREVIEW CTkToplevel
_preview_embedded = True                               # True = embedded, False = pop-out window
_embedded_preview_frame: "ctk.CTkFrame | None" = None  # the collapsible frame widget
_mapping_widget: "MappingListWidget | None" = None     # MappingListWidget instance

# Mode toggle state
_current_mode: str = "Live Cam"
_swap_button_widget: "ctk.CTkButton | None" = None
_target_frame_widget: "ctk.CTkFrame | None" = None
_start_button_widget: "ctk.CTkButton | None" = None
_live_button_widget: "ctk.CTkButton | None" = None
_settings_tabview_ref: "ctk.CTkTabView | None" = None


def init(start: Callable[[], None], destroy: Callable[[], None], lang: str) -> ctk.CTk:
    global ROOT, PREVIEW, _

    lang_manager = LanguageManager(lang)
    _ = lang_manager._
    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def _state_file_path() -> str:
    import platform as _platform
    if _platform.system() == "Windows":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.path.join(os.path.expanduser("~"), ".config")
    config_dir = os.path.join(base, "deep-live-cam")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "switch_states.json")


def save_switch_states():
    switch_states = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "use_png_frames": modules.globals.use_png_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "virtual_cam": modules.globals.virtual_cam,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "rife_enabled": modules.globals.rife_enabled,
        "rife_model": modules.globals.rife_model,
        "rife_multiplier": modules.globals.rife_multiplier,
        "half_rate_processing": modules.globals.half_rate_processing,
        "keyframe_interval": modules.globals.keyframe_interval,
        "live_max_fps": modules.globals.live_max_fps,
        "source_path": modules.globals.source_path,
        "target_path": modules.globals.target_path,
        "mappings": MAPPING_LIST.to_dict(),
        "ui_mode": _current_mode,
    }
    with open(_state_file_path(), "w") as f:
        json.dump(switch_states, f)


def load_switch_states():
    try:
        with open(_state_file_path(), "r") as f:
            switch_states = json.load(f)
        modules.globals.keep_fps = switch_states.get("keep_fps", True)
        modules.globals.keep_audio = switch_states.get("keep_audio", True)
        modules.globals.keep_frames = switch_states.get("keep_frames", False)
        modules.globals.use_png_frames = switch_states.get("use_png_frames", False)
        modules.globals.many_faces = switch_states.get("many_faces", False)
        modules.globals.map_faces = switch_states.get("map_faces", False)
        modules.globals.poisson_blend = switch_states.get("poisson_blend", False)
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.virtual_cam = switch_states.get("virtual_cam", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get(
            "show_mouth_mask_box", False
        )
        modules.globals.rife_enabled = switch_states.get("rife_enabled", False)
        modules.globals.rife_model = switch_states.get("rife_model", "rife-v4.25-lite")
        modules.globals.rife_multiplier = switch_states.get("rife_multiplier", 2)
        modules.globals.half_rate_processing = switch_states.get("half_rate_processing", False)
        modules.globals.keyframe_interval = switch_states.get("keyframe_interval", 2)
        modules.globals.live_max_fps = switch_states.get("live_max_fps", 30)
        # Restore last-used paths; validate existence before accepting.
        saved_target = switch_states.get("target_path")
        if saved_target and os.path.isfile(saved_target):
            modules.globals.target_path = saved_target
        # Restore mapping list from saved state, or migrate from legacy source_path
        saved_mappings = switch_states.get("mappings")
        if saved_mappings is not None:
            MAPPING_LIST.restore_from_dict(saved_mappings)
            modules.globals.source_path = MAPPING_LIST.effective_source_path()
            modules.globals.map_faces = MAPPING_LIST.effective_map_faces()
        else:
            # Migration: old state file without "mappings" key
            saved_source = switch_states.get("source_path")
            if saved_source and os.path.isfile(saved_source):
                MAPPING_LIST.restore_from_source_path(saved_source)
                modules.globals.source_path = saved_source
            modules.globals.map_faces = switch_states.get("map_faces", False)
        # Rebuild frame_processors from restored fp_ui so toggled enhancers
        # are included even if they weren't in the CLI --frame-processor list.
        _sync_enhancer_frame_processors()
        global _current_mode
        _current_mode = switch_states.get("ui_mode", "Live Cam")
    except FileNotFoundError:
        pass


def _restore_recent_paths() -> None:
    """Populate image labels from saved paths after labels have been created.

    Source face thumbnails are now handled by MappingListWidget (it reads
    source_cv2 from MappingEntry).  For restored paths without cv2 data,
    we reload the images and detect faces so thumbnails render correctly.
    """
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET

    # Reload source face data for restored mappings (paths only, no cv2/face)
    for entry in MAPPING_LIST.get_entries():
        if entry.source_path and is_image(entry.source_path) and entry.source_face is None:
            img = cv2.imread(entry.source_path)
            if img is not None:
                face = get_one_face(img)
                if face is not None:
                    x_min, y_min, x_max, y_max = face["bbox"]
                    cropped = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                    MAPPING_LIST.set_source(entry.id, entry.source_path, cropped, face)
            RECENT_DIRECTORY_SOURCE = os.path.dirname(entry.source_path)
        if entry.pin_path and is_image(entry.pin_path) and entry.pin_face is None:
            img = cv2.imread(entry.pin_path)
            if img is not None:
                face = get_one_face(img)
                if face is not None:
                    x_min, y_min, x_max, y_max = face["bbox"]
                    cropped = img[int(y_min):int(y_max), int(x_min):int(x_max)]
                    MAPPING_LIST.set_pin(entry.id, entry.pin_path, cropped, face)

    if modules.globals.target_path:
        if is_image(modules.globals.target_path):
            RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
            image = render_image_preview(modules.globals.target_path, SIDEBAR_THUMB_SIZE)
            target_label.configure(image=image)
        elif is_video(modules.globals.target_path):
            RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
            frame = render_video_preview(modules.globals.target_path, SIDEBAR_THUMB_SIZE)
            target_label.configure(image=frame)


def _setup_window(destroy: Callable) -> ctk.CTk:
    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(900, 600)
    root.title(
        f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
    )
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    # Two-column layout: fixed sidebar on left (col 0), preview on right (col 1)
    root.columnconfigure(0, weight=0)  # sidebar fixed to content width
    root.columnconfigure(1, weight=1)  # preview gets all extra space
    root.rowconfigure(0, weight=0)  # mode toggle
    root.rowconfigure(1, weight=0)  # top_frame (images)
    root.rowconfigure(2, weight=1)  # settings_tabview (grows)
    root.rowconfigure(3, weight=0)  # action_frame
    root.rowconfigure(4, weight=0)  # status_frame

    return root


def _on_mapping_change() -> None:
    """Observer callback: sync MappingList state to globals and FaceMapStore."""
    from modules.face_map_store import STORE as _MAP_STORE
    MAPPING_LIST.sync_to_store(_MAP_STORE)
    modules.globals.source_path = MAPPING_LIST.effective_source_path()
    modules.globals.map_faces = MAPPING_LIST.effective_map_faces()
    save_switch_states()


def _add_top_frame(root: ctk.CTk) -> None:
    global target_label, _mapping_widget, _swap_button_widget, _target_frame_widget

    top_frame = ctk.CTkFrame(root)
    top_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(10, 5))
    top_frame.columnconfigure(0, weight=1)
    top_frame.columnconfigure(1, weight=0)
    top_frame.columnconfigure(2, weight=1)

    # Source column — inline mapping list replaces the old single source selector
    source_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
    source_frame.grid(row=0, column=0, sticky="nsew", padx=3, pady=5)
    source_frame.columnconfigure(0, weight=1)

    _mapping_widget = MappingListWidget(
        source_frame, MAPPING_LIST, on_change_callback=_on_mapping_change,
    )
    MAPPING_LIST.on_change(_on_mapping_change)

    # Swap button
    swap_faces_button = ctk.CTkButton(
        top_frame, text="\u2194", cursor="hand2", width=30,
        command=lambda: swap_faces_paths(),
    )
    swap_faces_button.grid(row=0, column=1, padx=5)
    ToolTip(swap_faces_button, _("Swap source and target images"))
    _swap_button_widget = swap_faces_button

    # Target column
    target_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
    target_frame.grid(row=0, column=2, sticky="nsew", padx=3, pady=5)
    target_frame.columnconfigure(0, weight=1)
    _target_frame_widget = target_frame

    target_label = ctk.CTkLabel(target_frame, text="", width=120, height=120)
    target_label.grid(row=0, column=0, pady=(5, 5))

    select_target_button = ctk.CTkButton(
        target_frame, text=_("Select a target"), cursor="hand2",
        command=lambda: select_target_path(),
    )
    select_target_button.grid(row=1, column=0, pady=(0, 2), sticky="ew", padx=5)
    ToolTip(select_target_button, _("Choose the target image or video to apply face swap to"))

    capture_target_button = ctk.CTkButton(
        target_frame, text=_("Capture from camera"), cursor="hand2",
        command=lambda: capture_target_from_camera(),
    )
    capture_target_button.grid(row=2, column=0, pady=(0, 5), sticky="ew", padx=5)
    ToolTip(capture_target_button, _("Take a photo from your webcam to use as target"))


def _add_embedded_preview(root: ctk.CTk) -> None:
    """Create the embedded preview panel on the right side (column 1, all rows)."""
    global _embedded_preview_frame, _embedded_label, preview_label

    _embedded_preview_frame = ctk.CTkFrame(root)
    _embedded_preview_frame.grid(row=0, column=1, rowspan=5, sticky="nsew", padx=(5, 10), pady=10)

    _embedded_preview_frame.columnconfigure(0, weight=1)
    _embedded_preview_frame.rowconfigure(0, weight=0)
    _embedded_preview_frame.rowconfigure(1, weight=1)

    header = ctk.CTkFrame(_embedded_preview_frame, fg_color="transparent")
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)
    ctk.CTkLabel(header, text="Preview", anchor="w").grid(
        row=0, column=0, sticky="w", padx=10
    )
    ctk.CTkButton(
        header, text="⤢", width=30, cursor="hand2",
        command=pop_out_preview,
    ).grid(row=0, column=1, padx=(0, 5), pady=5)

    _embedded_label = ctk.CTkLabel(_embedded_preview_frame, text="")
    _embedded_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
    preview_label = _embedded_label


# --- Data-driven switch definitions ---

def _get_switch_defs():
    """Return switch definitions grouped by tab.

    Each definition: (label, global_attr_or_tumbler, default_value, tooltip)
    - If global_attr_or_tumbler starts with "fp_ui:", it's a tumbler key
    - Otherwise it's a modules.globals attribute name
    """
    return {
        "Processing": [
            (_("Mouth Mask"), "mouth_mask", False,
             _("Preserve original mouth movement in the swapped face")),
            (_("Show Mask Outline"), "show_mouth_mask_box", False,
             _("Display the mouth mask boundary for debugging")),
            (_("Swap All Faces"), "many_faces", False,
             _("Swap every detected face, not just the primary one")),
            (_("Seamless Blend"), "poisson_blend", False,
             _("Blend face edges smoothly into the target using Poisson blending")),
        ],
        "Enhancement": [
            (_("GFPGAN Enhancer"), "fp_ui:face_enhancer", False,
             _("Improve face quality using the GFPGAN restoration model")),
            (_("GPEN-256"), "fp_ui:face_enhancer_gpen256", False,
             _("Use GPEN face enhancement model at 256px resolution (faster)")),
            (_("GPEN-512"), "fp_ui:face_enhancer_gpen512", False,
             _("Use GPEN face enhancement model at 512px resolution (higher quality)")),
            (_("CodeFormer"), "fp_ui:face_enhancer_codeformer", False,
             _("Transformer-based face restoration with adjustable fidelity (best quality)")),
            (_("RIFE Interpolation"), "rife_enabled", False,
             _("Generate intermediate frames for smoother motion (2x or 4x frame rate)")),
        ],
        "Export": [
            (_("Preserve Frame Rate"), "keep_fps", True,
             _("Output video keeps the original frame rate")),
            (_("Preserve Audio"), "keep_audio", True,
             _("Copy audio track from the source video to output")),
            (_("Save Temp Frames"), "keep_frames", False,
             _("Keep extracted frames on disk after processing (uses disk space)")),
            (_("PNG Frames"), "use_png_frames", False,
             _("Use lossless PNG for intermediate frames — no compression artifacts, but ~10x more disk I/O")),
        ],
        "Live Mode": [
            (_("Color Correction"), "color_correction", False,
             _("Fix blue/green color cast from some webcams")),
            (_("Show FPS"), "show_fps", False,
             _("Display frames-per-second counter on the live preview")),
            (_("Virtual Camera"), "virtual_cam", False,
             _("Output to a virtual camera device for use in Zoom, Meet, etc.")),
            (_("Skip Frames"), "half_rate_processing", False,
             _("Process every other frame for better performance at the cost of smoothness")),
        ],
    }


def _get_switch_value(attr: str) -> bool:
    if attr.startswith("fp_ui:"):
        key = attr[len("fp_ui:"):]
        return modules.globals.fp_ui.get(key, False)
    return getattr(modules.globals, attr, False)


def _create_switch(
    parent: ctk.CTkFrame, label: str, attr: str, tooltip: str = "",
) -> ctk.CTkSwitch:
    value_var = ctk.BooleanVar(value=_get_switch_value(attr))

    if attr.startswith("fp_ui:"):
        key = attr[len("fp_ui:"):]
        command = lambda: (
            update_tumbler(key, value_var.get()),
            save_switch_states(),
        )
    else:
        command = lambda: (
            setattr(modules.globals, attr, value_var.get()),
            save_switch_states(),
        )

    switch = ctk.CTkSwitch(
        parent, text=label, variable=value_var, cursor="hand2", command=command,
    )
    if tooltip:
        ToolTip(switch, tooltip)
    return switch


def _add_settings_tabview(root: ctk.CTk, live_button: ctk.CTkButton) -> None:
    global _settings_tabview_ref
    tabview = ctk.CTkTabview(root)
    tabview.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
    _settings_tabview_ref = tabview

    switch_defs = _get_switch_defs()

    for tab_name, switches in switch_defs.items():
        tab = tabview.add(tab_name)
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)

        for i, (label, attr, _default, tooltip) in enumerate(switches):
            sw = _create_switch(tab, label, attr, tooltip)
            sw.grid(row=i, column=0, columnspan=2, sticky="w", padx=8, pady=3)

    # Enhancement tab: add sliders then RIFE model/multiplier dropdowns
    enhancement_tab = tabview.tab("Enhancement")
    _add_sliders_to_tab(enhancement_tab, len(switch_defs["Enhancement"]))
    # Sliders occupy +1 and +2; RIFE controls start at +3
    _add_rife_controls_to_tab(enhancement_tab, len(switch_defs["Enhancement"]), row_offset=3)

    # Live Mode tab: add camera dropdown and keyframe interval control
    live_tab = tabview.tab("Live Mode")
    _add_camera_to_tab(live_tab, root, len(switch_defs["Live Mode"]), live_button)
    _add_half_rate_controls_to_tab(live_tab, len(switch_defs["Live Mode"]))


def _add_sliders_to_tab(tab: ctk.CTkFrame, num_switches: int) -> None:
    start_row = num_switches + 1

    transparency_var = ctk.DoubleVar(value=1.0)

    def on_transparency_change(value: float):
        val = float(value)
        modules.globals.opacity = val
        percentage = int(val * 100)

        if percentage == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif percentage == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {percentage}%")

    transparency_label = ctk.CTkLabel(tab, text="Transparency:")
    transparency_label.grid(row=start_row, column=0, sticky="w", padx=8, pady=(15, 2))

    transparency_slider = ctk.CTkSlider(
        tab, from_=0, to=1, variable=transparency_var,
        command=on_transparency_change,
        fg_color="#E0E0E0", progress_color="#007BFF",
        button_color="#FFFFFF", button_hover_color="#CCCCCC",
        height=5, border_width=1, corner_radius=3,
    )
    transparency_slider.grid(
        row=start_row, column=1, sticky="ew", padx=(0, 8), pady=(15, 2),
    )
    ToolTip(transparency_slider, _("Blend between original and swapped face (0% = original, 100% = fully swapped)"))

    sharpness_var = ctk.DoubleVar(value=0.0)

    def on_sharpness_change(value: float):
        modules.globals.sharpness = float(value)
        update_status(f"Sharpness set to {value:.1f}")

    sharpness_label = ctk.CTkLabel(tab, text="Sharpness:")
    sharpness_label.grid(row=start_row + 1, column=0, sticky="w", padx=8, pady=2)

    sharpness_slider = ctk.CTkSlider(
        tab, from_=0, to=5, variable=sharpness_var,
        command=on_sharpness_change,
        fg_color="#E0E0E0", progress_color="#007BFF",
        button_color="#FFFFFF", button_hover_color="#CCCCCC",
        height=5, border_width=1, corner_radius=3,
    )
    sharpness_slider.grid(
        row=start_row + 1, column=1, sticky="ew", padx=(0, 8), pady=2,
    )
    ToolTip(sharpness_slider, _("Sharpen the enhanced face output"))


def _add_rife_controls_to_tab(tab: ctk.CTkFrame, num_switches: int, row_offset: int = 1) -> None:
    """Add RIFE model selector and multiplier dropdown below existing tab content."""
    start_row = num_switches + row_offset

    # Model selector
    model_label = ctk.CTkLabel(tab, text="RIFE Model:")
    model_label.grid(row=start_row, column=0, sticky="w", padx=8, pady=(15, 2))

    model_values = ["rife-v4.25-lite", "rife-v4.25"]
    current_model = getattr(modules.globals, "rife_model", "rife-v4.25-lite")
    model_variable = ctk.StringVar(value=current_model)

    def on_model_change(choice):
        modules.globals.rife_model = choice
        save_switch_states()
        update_status(f"RIFE model set to {choice}")

    model_optionmenu = ctk.CTkOptionMenu(
        tab, variable=model_variable, values=model_values,
        command=on_model_change,
    )
    model_optionmenu.grid(
        row=start_row, column=1, sticky="ew", padx=(0, 8), pady=(15, 2),
    )
    ToolTip(model_optionmenu, _("Choose RIFE interpolation model (lite = faster, full = better quality)"))

    # Multiplier selector
    multiplier_label = ctk.CTkLabel(tab, text="RIFE Multiplier:")
    multiplier_label.grid(row=start_row + 1, column=0, sticky="w", padx=8, pady=2)

    multiplier_values = ["2x", "4x"]
    current_mult = getattr(modules.globals, "rife_multiplier", 2)
    multiplier_variable = ctk.StringVar(value=f"{current_mult}x")

    def on_multiplier_change(choice):
        modules.globals.rife_multiplier = int(choice.replace("x", ""))
        save_switch_states()
        update_status(f"RIFE multiplier set to {choice}")

    multiplier_optionmenu = ctk.CTkOptionMenu(
        tab, variable=multiplier_variable, values=multiplier_values,
        command=on_multiplier_change,
    )
    multiplier_optionmenu.grid(
        row=start_row + 1, column=1, sticky="ew", padx=(0, 8), pady=2,
    )
    ToolTip(multiplier_optionmenu, _("Frame rate multiplication factor"))


def _add_half_rate_controls_to_tab(tab: ctk.CTkFrame, num_switches: int) -> None:
    """Add keyframe interval dropdown to the Live Mode tab (below the camera selector)."""
    # Camera selector occupies row (num_switches + 1) // 2 + 1; we start one below it
    start_row = num_switches + 2

    interval_label = ctk.CTkLabel(tab, text="Keyframe Interval:")
    interval_label.grid(row=start_row, column=0, sticky="w", padx=8, pady=(8, 2))

    interval_values = ["2", "3", "4", "5", "8", "10"]
    current_interval = str(getattr(modules.globals, "keyframe_interval", 2))
    if current_interval not in interval_values:
        current_interval = "2"
    interval_variable = ctk.StringVar(value=current_interval)

    def on_interval_change(choice):
        modules.globals.keyframe_interval = int(choice)
        save_switch_states()
        update_status(f"Keyframe interval set to every {choice} frames")

    interval_optionmenu = ctk.CTkOptionMenu(
        tab, variable=interval_variable, values=interval_values,
        command=on_interval_change,
    )
    interval_optionmenu.grid(
        row=start_row, column=1, sticky="ew", padx=(0, 8), pady=(8, 2),
    )
    ToolTip(interval_optionmenu, _("Process a full detection every N frames (higher = faster, lower = more accurate)"))

    # FPS cap
    fps_label = ctk.CTkLabel(tab, text=_("Max Preview FPS:"))
    fps_label.grid(row=start_row + 1, column=0, sticky="w", padx=8, pady=(8, 2))

    fps_values = ["15", "24", "30", "60"]
    current_fps = str(getattr(modules.globals, "live_max_fps", 30))
    if current_fps not in fps_values:
        current_fps = "30"
    fps_variable = ctk.StringVar(value=current_fps)

    def on_fps_change(choice):
        modules.globals.live_max_fps = int(choice)
        save_switch_states()
        update_status(f"Preview FPS cap set to {choice}")

    fps_optionmenu = ctk.CTkOptionMenu(
        tab, variable=fps_variable, values=fps_values,
        command=on_fps_change,
    )
    fps_optionmenu.grid(
        row=start_row + 1, column=1, sticky="ew", padx=(0, 8), pady=(8, 2),
    )
    ToolTip(fps_optionmenu, _("Cap the preview frame rate — lower values reduce CPU/GPU heat (30 FPS recommended)"))


def _add_camera_to_tab(
    tab: ctk.CTkFrame, root: ctk.CTk, num_switches: int, live_button: ctk.CTkButton,
) -> None:
    start_row = num_switches + 1

    camera_label = ctk.CTkLabel(tab, text=_("Select Camera:"))
    camera_label.grid(row=start_row, column=0, sticky="w", padx=8, pady=(15, 5))

    camera_variable = ctk.StringVar(value=_("Detecting cameras..."))
    camera_optionmenu = ctk.CTkOptionMenu(
        tab, variable=camera_variable,
        values=[_("Detecting cameras...")], state="disabled",
    )
    camera_optionmenu.grid(
        row=start_row, column=1, sticky="ew", padx=(0, 8), pady=(15, 5),
    )
    ToolTip(camera_optionmenu, _("Select which camera to use for live mode"))

    camera_indices: list = []
    camera_names: list = []

    def _on_camera_selected(choice):
        global _selected_camera_index
        if camera_names and choice in camera_names:
            _selected_camera_index = camera_indices[camera_names.index(choice)]

    # Wire up the live button command to use the camera selection from this tab.
    # Config snapshot is taken at click time so UI changes before clicking are captured.
    def _start_webcam():
        from modules.processing_config_factory import build_config_from_globals as _bcfg
        camera_index = (
            camera_indices[camera_names.index(camera_variable.get())]
            if camera_names and camera_names[0] != "No cameras found"
            else None
        )
        webcam_preview(root, camera_index, config=_bcfg())

    live_button.configure(command=_start_webcam)
    camera_optionmenu.configure(command=_on_camera_selected)

    def _finish_camera_probe(indices, names):
        global _selected_camera_index
        camera_indices.clear()
        camera_indices.extend(indices)
        camera_names.clear()
        camera_names.extend(names)
        if names and names[0] != "No cameras found":
            camera_variable.set(names[0])
            _selected_camera_index = indices[0]
            camera_optionmenu.configure(values=names, state="normal")
            live_button.configure(state="normal")
        else:
            camera_variable.set(_("No cameras found"))
            camera_optionmenu.configure(values=[_("No cameras found")], state="disabled")

    _camera_queue: queue.Queue = queue.Queue()

    def _poll_camera_queue():
        try:
            indices, names = _camera_queue.get_nowait()
            _finish_camera_probe(indices, names)
        except queue.Empty:
            root.after(100, _poll_camera_queue)

    def _enumerate_cameras():
        on_windows = platform.system() == "Windows"
        if on_windows:
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(0, 0)  # type: ignore[attr-defined]
        try:
            indices, names = get_available_cameras()
            _camera_queue.put((indices, names))
        finally:
            if on_windows:
                import ctypes
                ctypes.windll.ole32.CoUninitialize()  # type: ignore[attr-defined]

    threading.Thread(target=_enumerate_cameras, daemon=True).start()
    root.after(100, _poll_camera_queue)


def _add_action_buttons(root: ctk.CTk, start: Callable, destroy: Callable) -> ctk.CTkButton:
    """Create action bar with Start/Live, Destroy, Preview buttons. Returns the Live button."""
    global _start_button_widget, _live_button_widget

    action_frame = ctk.CTkFrame(root, fg_color="transparent")
    action_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
    action_frame.columnconfigure(0, weight=1)
    action_frame.columnconfigure(1, weight=1)

    # Start and Live share the same grid cell (row=0, col=0); mode toggle shows one at a time
    start_button = ctk.CTkButton(
        action_frame, text=_("Start"), cursor="hand2",
        command=lambda: analyze_target(start, root),
    )
    start_button.grid(row=0, column=0, sticky="ew", padx=3)
    ToolTip(start_button, _("Begin processing the target image/video with selected face"))
    start_button.grid_remove()  # hidden by default; shown in File Processing mode
    _start_button_widget = start_button

    # Live button — command and state wired up by _add_camera_to_tab after detection
    live_button = ctk.CTkButton(
        action_frame, text=_("Live"), cursor="hand2", state="disabled",
    )
    live_button.grid(row=0, column=0, sticky="ew", padx=3)  # same cell as start_button
    ToolTip(live_button, _("Start real-time face swap using webcam"))
    _live_button_widget = live_button

    stop_button = ctk.CTkButton(
        action_frame, text=_("Destroy"), cursor="hand2",
        command=lambda: destroy(),
    )
    stop_button.grid(row=0, column=1, sticky="ew", padx=3)
    ToolTip(stop_button, _("Stop processing and close the application"))

    preview_button = ctk.CTkButton(
        action_frame, text=_("Preview"), cursor="hand2",
        command=lambda: toggle_preview(),
    )
    preview_button.grid(row=1, column=0, columnspan=2, sticky="ew", padx=3, pady=(3, 0))
    ToolTip(preview_button, _("Show/hide a preview of the processed output"))

    return live_button


def _add_status_bar(root: ctk.CTk) -> None:
    global status_label, _download_progress_bar

    status_frame = ctk.CTkFrame(root, fg_color="transparent")
    status_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=(0, 5))
    status_frame.columnconfigure(0, weight=1)
    status_frame.columnconfigure(1, weight=1)

    status_label = ctk.CTkLabel(status_frame, text="", justify="left")
    status_label.grid(row=0, column=0, sticky="w")

    _download_progress_bar = ctk.CTkProgressBar(status_frame, width=200)
    _download_progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
    _download_progress_bar.set(0)
    _download_progress_bar.grid_remove()

    donate_label = ctk.CTkLabel(
        status_frame, text="Deep Live Cam", justify="right", cursor="hand2",
    )
    donate_label.grid(row=0, column=1, sticky="e")
    donate_label.configure(
        text_color=(ctk.ThemeManager.theme.get("URL") or {}).get("text_color")
    )
    donate_label.bind(
        "<Button>", lambda event: webbrowser.open("https://deeplivecam.net")
    )


def _add_mode_toggle(root: ctk.CTk) -> None:
    """Add the Live Cam / File Processing segmented button at row 0."""
    frame = ctk.CTkFrame(root, fg_color="transparent")
    frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
    frame.columnconfigure(0, weight=1)
    btn = ctk.CTkSegmentedButton(
        frame,
        values=[_("Live Cam"), _("File Processing")],
        command=_on_mode_change,
    )
    btn.set(_current_mode)
    btn.grid(row=0, column=0, sticky="ew", padx=5, pady=5)


def _on_mode_change(mode: str) -> None:
    global _current_mode
    _current_mode = mode
    is_live = mode == _("Live Cam")

    if _target_frame_widget:
        if is_live:
            _target_frame_widget.grid_remove()
        else:
            _target_frame_widget.grid()

    if _swap_button_widget:
        if is_live:
            _swap_button_widget.grid_remove()
        else:
            _swap_button_widget.grid()

    if _start_button_widget:
        if is_live:
            _start_button_widget.grid_remove()
        else:
            _start_button_widget.grid()

    if _live_button_widget:
        if is_live:
            _live_button_widget.grid()
        else:
            _live_button_widget.grid_remove()

    if _settings_tabview_ref:
        _settings_tabview_ref.set("Live Mode" if is_live else "Export")

    save_switch_states()


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    load_switch_states()
    root = _setup_window(destroy)
    _add_mode_toggle(root)
    _add_top_frame(root)
    _add_embedded_preview(root)
    live_button = _add_action_buttons(root, start, destroy)
    _add_settings_tabview(root, live_button)
    _add_status_bar(root)
    _restore_recent_paths()
    _on_mode_change(_current_mode)
    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_slider, _popout_label

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title(_("Preview"))
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", pop_in_preview)
    preview.resizable(width=True, height=True)

    # Pop-in button in top-right
    header = ctk.CTkFrame(preview, fg_color="transparent")
    header.pack(fill="x", side="top", padx=5, pady=(5, 0))
    ctk.CTkButton(
        header, text="⤡ Pop In", width=80, cursor="hand2",
        command=pop_in_preview,
    ).pack(side="right")

    _popout_label = ctk.CTkLabel(preview, text="")
    _popout_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(int(frame_value))  # type: ignore[arg-type]
    )

    return preview


def pop_out_preview() -> None:
    """Move preview display from the embedded panel into the floating PREVIEW window."""
    global preview_label, _preview_embedded
    assert _popout_label is not None and _embedded_label is not None
    assert _embedded_preview_frame is not None and ROOT is not None and PREVIEW is not None
    _preview_embedded = False
    img = getattr(_embedded_label, '_ctk_img', None)
    if img:
        setattr(_popout_label, '_ctk_img', img)
        _popout_label.configure(image=img)
        _embedded_label.configure(image=None)
    preview_label = _popout_label
    _embedded_preview_frame.grid_remove()
    ROOT.columnconfigure(1, weight=0)
    PREVIEW.deiconify()
    PREVIEW.focus()


def pop_in_preview() -> None:
    """Move preview display from the floating PREVIEW window back into the embedded panel."""
    global preview_label, _preview_embedded
    assert _popout_label is not None and _embedded_label is not None
    assert _embedded_preview_frame is not None and ROOT is not None and PREVIEW is not None
    _preview_embedded = True
    img = getattr(_popout_label, '_ctk_img', None)
    if img:
        setattr(_embedded_label, '_ctk_img', img)
        _embedded_label.configure(image=img)
    preview_label = _embedded_label
    _embedded_preview_frame.grid()
    ROOT.columnconfigure(1, weight=1)
    PREVIEW.withdraw()


def update_status(text: str) -> None:
    # May be called from background threads (e.g. face swapper model loading)
    # before ROOT is initialized. Guard against None to avoid AttributeError.
    if ROOT is None:
        return
    # Tkinter is not thread-safe: schedule the label update on the main thread.
    if status_label is None:
        return
    lbl = status_label
    ROOT.after(0, lambda t=text, l=lbl: l.configure(text=_(t)))


def download_progress_callback(filename: str, downloaded: int, total: int) -> None:
    """Called from background download thread. Throttles UI updates to ~10 Hz."""
    global _last_progress_update
    now = time.monotonic()
    if now - _last_progress_update < 0.1 and downloaded < total and downloaded > 0:
        return
    _last_progress_update = now

    if ROOT is None:
        return
    if total > 0 and downloaded < total:
        progress_value = downloaded / total
        ROOT.after(0, lambda f=filename, v=progress_value: _show_download_progress(f, v))
    else:
        ROOT.after(0, _hide_download_progress)


def _show_download_progress(filename: str, value: float) -> None:
    """Show and update the download progress bar (main thread only)."""
    if _download_progress_bar is None or status_label is None:
        return
    _download_progress_bar.set(value)
    _download_progress_bar.grid()
    pct = int(value * 100)
    lbl = status_label
    lbl.configure(text=f"Downloading {filename}... {pct}%")


def _hide_download_progress() -> None:
    """Hide the download progress bar (main thread only)."""
    if _download_progress_bar is None:
        return
    _download_progress_bar.grid_remove()


# Enhancer processor names — keys in fp_ui and corresponding frame_processor names
_ENHANCER_KEYS = ('face_enhancer', 'face_enhancer_gpen256', 'face_enhancer_gpen512', 'face_enhancer_codeformer')

# Map from processor NAME constant to fp_ui key for live-mode gating
_ENHANCER_NAME_TO_UI_KEY = {
    "DLC.FACE-ENHANCER": "face_enhancer",
    "DLC.FACE-ENHANCER-GPEN256": "face_enhancer_gpen256",
    "DLC.FACE-ENHANCER-GPEN512": "face_enhancer_gpen512",
    "DLC.FACE-ENHANCER-CODEFORMER": "face_enhancer_codeformer",
}


def _sync_enhancer_frame_processors() -> None:
    """Keep modules.globals.frame_processors in sync with fp_ui enhancer toggles.

    Non-enhancer processors (face_swapper, face_masking, etc.) are preserved;
    the enhancer slots are rebuilt from the current fp_ui state.
    """
    non_enhancers = [p for p in modules.globals.frame_processors if p not in _ENHANCER_KEYS]
    enabled = [k for k in _ENHANCER_KEYS if modules.globals.fp_ui.get(k, False)]
    modules.globals.frame_processors = non_enhancers + enabled


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    if var in _ENHANCER_KEYS:
        _sync_enhancer_frame_processors()
    save_switch_states()
    if _preview_embedded or (PREVIEW is not None and PREVIEW.state() == "normal"):
        global frame_processors
        frame_processors = get_frame_processors_modules(
            modules.globals.frame_processors
        )


def select_source_path() -> None:
    """Open file dialog and set source face for the first mapping entry.

    Delegates to MappingListWidget's select handler for the first entry,
    keeping backward compatibility with modules.globals.source_path.
    """
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    assert PREVIEW is not None
    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(
        title=_("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )
    if is_image(source_path):
        cv2_img = cv2.imread(source_path)
        face = get_one_face(cv2_img)
        if face is not None:
            x_min, y_min, x_max, y_max = face["bbox"]
            cropped = cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)]
            entries = MAPPING_LIST.get_entries()
            entry_id = entries[0].id if entries else 0
            MAPPING_LIST.set_source(entry_id, source_path, cropped, face)
            RECENT_DIRECTORY_SOURCE = os.path.dirname(source_path)
    else:
        entries = MAPPING_LIST.get_entries()
        if entries:
            entry = entries[0]
            entry.source_path = None
            entry.source_face = None
            entry.source_cv2 = None
        modules.globals.source_path = None


def swap_faces_paths() -> None:
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET

    source_path = modules.globals.source_path
    target_path = modules.globals.target_path

    if not source_path or not target_path:
        return
    if not is_image(source_path) or not is_image(target_path):
        return

    # Swap: old target becomes new source (first mapping), old source becomes new target
    new_source_path = target_path
    new_target_path = source_path

    # Update target global
    modules.globals.target_path = new_target_path
    RECENT_DIRECTORY_TARGET = os.path.dirname(new_target_path)

    # Update source via MAPPING_LIST (triggers observer → syncs globals)
    cv2_img = cv2.imread(new_source_path)
    if cv2_img is not None:
        face = get_one_face(cv2_img)
        if face is not None:
            x_min, y_min, x_max, y_max = face["bbox"]
            cropped = cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)]
            entries = MAPPING_LIST.get_entries()
            entry_id = entries[0].id if entries else 0
            MAPPING_LIST.set_source(entry_id, new_source_path, cropped, face)
            RECENT_DIRECTORY_SOURCE = os.path.dirname(new_source_path)

    assert PREVIEW is not None and target_label is not None
    PREVIEW.withdraw()

    saved_target = modules.globals.target_path
    if saved_target:
        target_image = render_image_preview(saved_target, SIDEBAR_THUMB_SIZE)
        target_label.configure(image=target_image)
    save_switch_states()


def capture_target_from_camera() -> None:
    """Open a live camera preview window with a Capture button."""
    cap = cv2.VideoCapture(_selected_camera_index)
    if not cap.isOpened():
        update_status("Failed to open camera.")
        return

    capture_window = ctk.CTkToplevel(ROOT)
    capture_window.title(_("Camera Capture"))
    capture_window.minsize(480, 400)
    capture_window.resizable(width=True, height=True)
    capture_window.protocol("WM_DELETE_WINDOW", lambda: _close_capture())

    feed_label = ctk.CTkLabel(capture_window, text="")
    feed_label.pack(fill="both", expand=True, padx=5, pady=5)

    button_frame = ctk.CTkFrame(capture_window, fg_color="transparent")
    button_frame.pack(fill="x", padx=10, pady=(0, 10))
    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(1, weight=1)

    capture_btn = ctk.CTkButton(
        button_frame, text=_("Capture"), cursor="hand2",
        command=lambda: _do_capture(),
    )
    capture_btn.grid(row=0, column=0, sticky="ew", padx=5)

    cancel_btn = ctk.CTkButton(
        button_frame, text=_("Cancel"), cursor="hand2",
        command=lambda: _close_capture(),
    )
    cancel_btn.grid(row=0, column=1, sticky="ew", padx=5)

    # Mutable state shared between callbacks
    _running = [True]
    _last_frame = [None]

    def _update_feed():
        if not _running[0]:
            return
        ret, frame = cap.read()
        if ret and frame is not None:
            _last_frame[0] = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # Scale to fit the label while preserving aspect ratio
            w = feed_label.winfo_width() or 480
            h = feed_label.winfo_height() or 360
            pil_img = ImageOps.contain(pil_img, (max(w, 1), max(h, 1)), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(pil_img, size=pil_img.size)
            feed_label.configure(image=ctk_img)
            feed_label._ctk_img = ctk_img  # prevent GC
        capture_window.after(33, _update_feed)  # ~30 fps

    def _do_capture():
        frame = _last_frame[0]
        if frame is None:
            update_status("No frame captured yet.")
            return
        _close_capture()

        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        capture_path = os.path.join(tmp_dir, "camera_capture.png")
        cv2.imwrite(capture_path, frame)

        global RECENT_DIRECTORY_TARGET
        modules.globals.target_path = capture_path
        RECENT_DIRECTORY_TARGET = tmp_dir
        image = render_image_preview(capture_path, SIDEBAR_THUMB_SIZE)
        assert target_label is not None
        target_label.configure(image=image)
        save_switch_states()
        update_status("Camera capture set as target.")

    def _close_capture():
        _running[0] = False
        cap.release()
        capture_window.destroy()

    # Start the feed after a short delay so the window has geometry
    capture_window.after(100, _update_feed)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    assert PREVIEW is not None
    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(
        title=_("select an target image or video"),
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft, vid_ft],
    )
    assert target_label is not None
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, SIDEBAR_THUMB_SIZE)
        target_label.configure(image=image)
        save_switch_states()
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, SIDEBAR_THUMB_SIZE)
        target_label.configure(image=video_frame)
        save_switch_states()
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    target_path = modules.globals.target_path
    if target_path is None:
        return
    if is_image(target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=_("save image output file"),
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=_("save video output file"),
            filetypes=[vid_ft],
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        # Snapshot globals at the moment Start is clicked; processing uses this config.
        from modules.processing_config_factory import build_config_from_globals as _bcfg
        start(config=_bcfg())


def fit_image_to_size(image, width: int, height: int):
    if width is None and height is None:
        return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    if width > height:
        ratio_h = height / h
    else:
        ratio_w = width / w
    ratio = max(ratio_w, ratio_h)
    new_size = (int(ratio * w), int(ratio * h))
    return gpu_resize(image, dsize=new_size)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
        video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if has_frame:
            image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
            if size:
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            return ctk.CTkImage(image, size=image.size)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def toggle_preview() -> None:
    if _preview_embedded:
        if modules.globals.source_path and modules.globals.target_path:
            init_preview()
            update_preview()
    else:
        assert PREVIEW is not None
        if PREVIEW.state() == "normal":
            PREVIEW.withdraw()
        elif modules.globals.source_path and modules.globals.target_path:
            init_preview()
            update_preview()


def init_preview() -> None:
    assert preview_slider is not None
    # Slider is pop-out only — hide it in embedded mode
    if _preview_embedded:
        preview_slider.pack_forget()
        return
    target_path = modules.globals.target_path
    if target_path is None:
        return
    if is_image(target_path):
        preview_slider.pack_forget()
    if is_video(target_path):
        video_frame_total = get_video_frame_total(target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        for frame_processor in get_frame_processors_modules(
                modules.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
            )
        image = Image.fromarray(gpu_cvt_color(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.Resampling.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        assert preview_label is not None
        preview_label.configure(image=image)
        update_status("Processing succeed!")
        if not _preview_embedded:
            assert PREVIEW is not None
            PREVIEW.deiconify()






