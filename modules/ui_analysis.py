from modules.utilities import has_image_extension


def analyze_target(start, root):
    import modules.globals
    from modules.face_analyser import (
        get_unique_faces_from_target_image,
        get_unique_faces_from_target_video,
    )
    from modules.mapping_list import MAPPING_LIST
    from modules.ui import (
        POPUP,
        create_source_target_popup,
        select_output_path,
        update_status,
    )
    from modules.utilities import is_image, is_video

    if POPUP is not None and POPUP.winfo_exists():
        update_status("Please complete pop-up or close it.")
        return

    if MAPPING_LIST.effective_map_faces():
        from modules.face_map_store import STORE as _MAP_STORE

        _MAP_STORE.clear()

        if is_image(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_image()
        elif is_video(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_video()

        entries = _MAP_STORE.get_entries()
        if len(entries) > 0:
            create_source_target_popup(start, root, entries)
        else:
            update_status("No faces found in target")
    else:
        select_output_path(start)


def check_and_ignore_nsfw(target, destroy=None):
    """Check if the target is NSFW.
    TODO: Consider to make blur the target.
    """
    from numpy import ndarray

    from modules.predicter import predict_frame, predict_image, predict_video
    from modules.ui import update_status

    check_nsfw = None
    if isinstance(target, str):  # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):  # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)  # Do not need to destroy the window frame if the target is NSFW
        update_status("Processing ignored!")
        return True
    else:
        return False
