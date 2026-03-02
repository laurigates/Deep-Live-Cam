"""Inline mapping list widget for the sidebar (Unified Face Mapping UI).

Renders a scrollable list of source→pin mapping entries.  Single-mapping
mode shows 120x120 thumbnails (matching current UI).  Multi-mapping mode
uses 80x80 thumbnails with row labels.
"""
from __future__ import annotations

from typing import Callable

import cv2
import customtkinter as ctk
from PIL import Image, ImageOps

from modules.mapping_list import MappingList


# Thumbnail sizes
_SINGLE_THUMB = (120, 120)
_MULTI_THUMB = (80, 80)


class MappingListWidget:
    """Renders and manages the mapping list in the sidebar.

    Subscribes to ``mapping_list.on_change`` so any mutation triggers a
    rebuild of the widget tree.  All widget updates are on the main thread.
    """

    def __init__(
        self,
        parent: ctk.CTkFrame,
        mapping_list: MappingList,
        on_change_callback: Callable[[], None] | None = None,
    ) -> None:
        self._parent = parent
        self._mapping_list = mapping_list
        self._on_change_cb = on_change_callback
        self._frame: ctk.CTkFrame | None = None
        self._mapping_list.on_change(self._rebuild)
        self._rebuild()

    # ------------------------------------------------------------------
    # Public helpers (used by tests and host code)
    # ------------------------------------------------------------------

    def _should_show_remove(self) -> bool:
        return len(self._mapping_list.get_entries()) > 1

    def _thumb_size(self) -> tuple[int, int]:
        if len(self._mapping_list.get_entries()) > 1:
            return _MULTI_THUMB
        return _SINGLE_THUMB

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Destroy and recreate the widget tree from current MappingList state."""
        if self._frame is not None:
            self._frame.destroy()

        self._frame = ctk.CTkFrame(self._parent, fg_color="transparent")
        self._frame.pack(fill="both", expand=True)
        self._frame.columnconfigure(0, weight=1)

        entries = self._mapping_list.get_entries()
        thumb = self._thumb_size()
        show_remove = self._should_show_remove()

        for row_idx, entry in enumerate(entries):
            self._build_entry_row(self._frame, entry, row_idx, thumb, show_remove)

        # "+ Add mapping" button
        add_btn = ctk.CTkButton(
            self._frame,
            text="+ Add mapping",
            cursor="hand2",
            command=self._on_add,
            height=28,
        )
        add_btn.grid(
            row=len(entries), column=0, pady=(5, 0), sticky="ew", padx=5,
        )

    def _build_entry_row(
        self,
        parent: ctk.CTkFrame,
        entry,
        row: int,
        thumb: tuple[int, int],
        show_remove: bool,
    ) -> None:
        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.grid(row=row, column=0, sticky="ew", padx=2, pady=2)
        row_frame.columnconfigure(0, weight=1)

        # Header with optional remove button
        if show_remove:
            header = ctk.CTkFrame(row_frame, fg_color="transparent")
            header.grid(row=0, column=0, sticky="ew")
            header.columnconfigure(0, weight=1)
            ctk.CTkLabel(header, text=f"Mapping {entry.id + 1}", anchor="w").grid(
                row=0, column=0, sticky="w", padx=5,
            )
            remove_btn = ctk.CTkButton(
                header, text="\u00d7", width=24, height=24, cursor="hand2",
                command=lambda eid=entry.id: self._on_remove(eid),
            )
            remove_btn.grid(row=0, column=1, padx=2)

        # Source thumbnail
        src_label = ctk.CTkLabel(row_frame, text=None, width=thumb[0], height=thumb[1])
        src_label.grid(row=1, column=0, pady=(2, 2))
        if entry.source_cv2 is not None:
            self._set_thumbnail(src_label, entry.source_cv2, thumb)

        # Select face button
        select_btn = ctk.CTkButton(
            row_frame, text="Select a face", cursor="hand2",
            command=lambda eid=entry.id: self._on_select_source(eid),
        )
        select_btn.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 2))

        # Pin section (optional target reference)
        if entry.pin_cv2 is not None:
            pin_label = ctk.CTkLabel(row_frame, text=None, width=thumb[0], height=thumb[1])
            pin_label.grid(row=3, column=0, pady=(2, 2))
            self._set_thumbnail(pin_label, entry.pin_cv2, thumb)

        pin_btn = ctk.CTkButton(
            row_frame, text="Pin to face", cursor="hand2", height=24,
            command=lambda eid=entry.id: self._on_select_pin(eid),
        )
        pin_btn.grid(row=4, column=0, sticky="ew", padx=5, pady=(0, 2))

        # Separator (if multi)
        if show_remove:
            sep = ctk.CTkFrame(row_frame, height=1, fg_color="gray50")
            sep.grid(row=5, column=0, sticky="ew", padx=10, pady=4)

    @staticmethod
    def _set_thumbnail(
        label: ctk.CTkLabel, cv2_img, size: tuple[int, int],
    ) -> None:
        """Set a CTkImage on a label from a BGR cv2 image."""
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        ctk_img = ctk.CTkImage(image, size=size)
        label.configure(image=ctk_img)
        # Keep reference to prevent GC
        label._ctk_img = ctk_img

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_add(self) -> None:
        self._mapping_list.add_entry()

    def _on_remove(self, entry_id: int) -> None:
        self._mapping_list.remove_entry(entry_id)

    def _on_select_source(self, entry_id: int) -> None:
        """Open file dialog and set source face for the given entry."""
        from modules.face_analyser import get_one_face

        path = ctk.filedialog.askopenfilename(
            title="Select a source face image",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.gif *.bmp")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        face = get_one_face(img)
        if face is None:
            return
        # Crop face region for thumbnail
        x_min, y_min, x_max, y_max = face["bbox"]
        cropped = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        self._mapping_list.set_source(entry_id, path, cropped, face)

    def _on_select_pin(self, entry_id: int) -> None:
        """Open file dialog and set pin face for the given entry."""
        from modules.face_analyser import get_one_face

        path = ctk.filedialog.askopenfilename(
            title="Select a target reference face image",
            filetypes=[("Image", "*.png *.jpg *.jpeg *.gif *.bmp")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        face = get_one_face(img)
        if face is None:
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        cropped = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        self._mapping_list.set_pin(entry_id, path, cropped, face)
