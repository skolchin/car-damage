#-------------------------------------------------------------------------------
# Name:        Car damage annotation UI
# Purpose:     Car damage analysis project
#
# Author:      kol
#
# Created:     15.01.2020
# Copyright:   (c) kol 2020
# Licence:     MIT
#-------------------------------------------------------------------------------

import cv2
import numpy as np
import tkinter as tk
import json

from pathlib import Path
from imutils.perspective import order_points
from skimage.measure import compare_ssim, find_contours
from skimage.draw import polygon
from copy import deepcopy

from tkinter import filedialog
from tkinter import ttk

from gr.ui_extra import ImagePanel, ImgButton, ImageMask, ImageTransform, ImgButtonGroup
from gr.utils import get_image_area

from img_utils import *


class ImageData:
    def __init__(self, file_name = None):
        self.image = None
        self.file_name = None
        self.key = None

        if file_name is not None:
            self.load(file_name)

    def load(self, file_name):
        self.image = cv2.imread(file_name)
        if self.image is None:
            raise Exception('File not found', file_name)
        self.file_name = file_name
        self.key = str(Path(file_name).name)


class MetaData(dict):
    IMG_DIR = "img/"
    META_FILE = "meta.json"

    def __init__(self, *args, **kwargs):
        self.update(self, *args, **kwargs)
        self.load()

    def load(self):
        fn = Path(__file__).parent.joinpath(self.IMG_DIR, self.META_FILE)
        if fn.exists():
            with open(str(fn), "r") as f:
                p = json.load(f)
                self.update(p)
                f.close()

    def add(self, img_data):
        meta = self.get(img_data.key)
        if meta is None:
            meta = {'file_name': img_data.file_name}
            self.__setitem__(img_data.key, meta)
        return meta

    def save(self):
        fn = Path(__file__).parent.joinpath(self.IMG_DIR, self.META_FILE)
        with open(str(fn), "w+") as f:
            json.dump(self, f, indent=4, sort_keys=True, ensure_ascii=False)
            f.close()

    def clear(self, img_data):
        if self.get(img_data.key):
            meta = {'file_name': img_data.file_name}
            self.__setitem__(img_data.key, meta)
        return meta


class AnnotateApp(tk.Tk):
    # Constructor
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, "Annotate")

        self.image_data = None
        self.meta_data = MetaData()

        self.title("Annotate cars")
        self.minsize(300, 400)

        self.internalFrame = tk.Frame(self)
        self.internalFrame.pack(fill = tk.BOTH, expand = True)

        self.def_img = cv2.imread("ui\\def_image.png")
        self.__init_menu()
        self.__init_toolbar()
        self.__init_statusbar()
        self.__init_window()

    def __init_menu(self):
        pass

    def __init_toolbar(self):
        self.toolbarPanel = tk.Frame(self.internalFrame, bd = 1, relief = tk.RAISED)
        self.toolbarPanel.pack(side = tk.TOP, fill = tk.X, expand = False)

        ImgButton(self.toolbarPanel,
            tag = "open", tooltip = "Open image",
            command = self.open_image_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "prev", tooltip = "Prev image",
            command = self.prev_image_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "next", tooltip = "Next image",
            command = self.next_image_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "split", tooltip = "Split image",
            command = self.split_image_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "area", tooltip = "Select damage area",
            command = self.transform_start_callback).pack(side = tk.LEFT, padx = 2, pady = 2)

        ImgButton(self.toolbarPanel,
            tag = "reset", tooltip = "Reset areas",
            command = self.reset_callback).pack(side = tk.LEFT, padx = 2, pady = 2)


        self.bg = ImgButtonGroup(self.toolbarPanel)
        self.bg.add_group('has_file', ['split', 'area', 'reset'])
        self.bg.add_group('transform', ['split', 'area'], ImgButtonGroup.BG_DEPENDENT)
        self.bg['has_file'].disabled = True

    def __init_statusbar(self):
        pass

    def __init_window(self):
        left_frame = tk.Frame(self.internalFrame, bd=1, relief=tk.GROOVE)
        left_frame.pack(side = tk.LEFT, fill = tk.BOTH, expand = True, pady=2)

        right_frame = tk.Frame(self.internalFrame, bd=1, relief=tk.GROOVE)
        right_frame.pack(side = tk.TOP, fill = tk.Y, expand = True, pady=2)

        # Image panel
        self.imagePanel = ImagePanel(left_frame,
            image = self.def_img,
            mode = "fit",
            max_size = 800)
        self.imagePanel.pack(side = tk.TOP, fill = tk.BOTH, expand = True,
            padx = 5, pady = 5)

        # Left/right images splitter
        self.imageSplit = ImageMask(self.imagePanel,
            allow_change = True,
            show_mask = False,
            mode = 'split',
            mask_callback = self.split_mask_callback)

        self.imageSplit.mask_color = 'green'
        self.imageSplit.mask_width = 1

        # Common transformer
        self.transform = ImageTransform(
            self.imagePanel,
            inplace = False,
            show_coord = False,
            keep = False,
            connect = True,
            allow_change = False,
            callback = self.transform_callback)

        # Panels to preview selection and area transformer objects
        self.selectedAreas = {}
        self.previewImages = {}
        self.transforms = {}

        def add_area(label, ncol):
            # Preview selection (non-transformed)
            area_frame = tk.Frame(right_frame)
            area_frame.grid(column = ncol, row = 0, sticky = "nswe")

            tk.Label(area_frame, text = label.title() + " image").pack(
                side = tk.TOP, fill = tk.Y, pady = 2, padx = 2)
            self.selectedAreas[label] = ImagePanel(area_frame,
                image = self.def_img,
                mode = "fit",
                max_size = 200,
                bd=1, relief=tk.GROOVE,
                frame_callback = self.preview_callback)
            self.selectedAreas[label].pack(side = tk.TOP, fill = tk.BOTH, expand = True,
                padx = 2, pady = 2)

            # Transformed selection
            area_frame = tk.Frame(right_frame)
            area_frame.grid(column = ncol, row = 1, sticky = "nswe")

            tk.Label(area_frame, text = label.title() + " image normalized").pack(
                side = tk.TOP, fill = tk.Y, pady = 2, padx = 2)
            self.previewImages[label] = ImagePanel(area_frame,
                image = cv2.imread("ui\\def_image.png"),
                mode = "fit",
                max_size = 200,
                bd=1, relief=tk.GROOVE,
                frame_callback = self.preview_callback)
            self.previewImages[label].pack(side = tk.TOP, fill = tk.BOTH, expand = True,
                padx = 2, pady = 2)

            # Transformers
            self.transforms[label] = ImageTransform(
                self.imagePanel,
                inplace = False,
                show_coord = False,
                keep = True,
                connect = True,
                allow_change = True,
                callback = self.transform_callback)

        add_area('left', 0)
        add_area('right', 1)

        # Differences
        area_frame = tk.Frame(right_frame)
        area_frame.grid(columnspan = 2, row = 2, sticky = "nswe")

        self.diffScore = tk.Label(area_frame, text = "Images similarity score")
        self.diffScore.pack(side = tk.TOP, fill = tk.Y, pady = 2, padx = 2)

        # Diff between transformed images
        self.diffArea = ImagePanel(area_frame,
            image = cv2.imread("ui\\def_image.png"),
            mode = "fit",
            max_size = 200,
            bd=1, relief=tk.GROOVE,
            frame_callback = self.preview_callback)
        self.diffArea.pack(side = tk.LEFT, fill = tk.BOTH, expand = True,
            padx = 2, pady = 2)

        # Diff applied to damaged image
        self.diffResult = ImagePanel(area_frame,
            image = cv2.imread("ui\\def_image.png"),
            mode = "fit",
            max_size = 200,
            bd=1, relief=tk.GROOVE,
            frame_callback = self.preview_callback)
        self.diffResult.pack(side = tk.LEFT, fill = tk.BOTH, expand = True,
            padx = 2, pady = 2)

        area_frame = tk.Frame(right_frame)
        area_frame.grid(columnspan = 2, row = 3, sticky = "nswe")

        self.varFilter = tk.IntVar()
        self.varFilter.set(0)
        ttk.Checkbutton(area_frame, text="Filter",
                        variable=self.varFilter,
                        command=self.apply_filter_changed).pack(side = tk.LEFT,
                        padx = 2, pady = 5)


    @property
    def buttons(self):
        r = {}
        for w in self.toolbarPanel.winfo_children():
            if isinstance(w, ImgButton):
                r[w.tag] = w
        return r

    def open_image_callback(self, event):
        fn = filedialog.askopenfilename(title = "Select file",
           filetypes = (("JPEG files","*.jpg"),("PNG files","*.png"),("All files","*.*")))
        if fn != "":
            self.load_image(fn)
        event.cancel = True

    def next_image_callback(self, event):
        self.change_file(1)
        event.cancel = True

    def prev_image_callback(self, event):
        self.change_file(-1)
        event.cancel = True

    def split_image_callback(self, event):
        if self.image_data is None:
            event.cancel = True
            return

        if event.state:
            self.imageSplit.show()
            if not 'split' in self.meta_data[self.image_data.key]:
                self.meta_data[self.image_data.key]['split'] = self.imageSplit.scaled_mask[2]
                self.meta_data.save()
        else:
            self.imageSplit.hide()

    def transform_start_callback(self, event):
        if self.image_data is None:
            event.cancel = True
            return

        if event.state:
            has_rect = 0
            for t in self.transforms.values():
                if t.transform_rect is not None:
                    t.show()
                    has_rect += 1
            if has_rect < len(self.transforms):
                self.transform.start()
        else:
            if self.transform.started:
                self.transform.cancel()
            for t in self.transforms.values():
                t.hide()

    def reset_callback(self, event):
        event.cancel = True
        if self.image_data is None: return

        self.buttons['split'].release()
        self.buttons['area'].release()

        self.imageSplit.default_mask()
        self.transform.transform_rect = None

        for k, t in self.transforms.items():
            self.clear_preview(k)

        self.diffScore.configure(text = "Images similarity score")
        self.diffArea.image = self.def_img

        self.meta_data.clear(self.image_data)
        self.meta_data.save()

    def split_mask_callback(self, mask):
        self.meta_data[self.image_data.key]['split'] = mask.scaled_mask[2]
        self.meta_data.save()

    def transform_callback(self, transform, img):
        self.buttons['area'].state = False
        if img is not None:
            label = self.set_preview(img, transform.bounding_rect)
            self.meta_data[self.image_data.key][label] = transform.scaled_rect
            self.set_diff()

            if transform.tag != self.transforms[label].tag:
                self.transforms[label].scaled_rect = transform.scaled_rect
                self.transforms[label].show()

            self.meta_data.save()

    def preview_callback(self, event):
        def get_panel(widget):
            panel = widget
            while panel is not None and not isinstance(panel, ImagePanel):
                panel = panel.master
            return panel

        cv2.imshow('Preview', get_panel(event.widget).src_image)

    def apply_filter_changed(self):
        if self.image_data is None:
            return

        meta = self.meta_data[self.image_data.key]
        for t in self.transforms.values():
            self.set_preview(t.transform_image, t.bounding_rect)

        self.set_diff()

        apply_filter = self.varFilter.get() > 0
        self.meta_data[self.image_data.key]['filter'] = apply_filter
        self.meta_data.save()

    def load_image(self, file_name):
        self.bg['has_file'].release()

        self.image_data = ImageData(str(file_name))
        meta = self.meta_data.add(self.image_data)

        self.imagePanel.image = self.image_data.image
        self.imageSplit.default_mask()

        if 'split' in meta:
            x = meta['split']
            m = [x, 0, x, self.imageSplit.scaled_mask[3]]
            self.imageSplit.scaled_mask = m

        apply_filter = False
        if 'filter' in meta:
            apply_filter = self.meta_data[self.image_data.key]['filter']
        self.varFilter.set(apply_filter)

        for k, t in self.transforms.items():
            if not k in meta:
                self.clear_preview(k)
            else:
                t.scaled_rect = meta[k]
                self.set_preview(t.transform_image, t.bounding_rect)

        self.set_diff()
        self.title('Annotate cars - ' + str(file_name))
        self.bg['has_file'].disabled = False

    def change_file(self, direction):
        path = Path(__file__).parent.joinpath(self.meta_data.IMG_DIR)
        names = [x for x in path.glob("*.jpg")]
        names.extend([x for x in path.glob("*.png")])
        if len(names) == 0:
            return

        names = sorted(names)
        if self.image_data is None:
            self.load_image(names[0] if direction > 0 else names[-1])
        else:
            for i, f in enumerate(names):
                if f.name == self.image_data.key:
                    n = i + direction
                    if n < 0: n = len(names) - 1
                    elif n >= len(names): n = 0

                    self.load_image(names[n])
                    break

    def set_preview(self, img, bbox):
        h, w = self.image_data.image.shape[:2]

        min_x = max(min([x[0] for x in bbox]), 0)
        max_x = min(max([x[0] for x in bbox]), w)
        min_y = max(min([x[1] for x in bbox]), 0)
        max_y = min(max([x[1] for x in bbox]), h)
        m = [min_x, min_y, max_x, max_y]

        label = 'left' if max_x <= self.imageSplit.scaled_mask[2] else 'right'
        self.previewImages[label].image = img

        area_img = get_image_area(self.image_data.image, m)
        self.selectedAreas[label].image = area_img

        return label

    def clear_preview(self, label):
        self.previewImages[label].image = self.def_img
        self.selectedAreas[label].image = self.def_img
        self.transforms[label].hide()
        self.transforms[label].transform_rect = None

    def set_diff(self):
        def clean_up():
            self.diffScore.configure(text = "Images similarity score")
            self.diffArea.image = self.def_img
            self.diffResult.image = self.def_img

        if self.image_data is None:
            clean_up()
            return False

        meta = self.meta_data[self.image_data.key]
        for k in ['split', 'left', 'right']:
            if not k in meta:
                clean_up()
                return False

        apply_filter = self.varFilter.get() > 0
        score, diff, result = get_diff(self.image_data.image, meta, apply_filter=apply_filter)
        self.diffScore.configure(text = "Images similarity score: {}%".format(np.round(score*100, 2)))
        self.diffArea.image = diff
        self.diffResult.image = result

        return True

# Main function
def main():
    window = AnnotateApp()
    window.mainloop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
