# -*- coding:utf-8 -*-

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import QThread

import shutil
import sys
import os
import time
import random
import csv
import cv2
import yaml
import re
import numpy as np
from datetime import datetime
from MR_TLDetector.paths import application_path, output_root, resource_path
from MR_TLDetector.restoration import (
    OneRestoreRunner,
    RestormerDerainRunner,
    SnowFormerDesnowRunner,
    WeatherArtifactCleaner,
)

# Keep Ultralytics configuration in the writable output area.
os.environ.setdefault("YOLO_CONFIG_DIR", str(output_root() / ".ultralytics"))
from ultralytics import YOLO

# Keep the generated UI definition separate from application logic.
main_ui, _ = loadUiType(str(resource_path("UI", "main.ui")))


class MainGui(QMainWindow, main_ui):
    preview_requested = QtCore.pyqtSignal(object)

    # Initialize the main window and its runtime state.
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("MR_TLDetector")
        self.label_2.setText(
            "Meteorological Resilient Transmission Line Defect Detection System"
        )
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowMinMaxButtonsHint | QtCore.Qt.WindowCloseButtonHint)
        self.setMinimumSize(960, 720)
        self.label_img.setScaledContents(False)
        self.label_img.setAlignment(QtCore.Qt.AlignCenter)
        self.pushButton_export.setText("Export Data")
        self.pushButton_end.setEnabled(True)
        self._base_window_size = QtCore.QSize(self.size())
        self._base_widget_geometries = []
        self._remember_widget_geometries(self.centralwidget)

        # ----------------------------- UI icons -----------------------------
        # Configure the application and control icons.
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(str(resource_path("icon", "app.png"))), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(str(resource_path("icon", "dirs.png"))), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_img.setIcon(icon1)

        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(str(resource_path("icon", "dir.png"))), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_dir.setIcon(icon2)

        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(str(resource_path("icon", "shipinwenjian.png"))), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_video.setIcon(icon3)

        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(str(resource_path("icon", "shexiangtou.png"))), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_cap.setIcon(icon4)

        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(str(resource_path("icon", "data.png"))), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_data.setIcon(icon5)

        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(str(resource_path("icon", "weights.png"))), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_weights.setIcon(icon6)

        palette = self.palette()
        self._background_pixmap = QtGui.QPixmap(str(resource_path("icon", "backgroung.png")))
        palette.setColor(QPalette.Window, QtCore.Qt.transparent)
        self.setPalette(palette)

        self.label_4.setPixmap(QtGui.QPixmap(str(resource_path("icon", "IOU.png"))))
        self.label_5.setPixmap(QtGui.QPixmap(str(resource_path("icon", "zhixindu.png"))))
        self.label_11.setPixmap(QtGui.QPixmap(str(resource_path("icon", "yongshi.png"))))
        self.label_14.setPixmap(QtGui.QPixmap(str(resource_path("icon", "zhixindu.png"))))
        self.label_15.setPixmap(QtGui.QPixmap(str(resource_path("icon", "leibie.png"))))
        self.label_17.setPixmap(QtGui.QPixmap(str(resource_path("icon", "weizhi.png"))))
        self.label_22.setPixmap(QtGui.QPixmap(str(resource_path("icon", "image_size.png"))))
        self.label_25.setPixmap(QtGui.QPixmap(str(resource_path("icon", "preprocess.png"))))

        self.label_width, self.label_height = self.label_img.size().width(), self.label_img.size().height()
        self._base_title_font_size = 26
        self._base_body_font_size = 11
        self._apply_fonts()

        # Name of the current source file.
        self.img_name = None
        # Path used to save the annotated image.
        self.result_img_name = None
        # Input type: image, directory, video, or camera.
        self.start_type = None
        # Keep Stop available; it is a no-op when no task is running.
        self.pushButton_end.setEnabled(True)

        self.img_path = None
        self.img_path_dir = None

        self.video = None
        self.video_path = None
        self.cap = None
        self.lineEdit.setText("640")
        self.worker_thread = None
        self._closing_after_worker = False
        self._preview_image = None
        self._detection_base_image = None
        self.preview_requested.connect(self._set_preview_image)

        # Annotation text color.
        self.color = {"font": (255, 255, 255)}

        self.all_result = []
        self.comboBox_name = []
        self.yaml_classes = []
        self.preprocess_methods = []

        self.selected_text = None
        self.number = 1
        self.RowLength = 0
        self.input_time = 0

        # ---------------------------- Output paths ----------------------------
        # Retain the working directory for compatibility with existing workflows.
        self.ProjectPath = os.getcwd()

        # Store all generated output under a writable runtime directory.
        self.output_dir = str(output_root())

        run_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.result_time_path = os.path.join(self.output_dir, run_time)
        os.mkdir(self.result_time_path)

        self.result_img_path = os.path.join(self.result_time_path, 'img_result')
        os.mkdir(self.result_img_path)
        # Store copies of the original source images.
        self.result_org_img_path = os.path.join(self.result_time_path, 'org_img')
        os.mkdir(self.result_org_img_path)

        # --------------------------- Model parameters -------------------------
        # The lightweight edition always performs FP32 inference on the CPU.
        self.device = "cpu"
        # Editable default model files are stored beside the packaged executable.
        default_data = application_path("config", "mydata.yaml")
        default_weights = application_path("models", "best.pt")
        self.data_file_name = str(default_data) if default_data.is_file() else ""
        self.weights_file_name = str(default_weights) if default_weights.is_file() else ""
        # Loaded inference model.
        self.model = None
        self.model_error = None

        # ------------------------- Mutable parameters -------------------------
        self.save_parameter = {}
        self.pre_parameter = {}

        if self.data_file_name:
            self.label_13.setText(default_data.name)
            self._load_yaml_classes(self.data_file_name)
        if self.weights_file_name:
            self.label_12.setText(default_weights.name)
            self.save_parameter["weights"] = self.weights_file_name

        # Initialize table sizing and signal connections.
        self.update_table_width()
        self.handle_buttons()
        # Build preprocessing options after all widgets and signals are initialized.
        self._setup_preprocess_selector()
        self.update_layout_for_size()

    def _setup_preprocess_selector(self):
        """Initialize the multi-select image preprocessing control."""
        self.comboBox.clear()
        self.comboBox.setEditable(True)
        self.comboBox.lineEdit().setReadOnly(True)
        self.comboBox.lineEdit().setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.comboBox.setMaxVisibleItems(7)
        # Use a dedicated view so the UI file cannot clip the popup to one row.
        preprocess_view = QtWidgets.QListView(self.comboBox)
        preprocess_view.setUniformItemSizes(True)
        preprocess_view.setMinimumHeight(7 * 32)
        preprocess_view.setMaximumHeight(7 * 40)
        self.comboBox.setView(preprocess_view)
        self._preprocess_click_targets = [self.comboBox]
        self._preprocess_click_targets.extend(self.comboBox.findChildren(QtWidgets.QWidget))
        for click_target in self._preprocess_click_targets:
            click_target.installEventFilter(self)
        self._object_click_targets = [self.comboBox_2]
        self._object_click_targets.extend(self.comboBox_2.findChildren(QtWidgets.QWidget))
        for click_target in self._object_click_targets:
            click_target.installEventFilter(self)

        # The floating popup overlays controls below without changing panel layout.
        self.preprocess_popup = QtWidgets.QFrame(None, QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.preprocess_popup.setStyleSheet(
            "QFrame { background: white; border: 1px solid #8d979f; }"
            "QListWidget { border: none; background: white; color: #6f7f8c; }"
            "QListWidget::item { height: 32px; padding-left: 8px; }"
            "QListWidget::item:selected { background: #087dcc; color: white; }"
        )
        popup_layout = QtWidgets.QVBoxLayout(self.preprocess_popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        self.preprocess_list = QtWidgets.QListWidget(self.preprocess_popup)
        self.preprocess_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        # Preserve popup size and show a scrollbar when options exceed five rows.
        self.preprocess_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.preprocess_list.itemPressed.connect(self._toggle_preprocess_popup_item)
        popup_layout.addWidget(self.preprocess_list)

        model = QtGui.QStandardItemModel(self.comboBox)
        desnow_option = (
            "SnowFormer Desnow"
            if SnowFormerDesnowRunner.is_model_available()
            else "Fast Desnow (Snow Spots)"
        )
        options = [
            "None",
            "DCP (Dehazing)",
            "Restormer Derain",
            desnow_option,
            "LIME-Lite",
            "CLAHE (Local Contrast)",
            "OneRestore Auto",
        ]
        for row, text in enumerate(options):
            item = QtGui.QStandardItem(text)
            item.setFlags(
                QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable
            )
            item.setCheckState(QtCore.Qt.Checked if row == 0 else QtCore.Qt.Unchecked)
            model.appendRow(item)
            popup_item = QtWidgets.QListWidgetItem(text, self.preprocess_list)
            popup_item.setFlags(
                QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable
            )
            popup_item.setCheckState(QtCore.Qt.Checked if row == 0 else QtCore.Qt.Unchecked)
        self.comboBox.setModel(model)
        self.comboBox.view().setMinimumHeight(7 * 32)
        self.comboBox.setCurrentIndex(0)
        self.comboBox.lineEdit().setText(options[0])

    def eventFilter(self, watched, event):
        if watched in self._preprocess_click_targets and event.type() == QtCore.QEvent.MouseButtonPress:
            self.comboBox_2.hidePopup()
            self._show_preprocess_popup()
            return True
        if watched in self._object_click_targets and event.type() == QtCore.QEvent.MouseButtonPress:
            self.preprocess_popup.hide()
        return super().eventFilter(watched, event)

    def _show_preprocess_popup(self):
        """Show preprocessing options without moving the controls below."""
        width = self.comboBox.width()
        height = 7 * 32
        top_left = self.comboBox.mapToGlobal(QtCore.QPoint(0, self.comboBox.height()))
        screen = QtWidgets.QApplication.screenAt(top_left)
        if screen is not None and top_left.y() + height > screen.availableGeometry().bottom():
            top_left.setY(self.comboBox.mapToGlobal(QtCore.QPoint(0, 0)).y() - height)
        self.preprocess_popup.setFixedSize(width, height)
        self.preprocess_popup.move(top_left)
        self.preprocess_popup.show()

    def _toggle_preprocess_popup_item(self, item):
        """Toggle a preprocessing option and synchronize the summary."""
        row = self.preprocess_list.row(item)
        if row == 0:
            for index in range(self.preprocess_list.count()):
                self.preprocess_list.item(index).setCheckState(
                    QtCore.Qt.Checked if index == 0 else QtCore.Qt.Unchecked
                )
        else:
            item.setCheckState(
                QtCore.Qt.Unchecked if item.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked
            )
            has_method = any(
                self.preprocess_list.item(index).checkState() == QtCore.Qt.Checked
                for index in range(1, self.preprocess_list.count())
            )
            self.preprocess_list.item(0).setCheckState(
                QtCore.Qt.Unchecked if has_method else QtCore.Qt.Checked
            )

        self.preprocess_methods = [
            self.preprocess_list.item(index).text()
            for index in range(1, self.preprocess_list.count())
            if self.preprocess_list.item(index).checkState() == QtCore.Qt.Checked
        ]
        for index in range(self.comboBox.model().rowCount()):
            self.comboBox.model().item(index).setCheckState(
                self.preprocess_list.item(index).checkState()
            )
        self.comboBox.lineEdit().setText(
            "、".join(self.preprocess_methods) if self.preprocess_methods else "None"
        )

    def _toggle_preprocess_item(self, index):
        """Toggle a preprocessing option and update the combo-box summary."""
        model = self.comboBox.model()
        item = model.itemFromIndex(index)
        if item is None:
            return

        if index.row() == 0:
            for row in range(model.rowCount()):
                model.item(row).setCheckState(QtCore.Qt.Checked if row == 0 else QtCore.Qt.Unchecked)
        else:
            item.setCheckState(
                QtCore.Qt.Unchecked if item.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked
            )
            has_method = any(
                model.item(row).checkState() == QtCore.Qt.Checked
                for row in range(1, model.rowCount())
            )
            model.item(0).setCheckState(QtCore.Qt.Unchecked if has_method else QtCore.Qt.Checked)

        self.preprocess_methods = [
            model.item(row).text()
            for row in range(1, model.rowCount())
            if model.item(row).checkState() == QtCore.Qt.Checked
        ]
        self.comboBox.lineEdit().setText(
            "、".join(self.preprocess_methods) if self.preprocess_methods else "None"
        )
        self.comboBox.hidePopup()

    def _needs_original_comparison(self):
        return any(
            method in (
                "Restormer Derain",
                "SnowFormer Desnow",
                "Fast Desnow (Snow Spots)",
                "OneRestore Auto",
            )
            for method in self.preprocess_methods
        )

    def apply_preprocessing(self, image):
        """Apply selected preprocessing operations to a BGR image in order."""
        if image is None or not self.preprocess_methods:
            return image

        processed = image.copy()
        for method in self.preprocess_methods:
            if QThread.currentThread().isInterruptionRequested():
                break
            if method == "DCP (Dehazing)":
                processed = self._dehaze(processed)
            elif method == "Restormer Derain":
                processed = RestormerDerainRunner.instance().restore(processed)
            elif method in ("SnowFormer Desnow", "Fast Desnow (Snow Spots)"):
                processed = SnowFormerDesnowRunner.instance().restore(processed)
            elif method == "LIME-Lite":
                processed = self._enhance_low_light(processed)
            elif method == "CLAHE (Local Contrast)":
                processed = self._clahe_local_contrast(processed)
            elif method == "OneRestore Auto":
                processed = OneRestoreRunner.instance().restore_auto_conservative(processed)
        return processed

    @staticmethod
    def _clahe_local_contrast(image):
        """Enhance local contrast on the luminance channel without changing hue strongly."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        enhanced = cv2.merge((lightness, channel_a, channel_b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _enhance_low_light(image):
        """Enhance dark images with an illumination map while limiting highlights."""
        source = image.astype(np.float32) / 255.0
        scene_light = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))) / 255.0
        if scene_light >= 0.50:
            return image.copy()
        illumination = np.max(source, axis=2)
        illumination = cv2.bilateralFilter(illumination, 9, 0.12, 15)
        illumination = np.maximum(illumination, 0.10)
        strength = float(np.clip((0.56 - scene_light) / 0.42, 0.25, 1.0))
        gain = 1.0 / np.power(illumination, 0.72 * strength)
        result = source * gain[:, :, None]
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        highlight_ratio = float(np.mean(gray >= 248))
        if highlight_ratio > 0.035:
            scale = float(np.clip(1.0 - (highlight_ratio - 0.035) * 2.0, 0.82, 1.0))
            result = np.clip(result.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return result

    def _predict_yolo(self, image):
        return self.model.predict(
            image,
            imgsz=self.image_size(),
            device="cpu",
            conf=self.Confidence(),
            iou=self.IOU(),
        )[0]

    @staticmethod
    def _detection_score(results):
        if min(results.boxes.shape) == 0:
            return (0, 0.0, 0.0)
        confidences = [float(conf) for conf in results.boxes.conf.tolist()]
        return (len(confidences), sum(confidences), max(confidences))

    def _select_detection_result(self, original_img, processed_img):
        processed_results = self._predict_yolo(processed_img)
        if not self._needs_original_comparison():
            return processed_img, processed_results, "preprocessed"
        if QThread.currentThread().isInterruptionRequested():
            return processed_img, processed_results, "preprocessed"

        original_results = self._predict_yolo(original_img)
        if self._detection_score(original_results) > self._detection_score(processed_results):
            return original_img.copy(), original_results, "original"
        return processed_img, processed_results, "restored"

    @staticmethod
    def _dehaze(image):
        """Apply dark-channel-prior dehazing with guided transmission refinement."""
        minimum = np.minimum(np.minimum(image[:, :, 0], image[:, :, 1]), image[:, :, 2])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark = cv2.erode(minimum, kernel)
        flat = dark.reshape(-1)
        sample_count = max(1, int(flat.size * 0.001))
        indices = np.argpartition(flat, -sample_count)[-sample_count:]
        candidates = image.reshape(-1, 3)[indices]
        atmosphere = candidates[np.argmax(np.mean(candidates, axis=1))].astype(np.float32)
        normalized = image.astype(np.float32) / np.maximum(atmosphere, 1.0)
        normalized_dark = cv2.erode(np.min(normalized, axis=2), kernel)
        transmission = 1.0 - 0.95 * normalized_dark
        guidance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        transmission = MainGui._guided_filter(guidance, transmission.astype(np.float32), radius=40, eps=1e-3)
        transmission = np.clip(transmission, 0.10, 1.0)[:, :, None]
        result = (image.astype(np.float32) - atmosphere) / transmission + atmosphere
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _guided_filter(guidance, source, radius=40, eps=1e-3):
        diameter = radius * 2 + 1
        mean_i = cv2.boxFilter(guidance, cv2.CV_32F, (diameter, diameter), normalize=True)
        mean_p = cv2.boxFilter(source, cv2.CV_32F, (diameter, diameter), normalize=True)
        corr_i = cv2.boxFilter(guidance * guidance, cv2.CV_32F, (diameter, diameter), normalize=True)
        corr_ip = cv2.boxFilter(guidance * source, cv2.CV_32F, (diameter, diameter), normalize=True)
        var_i = corr_i - mean_i * mean_i
        cov_ip = corr_ip - mean_i * mean_p
        a = cov_ip / (var_i + eps)
        b = mean_p - a * mean_i
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (diameter, diameter), normalize=True)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (diameter, diameter), normalize=True)
        return mean_a * guidance + mean_b

    def handle_buttons(self):
        """
        Connect UI controls to their event handlers.
        """
        # Model files.
        self.pushButton_data.clicked.connect(self.SelectData)
        self.pushButton_weights.clicked.connect(self.SelectWeights)

        # Inference sources.
        self.pushButton_img.clicked.connect(self.SelectImg)
        self.pushButton_dir.clicked.connect(self.SelectImgFile)
        self.pushButton_video.clicked.connect(self.SelectVideo)
        self.pushButton_cap.clicked.connect(self.SelectCap)

        # Start inference.
        self.pushButton_start.clicked.connect(self.Infer)

        # Stop inference.
        self.pushButton_end.clicked.connect(self.InferEnd)

        # Export results.
        self.pushButton_export.clicked.connect(self.write_csv)

        # # ---------------------------- classes ---------------------------------
        self.comboBox_2.activated.connect(self.onComboBoxActivatedDetection)

        # Handle result-table clicks.
        self.tableWidget_info.cellClicked.connect(self.cell_clicked)

    def update_table_width(self):
        """
        Configure the result table.
        """
        # Set the initial column widths.
        column_widths = [50, 220, 120, 200, 80, 80, 140]
        for column, width in enumerate(column_widths):
            self.tableWidget_info.setColumnWidth(column, width)

    def _apply_fonts(self):
        """
        Scale the main fonts consistently with the current window size.
        """
        width_scale = self.width() / self._base_window_size.width()
        height_scale = self.height() / self._base_window_size.height()
        scale = min(width_scale, height_scale)
        scale = min(max(scale, 0.9), 1.2)
        body_size = max(10, round(self._base_body_font_size * scale))
        title_size = max(20, round(self._base_title_font_size * scale))

        app_font = QtGui.QFont("Microsoft YaHei", body_size)
        self.setFont(app_font)
        self.centralwidget.setFont(app_font)
        self.tableWidget_info.setFont(app_font)
        self.comboBox.setFont(app_font)
        self.comboBox_2.setFont(app_font)
        self.doubleSpinBox.setFont(app_font)
        self.doubleSpinBox_2.setFont(app_font)
        for line_edit in self.findChildren(QLineEdit):
            line_edit.setFont(app_font)
        for name in (
            "label_img_path", "label_dir_path", "label_video_path", "label_13", "label_12",
            "label_cap_path", "lineEdit", "comboBox", "comboBox_2", "doubleSpinBox", "doubleSpinBox_2",
            "label_time", "label_score"
        ):
            control = self.findChild(QtWidgets.QWidget, name)
            if control is not None:
                control.setFont(app_font)
        for button in self.findChildren(QPushButton):
            button.setFont(app_font)
        for label in self.findChildren(QLabel):
            if label.objectName() != "label_2":
                label.setFont(app_font)

        # Replace font sizes embedded in the UI stylesheet with the scaled size.
        style_font_size = max(12, round(body_size * 1.33))
        for widget in self.findChildren(QtWidgets.QWidget):
            style_sheet = widget.styleSheet()
            if "font-size:" in style_sheet:
                widget.setStyleSheet(
                    re.sub(r"font-size:\s*\d+px", f"font-size: {style_font_size}px", style_sheet)
                )

        # Use Qt fonts for white fields so pixel styles do not override the reference size.
        white_field_names = (
            "label_img_path", "label_dir_path", "label_video_path", "label_cap_path",
            "label_13", "label_12", "lineEdit", "comboBox", "comboBox_2",
            "doubleSpinBox", "doubleSpinBox_2", "label_time", "label_score"
        )
        for name in white_field_names:
            field = self.findChild(QtWidgets.QWidget, name)
            if field is not None:
                field.setStyleSheet(re.sub(r"\s*font-size:\s*\d+px;?", "", field.styleSheet()))
                field.setFont(app_font)

        # Use the Select Video File control as the font reference for all white fields.
        reference_font = QtGui.QFont(self.label_video_path.font())
        if reference_font.pointSize() <= 0:
            reference_font.setPointSize(body_size)
        for name in white_field_names:
            field = self.findChild(QtWidgets.QWidget, name)
            if field is not None:
                field.setFont(reference_font)
                field.setStyleSheet(
                    re.sub(r"\s*font-size:\s*\d+px;?", "", field.styleSheet()).rstrip()
                    + f"\nfont-size: {reference_font.pointSize()}pt;"
                )

        title_font = self.label_2.font()
        title_font.setFamily("Microsoft YaHei")
        title_font.setBold(True)
        title_font.setPointSize(title_size)
        self.label_2.setFont(title_font)
        self._fit_label_font(self.label_2, minimum_size=16)

        # Keep result labels readable in narrow windows.
        for name in ("label_xmin", "label_19", "label_20", "label_21"):
            label = self.findChild(QLabel, name)
            if label is not None:
                self._fit_label_font(label, minimum_size=9)
        export_font = QtGui.QFont(app_font)
        export_font.setFamily("Arial")
        self.pushButton_export.setFont(export_font)
        self._fit_button_font(self.pushButton_export, minimum_size=9)

        # Match result-field labels to the Object label font size.
        object_label = self.findChild(QLabel, "label_7")
        if object_label is not None:
            result_font = QtGui.QFont(object_label.font())
            for name in ("label", "label_6", "label_7", "label_9"):
                label = self.findChild(QLabel, name)
                if label is not None:
                    label.setFont(result_font)
                    label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

    @staticmethod
    def _fit_label_font(label, minimum_size=9):
        """Shrink a single-line label until its text fits its width."""
        font = label.font()
        while font.pointSize() > minimum_size and QtGui.QFontMetrics(font).horizontalAdvance(label.text()) > label.width() - 4:
            font.setPointSize(font.pointSize() - 1)
        label.setFont(font)

    @staticmethod
    def _fit_button_font(button, minimum_size=8):
        """Shrink button text until it fits inside the button."""
        font = button.font()
        while font.pointSize() > minimum_size and QtGui.QFontMetrics(font).horizontalAdvance(button.text()) > button.width() - 2:
            font.setPointSize(font.pointSize() - 1)
        button.setFont(font)

    def update_layout_for_size(self):
        """
        Scale widget geometry and fonts according to the design dimensions.
        """
        if self.centralwidget.width() < 500 or self.centralwidget.height() < 300:
            return
        self._scale_widget_geometries()
        self._apply_fonts()
        self._align_result_rows()
        self.label_width, self.label_height = self.label_img.width(), self.label_img.height()

        table_total_width = max(self.tableWidget_info.width() - 2, 560)
        base_total = sum([50, 220, 120, 200, 80, 80, 140])
        column_widths = [max(40, round(table_total_width * ratio / base_total)) for ratio in [50, 220, 120, 200, 80, 80, 140]]
        for column, width in enumerate(column_widths):
            self.tableWidget_info.setColumnWidth(column, width)
        self._render_preview_image()

    def _align_result_rows(self):
        """Align each result icon with the vertical center of its label."""
        row_pairs = (
            ("label_11", "label"),
            ("label_15", "label_7"),
            ("label_14", "label_6"),
            ("label_17", "label_9"),
        )
        for icon_name, label_name in row_pairs:
            icon = self.findChild(QLabel, icon_name)
            label = self.findChild(QLabel, label_name)
            if icon is not None and label is not None:
                icon.move(icon.x(), label.y() + (label.height() - icon.height()) // 2)

    def _remember_widget_geometries(self, parent):
        """Remember UI-file geometry for later window scaling."""
        # Visit direct children only so nested widgets are not recorded twice.
        for widget in parent.children():
            if not isinstance(widget, QtWidgets.QWidget):
                continue
            # Exclude only temporary combo-box views. Keep the result table even
            # though it inherits QAbstractItemView so it scales with the main area.
            if isinstance(widget, QtWidgets.QAbstractItemView) and widget.objectName() != "tableWidget_info":
                continue
            base_parent_size = self._base_window_size if parent is self.centralwidget else parent.size()
            self._base_widget_geometries.append(
                (widget, parent, QtCore.QSize(base_parent_size), QtCore.QRect(widget.geometry()))
            )
            self._remember_widget_geometries(widget)

    def _scale_widget_geometries(self):
        """Scale direct children using the current size of each parent container."""
        for widget, parent, base_parent_size, base_rect in self._base_widget_geometries:
            if base_parent_size.width() <= 0 or base_parent_size.height() <= 0:
                continue
            try:
                sx = parent.width() / base_parent_size.width()
                sy = parent.height() / base_parent_size.height()
                widget.objectName()
            except RuntimeError:
                continue
            x = round(base_rect.x() * sx)
            y = round(base_rect.y() * sy)
            width = max(1, round(base_rect.width() * sx))
            height = max(1, round(base_rect.height() * sy))

            # Keep the right control panel compact and anchored to the window edge.
            if parent is self.centralwidget and base_rect.x() >= 915:
                panel_scale = min(max(sy, 0.8), 1.48)
                panel_width = round(331 * panel_scale)
                panel_left = self.centralwidget.width() - panel_width - round(10 * panel_scale)
                x = panel_left + round((base_rect.x() - 915) * panel_scale)
                width = max(1, round(base_rect.width() * panel_scale))
                vertical_scale = min(max(sy, 0.8), 1.1)
                desired_panel_top = round(220 * vertical_scale)
                max_panel_top = self.centralwidget.height() - round(761 * panel_scale) - round(10 * panel_scale)
                panel_top = max(round(100 * vertical_scale), min(desired_panel_top, max_panel_top))
                y = round(panel_top + (base_rect.y() - 120) * panel_scale)
                height = max(1, round(base_rect.height() * panel_scale))

            # Keep the title centered above the right-side panel.
            if parent is self.centralwidget and widget.objectName() == "label_2":
                x = 0
                width = self.centralwidget.width()

            if parent is self.centralwidget and widget.objectName() in (
                "label_img", "tableWidget_info"
            ):
                panel_scale = min(max(sy, 0.8), 1.48)
                panel_width = round(331 * panel_scale)
                panel_left = self.centralwidget.width() - panel_width - round(10 * panel_scale)
                gap = round(28 * panel_scale)
                x = max(0, round(base_rect.x() * sx))
                width = max(420, panel_left - x - gap)

            if parent is self.centralwidget and widget.objectName() in (
                "label", "label_6", "label_7", "label_9"
            ):
                panel_scale = min(max(sy, 0.8), 1.48)
                panel_width = round(331 * panel_scale)
                panel_left = self.centralwidget.width() - panel_width - round(10 * panel_scale)
                x = panel_left + round(55 * panel_scale)
                width = round(130 * panel_scale)

            if parent is self.centralwidget and widget.objectName() in (
                "label_time", "comboBox_2", "label_score"
            ):
                panel_scale = min(max(sy, 0.8), 1.48)
                panel_width = round(331 * panel_scale)
                panel_left = self.centralwidget.width() - panel_width - round(10 * panel_scale)
                x = panel_left + round(190 * panel_scale)
                width = round(126 * panel_scale)

            if parent is self.centralwidget and widget.objectName() in (
                "pushButton_start", "pushButton_end", "pushButton_export"
            ):
                panel_scale = min(max(sy, 0.8), 1.48)
                panel_width = round(331 * panel_scale)
                panel_left = self.centralwidget.width() - panel_width - round(10 * panel_scale)
                button_offsets = {
                    "pushButton_start": (0, 95),
                    "pushButton_end": (95, 95),
                    "pushButton_export": (190, 141),
                }
                offset, button_width = button_offsets[widget.objectName()]
                x = panel_left + round(offset * panel_scale)
                width = round(button_width * panel_scale)

            if parent is self.centralwidget and widget.objectName() in (
                "pushButton_img", "pushButton_dir", "pushButton_video", "pushButton_cap",
                "pushButton_data", "pushButton_weights", "label_4", "label_5",
                "label_11", "label_14", "label_15", "label_17", "label_22",
                "label_25"
            ):
                panel_scale = min(max(sy, 0.8), 1.48)
                panel_width = round(331 * panel_scale)
                panel_left = self.centralwidget.width() - panel_width - round(10 * panel_scale)
                icon_center = panel_left + round(35 * panel_scale)
                icon_width = round(45 * panel_scale)
                x = icon_center - icon_width // 2
                width = icon_width
                if isinstance(widget, QLabel):
                    widget.setAlignment(QtCore.Qt.AlignCenter)
                elif isinstance(widget, QPushButton):
                    widget.setIconSize(QtCore.QSize(round(32 * panel_scale), round(32 * panel_scale)))

            try:
                widget.setGeometry(x, y, width, height)
            except RuntimeError:
                # Qt may delete transient widgets such as an old combo-box view.
                continue

    def paintEvent(self, event):
        """Draw one background image instead of tiling it with QBrush."""
        if not self._background_pixmap.isNull():
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtGui.QColor(7, 27, 38))
            background_scale = 0.82
            target_size = QtCore.QSize(
                max(1, round(self.width() * background_scale)),
                max(1, round(self.height() * background_scale)),
            )
            scaled = self._background_pixmap.scaled(
                target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()
        super().paintEvent(event)

    def Confidence(self):
        """
        Return the selected confidence threshold.
        """
        confidence = '%.2f' % self.doubleSpinBox.value()
        return eval(confidence)

    def IOU(self):
        """
        Return the selected IoU threshold.
        """
        iou = '%.2f' % self.doubleSpinBox_2.value()
        return eval(iou)

    def image_size(self):
        """Read the inference size and use a safe default for invalid input."""
        try:
            value = int(float(self.lineEdit.text().strip()))
        except (TypeError, ValueError):
            value = 640
        return max(32, value)

    def SelectData(self):
        """
        Select the model dataset configuration.
        """
        self.data_file_name, _ = QFileDialog.getOpenFileName(self, "Select YAML File", "", "YAML files (*.yaml)")
        if self.data_file_name:
            self.label_13.setText(os.path.split(self.data_file_name)[-1])
            self._load_yaml_classes(self.data_file_name)

    def _load_yaml_classes(self, file_name):
        """Load class names from a dataset configuration file."""
        with open(file_name, 'r', encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        names = data.get("names", [])
        if isinstance(names, dict):
            names = [names[key] for key in sorted(names, key=lambda value: int(value))]
        self.yaml_classes = [str(name) for name in names]
        self.color.update(
            {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
             for name in self.yaml_classes})
        self._update_object_selector(self.yaml_classes)

    def _update_object_selector(self, classes=None):
        """Load YAML classes into Object without mixing preprocessing options."""
        classes = self.yaml_classes if classes is None else classes
        unique_classes = list(dict.fromkeys(str(name) for name in classes))
        self.comboBox_2.blockSignals(True)
        self.comboBox_2.clear()
        self.comboBox_2.addItem("None")
        self.comboBox_2.addItems(unique_classes)
        self.comboBox_2.setCurrentIndex(0)
        self.comboBox_2.blockSignals(False)

    def SelectWeights(self):
        """
        Select model weights.
        """
        self.weights_file_name, _ = QFileDialog.getOpenFileName(self, "Select Weights File", "",
                                                                "All supported files (*.pt *.onnx *.torchscript *.engine "
                                                                "*.mlmodel *.pb *.tflite *openvino_model  "
                                                                "*saved_model *paddle_model)")
        if self.weights_file_name:
            self.label_12.setText(os.path.split(self.weights_file_name)[-1])
            # Preserve the previous value for change detection.
            old_parameter = self.save_parameter.copy()
            # Read the newly selected value.
            self.save_parameter["weights"] = self.weights_file_name
            # Invalidate the loaded model when parameters change.
            if old_parameter != self.save_parameter:
                # Store the new value.
                self.pre_parameter = old_parameter

    def SelectImg(self):
        """
        Select an image file.
        """
        self.img_path, filetype = QFileDialog.getOpenFileName(self, "Select Image File", "",
                                                              "Image files (*.jpg *.bmp *.dng" " *.jpeg *.jpg *.mpo"
                                                              " *.png *.tif *.tiff *.webp *.pfm)")
        if self.img_path == "":
            self.start_type = None
            return

        # Display the selected path.
        self.start_type = 'img'
        self.img_name = os.path.split(self.img_path)[-1]

        self.label_img_path.setText(self.img_name)
        self.label_dir_path.setText("Select Image Folder")
        self.label_video_path.setText("Select Video File")
        self.label_cap_path.clear()
        self.label_cap_path.setPlaceholderText("Select Camera Source")

        self.org_img_save_path = os.path.join(self.result_org_img_path, self.img_name)

        # Display the source image.
        self._clear_preview_image()
        self.ShowSource(cv2.imread(self.img_path))
        shutil.copy(self.img_path, self.org_img_save_path)

    def SelectImgFile(self):
        """
        Select an image directory.
        """
        self.img_path_dir = QFileDialog.getExistingDirectory(None, "Select Image Folder")
        if self.img_path_dir == '':
            self.start_type = None
            return

        self.start_type = 'dir'

        self.label_img_path.setText("Select Image File")
        self.label_dir_path.setText(os.path.split(self.img_path_dir)[-1])
        self.label_video_path.setText("Select Video File")
        self.label_cap_path.clear()
        self.label_cap_path.setPlaceholderText("Select Camera Source")

        self.image_files = [os.path.join(self.img_path_dir, file) for file in os.listdir(self.img_path_dir) if
                            file.lower().endswith(
                                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]
        if self.image_files:
            self.img_path = self.image_files[0]
            self.img_name = os.path.split(self.img_path)[-1]

            self._clear_preview_image()
            self.ShowSource(cv2.imread(self.img_path))

    def SelectVideo(self):
        """
        Select a video file.
        """
        # Open the file picker.
        self.video_path, filetype = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                                "Video files (*.asf *.avi *.gif *.m4v *.mkv "
                                                                "*.mov *.mp4 *.mpeg *.mpg *.ts *.wmv)")
        if self.video_path == "":  # No file was selected.
            self.start_type = None
            return
        self.start_type = 'video'
        self.img_name = os.path.split(self.video_path)[-1]

        self.label_img_path.setText("Select Image File")
        self.label_dir_path.setText("Select Image Folder")
        self.label_video_path.setText(self.img_name)
        self.label_cap_path.clear()
        self.label_cap_path.setPlaceholderText("Select Camera Source")

        shutil.copy(self.video_path, os.path.join(self.result_org_img_path, self.img_name))

    def SelectCap(self):
        """
        Select a camera source. Blank uses device 0; device IDs and URLs are supported.
        """
        self.label_img_path.setText("Select Image File")
        self.label_dir_path.setText("Select Image Folder")
        self.label_video_path.setText("Select Video File")

        source_text = self.label_cap_path.text().strip()
        if not source_text:
            source_text = "0"
            self.label_cap_path.setText(source_text)
        try:
            self.video_path = int(source_text)
        except ValueError:
            self.video_path = source_text
        self.start_type = 'cap'
        self.img_name = 'camera.mp4'

    def load_model(self):
        """
        Load the model when needed and retain any error for the GUI thread.
        """
        self.model_error = None
        if self.save_parameter:
            # Reload the model only when its parameters have changed.
            if self.model is None or self.pre_parameter.items() != self.save_parameter.items():
                self.pre_parameter = self.save_parameter

                # Verify that all required model parameters are present.
                result = self.check_none_variables(weights=self.weights_file_name)
                if not result:
                    try:
                        self.model = YOLO(model=self.weights_file_name)
                        return True
                    except ModuleNotFoundError as error:
                        self.model = None
                        missing_module = error.name or str(error)
                        self.model_error = (
                            "Weight Load Failed",
                            "This weight file depends on a custom training module:\n"
                            f"{missing_module}\n\n"
                            "Use the matching Ultralytics environment, or export the model to ONNX.",
                        )
                        return False
                    except Exception as error:
                        self.model = None
                        self.model_error = (
                            "Weight Load Failed", f"Unable to load weight file:\n{error}"
                        )
                        return False
                else:
                    self.model_error = (
                        "Parameter Selection", f"Missing parameters: {','.join(result)}"
                    )
                    return False
            else:
                return True
        else:
            self.model_error = ("Parameter Selection", "Model parameters have not been selected.")
            return False

    def Infer(self):
        """
        Start inference for the selected input type.
        """
        # Discard a completed worker left after Stop so it cannot block a restart.
        if self.worker_thread is not None and not self.worker_thread.isRunning():
            self.worker_thread.deleteLater()
            self.worker_thread = None

        if self.start_type not in {'img', 'dir', 'video', 'cap'}:
            QtWidgets.QMessageBox.warning(self, "Input Selection", "Select an input source first.")
            return

        self.pushButton_start.setEnabled(False)
        self.pushButton_end.setEnabled(True)
        self.pushButton_end.setText("Stop")
        self.worker_thread = WorkerThread(self)
        self.worker_thread.error_requested.connect(self._show_worker_error)
        self.worker_thread.finished.connect(self._on_worker_finished)
        self.worker_thread.start()

    @QtCore.pyqtSlot(str, str)
    def _show_worker_error(self, title, message):
        """Display worker errors safely on the GUI thread."""
        QtWidgets.QMessageBox.critical(self, title, message)

    def _on_worker_finished(self):
        """Restore controls after normal completion or an interrupted run."""
        worker = self.sender()
        if self.worker_thread is worker:
            self.worker_thread = None
        if worker is not None:
            worker.deleteLater()
        self.pushButton_start.setEnabled(True)
        self.pushButton_end.setEnabled(False)
        self.pushButton_end.setText("Stop")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def InferEnd(self):
        """
        Stop video, camera, or directory inference.
        """
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.pushButton_end.setText("Stopping...")
            self.worker_thread.stop()

        # One preprocessing or prediction call may still be completing.
        # Keep Start disabled until the worker confirms a clean exit.
        self.pushButton_end.setEnabled(False)
        self.pushButton_start.setEnabled(self.worker_thread is None)

        self._update_object_selector(self.yaml_classes or self.comboBox_name)

    def predict_img(self, img):
        """
        Detect objects after applying the selected preprocessing pipeline.
        """
        processed_img = self.apply_preprocessing(img)
        if QThread.currentThread().isInterruptionRequested():
            return False
        start_time = time.time()
        self.result_img_name = os.path.join(self.result_img_path, self.img_name)
        selected_img, results, _ = self._select_detection_result(img, processed_img)
        if QThread.currentThread().isInterruptionRequested():
            return False
        self.end_time = str(round(time.time() - start_time, 4)) + 's'
        self._detection_base_image = selected_img.copy()

        if min(results.boxes.shape) != 0:
            self.all_result = []
            self.comboBox_name = []
            for boxs, cls, conf in zip(results.boxes.xyxy.tolist(), results.boxes.cls.tolist(),
                                       results.boxes.conf.tolist()):
                self.all_result.extend([[round(i, 2) for i in boxs] + [results.names[int(cls)]] + [round(conf, 4)]])
                if results.names[int(cls)] not in self.comboBox_name:
                    self.comboBox_name.append(results.names[int(cls)])

            if self.all_result:
                # Save the annotated result image.
                im_array = results.orig_img
                self.draw = self.draw_info(im_array, self.all_result)
                cv2.imwrite(self.result_img_name, self.draw)

                self.results_index = {f'Target {index + 1}': result for index, result in enumerate(self.all_result)}

                self.input_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Add detection information to the result table.
                self.show_table()
                self.number += 1

                self._update_object_selector(self.yaml_classes or self.comboBox_name)
        else:
            self._update_object_selector(self.yaml_classes or self.comboBox_name)
            self.draw = selected_img

        self.ShowSource(self.draw)
        self.clear_info()
        self.label_time.setText(self.end_time)
        return True

    def ShowSource(self, img):
        """
        Display an image in the preview area.
        """
        if img is None:
            self.preview_requested.emit(None)
            return
        self.preview_requested.emit(img.copy())

    @QtCore.pyqtSlot(object)
    def _set_preview_image(self, img):
        """Update preview data on the GUI thread."""
        if img is None:
            self._clear_preview_image()
            return
        self._preview_image = img
        self._render_preview_image()

    def _clear_preview_image(self):
        """Clear the preview image and its cached source."""
        self._preview_image = None
        self.label_img.clear()

    def _render_preview_image(self):
        """Render the current preview image centered inside the fixed preview area."""
        if self._preview_image is None or self.label_img.width() <= 0 or self.label_img.height() <= 0:
            return

        source_height, source_width = self._preview_image.shape[:2]
        scale = min(
            self.label_img.width() / source_width,
            self.label_img.height() / source_height,
        )
        target_width = max(1, round(source_width * scale))
        target_height = max(1, round(source_height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        preview = cv2.resize(
            self._preview_image,
            (target_width, target_height),
            interpolation=interpolation,
        )
        rgb_image = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        qt_image = QtGui.QImage(
            rgb_image.data,
            rgb_image.shape[1],
            rgb_image.shape[0],
            rgb_image.strides[0],
            QtGui.QImage.Format_RGB888,
        ).copy()
        self.label_img.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def show_table(self):
        """
        Add the current inference result to the table.
        """
        # Append one table row.
        self.RowLength = self.RowLength + 1
        self.tableWidget_info.setRowCount(self.RowLength)
        for column, content in enumerate(
                [self.number, self.org_img_save_path, self.input_time, self.all_result, len(self.all_result),
                 self.end_time,
                 self.result_img_name]):
            row = self.RowLength - 1
            item = QtWidgets.QTableWidgetItem(str(content))
            # Center the cell text.
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            # Set the cell text color.
            item.setForeground(QColor.fromRgb(197, 223, 250))
            # Create the cell font.
            font = QtGui.QFont()
            # Use a 10-point font.
            font.setPointSize(10)
            # Apply the font to the table item.
            item.setFont(font)
            self.tableWidget_info.setItem(row, column, item)
        # Scroll to the latest result.
        self.tableWidget_info.scrollToBottom()

    def cell_clicked(self, row):
        """
        Restore a detection result selected from the table.
        """
        if self.tableWidget_info.item(row, 1) is None:
            return
        # Source image path.
        self.org_img_save_path = self.tableWidget_info.item(row, 1).text()
        # Detection results.
        self.all_result = eval(self.tableWidget_info.item(row, 3).text())
        # Inference duration.
        self.infer_time = self.tableWidget_info.item(row, 5).text()
        # Annotated image path.
        self.result_img_name = self.tableWidget_info.item(row, 6).text()

        self.results_index = {f'Target {index + 1}': result for index, result in enumerate(self.all_result)}

        self._clear_preview_image()
        draw_img = cv2.imread(self.result_img_name)
        self.ShowSource(draw_img)

        self._update_object_selector(sorted({result[4] for result in self.all_result}))

        self.label_time.setText(self.infer_time)
        self.clear_info()

    def show_info(self, result):
        """
        Display bounding-box coordinates and confidence.
        """
        self.label_score.setText(str(result[5]))
        self.label_xmin_v.setText(str(result[0]))
        self.label_ymin_v.setText(str(result[1]))
        self.label_xmax_v.setText(str(result[2]))
        self.label_ymax_v.setText(str(result[3]))
        # Refresh the UI.
        self.update()

    def clear_info(self):
        """
        Clear displayed bounding-box coordinates and confidence.
        """
        self.label_score.clear()
        self.label_xmin_v.clear()
        self.label_ymin_v.clear()
        self.label_xmax_v.clear()
        self.label_ymax_v.clear()
        # Refresh the UI.
        self.update()

    @staticmethod
    def check_none_variables(**args):
        """
        Return the names of parameters whose values are empty.
        """
        none_variables = []
        for var_name, var_value in args.items():
            if var_value == '':
                none_variables.append(var_name)
        return none_variables

    def onComboBoxActivated(self):
        """
        Retain compatibility with the legacy combo-box callback.
        """
        return

    def onComboBoxActivatedDetection(self):
        """
        Filter targets by YAML class; None displays every detection.
        """
        self.selected_text = self.comboBox_2.currentText()
        if self.selected_text != 'None':
            if self._detection_base_image is None:
                draw_img = cv2.imread(self.org_img_save_path)
            else:
                draw_img = self._detection_base_image.copy()
            selected_results = [result for result in self.all_result if result[4] == self.selected_text]
            draw_img = self.draw_info(draw_img, selected_results)
            self._clear_preview_image()
            self.ShowSource(draw_img)
            if selected_results:
                self.show_info(selected_results[0])
            else:
                self.clear_info()
        else:
            draw_img = cv2.imread(self.result_img_name)
            self.ShowSource(draw_img)
            self.clear_info()

    def draw_info(self, draw_img, results):
        """
        Draw bounding boxes and labels on an image.
        """
        lw = max(round(sum(draw_img.shape) / 2 * 0.003), 2)  # line width
        tf = max(lw - 1, 1)  # font thickness
        sf = lw / 3  # font scale
        for result in results:
            box = result[:4]
            cls_name = result[4]
            conf = result[5]

            color = self.color[cls_name]
            label = f'{cls_name} {conf}'

            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            # Draw the bounding box.
            cv2.rectangle(draw_img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
            # text width, height
            w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
            # label fits outside box
            outside = box[1] - h - 3 >= 0
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            # Draw the label background.
            cv2.rectangle(draw_img, p1, p2, color, -1, cv2.LINE_AA)
            # Draw the label text.
            cv2.putText(draw_img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        sf,
                        self.color["font"],
                        thickness=2,
                        lineType=cv2.LINE_AA)

        return draw_img

    def write_csv(self):
        """
        Export inference results to CSV.
        """
        result_csv = os.path.join(self.result_time_path, 'result.csv')

        num_rows = self.tableWidget_info.rowCount()
        num_cols = self.tableWidget_info.columnCount()
        datas = []
        for row in range(num_rows):
            row_data = []
            for col in range(num_cols):
                item = self.tableWidget_info.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append('')
            datas.append(row_data)

        with open(result_csv, "w", newline="") as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['Index', 'Image Name', 'Recording Time', 'Detection Results', 'Target Count', 'Inference Time', 'Save Path'])
            for data in datas:
                writer.writerow(data)

        QMessageBox.information(None, "Export Complete", f"Data saved successfully.\nSave path: {result_csv}", QMessageBox.Yes)

    def closeEvent(self, event):
        """Close the application after user confirmation and worker shutdown."""
        if self._closing_after_worker:
            event.accept()
            return

        reply = QMessageBox.question(self, 'Exit', "Do you want to close the application?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.worker_thread is not None and self.worker_thread.isRunning():
                self._closing_after_worker = True
                self.worker_thread.finished.connect(self.close)
                self.worker_thread.stop()
                event.ignore()
                return
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event):
        """
        Update layout and fonts when the window is resized.
        """
        super().resizeEvent(event)
        self.update_layout_for_size()


class WorkerThread(QThread):
    """
    Run video, camera, and directory inference outside the UI thread.
    """

    error_requested = QtCore.pyqtSignal(str, str)

    def __init__(self, main_window):
        super().__init__()
        self.running = True
        self.main_window = main_window

    def run(self):
        # Loading PyTorch weights can take several seconds in a packaged app.
        # Keep it here so the GUI event loop can always receive Stop clicks.
        if not self.main_window.load_model():
            title, message = self.main_window.model_error or (
                "Model Load Failed", "Unable to load the selected model."
            )
            self.error_requested.emit(title, message)
            return
        if not self.running or self.isInterruptionRequested():
            return

        if self.main_window.start_type == 'img':
            image = cv2.imread(self.main_window.img_path)
            if image is not None and self.running and not self.isInterruptionRequested():
                self.main_window.predict_img(image)

        elif self.main_window.start_type == 'video' or self.main_window.start_type == "cap":
            if self.main_window.start_type == 'cap' and isinstance(self.main_window.video_path, int):
                capture = cv2.VideoCapture(self.main_window.video_path, cv2.CAP_DSHOW)
                if not capture.isOpened():
                    capture.release()
                    capture = cv2.VideoCapture(self.main_window.video_path)
            else:
                capture = cv2.VideoCapture(self.main_window.video_path)
            self.main_window.cap = capture

            if not self.main_window.cap.isOpened():
                self.main_window.cap.release()
                self.main_window.cap = None
                self.error_requested.emit(
                    "Camera/Video Open Failed",
                    f"Unable to open source: {self.main_window.video_path}",
                )
                return
            video_name = self.main_window.img_name if '.mp4' in self.main_window.img_name else \
                self.main_window.img_name.split(".")[0] + '.mp4'
            frame_num = 0
            save_path = os.path.join(self.main_window.result_img_path, video_name)
            fps = 30.0 if 'camera' in video_name else self.main_window.cap.get(cv2.CAP_PROP_FPS)
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         (int(self.main_window.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                          int(self.main_window.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            while self.running and not self.isInterruptionRequested():
                self.main_window.img_name = video_name.split(".")[0] + '_' + str(frame_num) + '.jpg'

                ret, frame = self.main_window.cap.read()
                if ret:
                    if not self.running or self.isInterruptionRequested():
                        break
                    self.main_window.org_img_save_path = os.path.join(self.main_window.result_org_img_path,
                                                                      self.main_window.img_name)
                    cv2.imwrite(self.main_window.org_img_save_path, frame)

                    if not self.main_window.predict_img(frame):
                        break

                    frame_num += 1

                    vid_writer.write(self.main_window.draw)
                else:
                    break
            self.main_window.cap.release()
            vid_writer.release()

        elif self.main_window.start_type == 'dir':
            for img_path in self.main_window.image_files:
                if not self.running or self.isInterruptionRequested():
                    break
                img = cv2.imread(img_path)
                self.main_window.img_name = os.path.split(img_path)[-1]
                self.main_window.org_img_save_path = os.path.join(self.main_window.result_org_img_path,
                                                                  self.main_window.img_name)
                shutil.copy(img_path, self.main_window.org_img_save_path)
                if not self.main_window.predict_img(img):
                    break

    def stop(self):
        """Request cooperative cancellation without blocking the UI thread."""
        self.running = False
        self.requestInterruption()
        capture = self.main_window.cap
        if capture is not None:
            capture.release()


def main():
    app = QApplication(sys.argv)
    window = MainGui()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
