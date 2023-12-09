import os
import shutil
from md2pdf.core import md2pdf
from typing import List
from pydicom import dcmread
from mdutils.mdutils import MdUtils
from enum import Enum

"""
Release Notes
Mode: Alpha
Version: v0.1.0
Date: March 4rd, 2022
Authors: Raiven Team
"""


class Raiven:
    """Raiven is the main class that represents the API to raiven platform.
    It controls all input-output tasks to save the DICOM / text files in the backend filesystem.
    It abstracts away the details of the implementation from the users by exposing an easy to use interface."""

    class LogStatus(Enum):
        Error = "ERROR"
        Info = "INFO"
        Warning = "WARNING"

    def __init__(self):
        self.input_dir = os.environ["RAIVEN_INPUT_DIR"]
        self.output_dir = os.environ["RAIVEN_OUTPUT_DIR"]
        self.job_id = os.environ["JOB_ID"]
        self.run_id = os.environ["RUN_ID"]
        self.serial_file = "conditional.txt"
        self.condition_tree = {}
        self.container_name = os.environ["CONTAINER_NAME"]

    def dicom_files(self) -> List[str]:
        """
        Returns a list of paths(str) to the DICOM files at this node
        """
        path_list = []

        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".dcm"):
                    path_list.append(os.path.join(root, file))

        return path_list

    def get_output_dir(self):
        return self.output_dir

    def get_input_dir(self):
        return self.input_dir

    def add_condition(self, file: str, condition: str) -> str:
        """
        Add a DICOM file to a specific condition category
        Returns the file path to the specific condition folder to save to
        """

        if condition not in self.condition_tree:
            self.condition_tree[condition] = []
        self.condition_tree[condition].append(file)

        file_array = file.split("/")
        file_name = file_array[-1]

        condition_dir = os.path.join(self.output_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)
        pathing = os.path.join(condition_dir, file_name)

        return pathing

    def finalize_conditions(self) -> bool:
        """
        Finalize the condition dictionary by parsing and separating the DICOM
        files into categorized folders
        """
        for key, value in self.condition_tree.items():
            path = os.path.join(self.output_dir, key)
            os.mkdir(path)
            move_files(self.input_dir, path, value)

        return True

    def condition_add(self, condition: str):
        """
        Manually add a condition to perform by writing to the text file
        """
        with open(self.serial_path(), "w") as f:
            f.write(condition)
            f.write("\n")

        print("Condition, {} added".format(condition))

    def add_dicom(dicom):
        """
        Add a DICOM file to the condition
        """
        print("Dicom, {} added".format(dicom))

    def serial_path(self) -> str:
        """
        Return the path to the text file which controls the conditions
        """
        return os.path.join(self.output_dir, self.serial_file)

    def savemd(self, mdFile, src):
        """
        Save markdown file to a specifiction output folder
        """
        mdFile.create_md_file()
        destination = self.output_dir + "/" + str(self.job_id) + "-" + src

        print("[RAIVEN LIB]: Copying from {src} to {destination}", src, destination)

        shutil.copyfile(src, destination)
        file_pdf_name = src.replace(".md", ".pdf")
        dst_path = os.path.join(self.output_dir, self.job_id + "-" + file_pdf_name)

        md2pdf(
            dst_path,
            md_file_path=destination,
            css_file_path=None,
            base_url=str(dst_path),
        )

    def log(self, content, log_status: LogStatus):
        """
        logs messages to the corresponding text file for the container
        """
        print("[RAIVEN LIB]: logging into {destination}", self.output_dir)
        with open(
            self.output_dir
            + "/"
            + str(self.container_name)
            + "-"
            + str(self.job_id)
            + ".log",
            "w",
        ) as f:
            f.write(log_status.value + ": " + content)
            f.write("\n")
