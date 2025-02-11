"""Utility of downloading"""
import bz2
import gzip
import hashlib
import logging
import os
import ssl
import tarfile
import urllib
import urllib.error
import urllib.request
import zipfile
from copy import deepcopy
from typing import Optional

from tqdm import tqdm

from .path import detect_file_type

_logger = logging.getLogger(__name__)
# The default root directory where we save downloaded files.
# Use Get/Set to R/W this variable.
_DEFAULT_DOWNLOAD_ROOT = os.path.join(os.path.expanduser("~"), ".mindspore")


def get_default_download_root():
    return deepcopy(_DEFAULT_DOWNLOAD_ROOT)


def set_default_download_root(path):
    global _DEFAULT_DOWNLOAD_ROOT
    _DEFAULT_DOWNLOAD_ROOT = path


class DownLoad:
    """Base utility class for downloading."""

    USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/92.0.4515.131 Safari/537.36"
    )

    @staticmethod
    def calculate_md5(file_path: str, chunk_size: int = 1024 * 1024) -> str:
        """Calculate md5 value."""
        md5 = hashlib.md5()
        with open(file_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def check_md5(self, file_path: str, md5: Optional[str] = None) -> bool:
        """Check md5 value."""
        return md5 == self.calculate_md5(file_path)

    @staticmethod
    def extract_tar(from_path: str, to_path: Optional[str] = None, compression: Optional[str] = None) -> None:
        """Extract tar format file."""

        with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
            tar.extractall(to_path)

    @staticmethod
    def extract_zip(from_path: str, to_path: Optional[str] = None, compression: Optional[str] = None) -> None:
        """Extract zip format file."""

        compression_mode = zipfile.ZIP_BZIP2 if compression else zipfile.ZIP_STORED
        with zipfile.ZipFile(from_path, "r", compression=compression_mode) as zip_file:
            zip_file.extractall(to_path)

    def extract_archive(self, from_path: str, to_path: str = None) -> str:
        """Extract and  archive from path to path."""
        archive_extractors = {
            ".tar": self.extract_tar,
            ".zip": self.extract_zip,
        }
        compress_file_open = {
            ".bz2": bz2.open,
            ".gz": gzip.open,
        }

        if not to_path:
            to_path = os.path.dirname(from_path)

        suffix, archive_type, compression = detect_file_type(from_path)  # pylint: disable=unused-variable

        if not archive_type:
            to_path = from_path.replace(suffix, "")
            compress = compress_file_open[compression]
            with compress(from_path, "rb") as rf, open(to_path, "wb") as wf:
                wf.write(rf.read())
            return to_path

        extractor = archive_extractors[archive_type]
        extractor(from_path, to_path, compression)

        return to_path

    def download_file(self, url: str, file_path: str, chunk_size: int = 1024):
        """Download a file."""

        # no check certificate
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # Define request headers.
        headers = {"User-Agent": self.USER_AGENT}

        _logger.info(f"Downloading from {url} to {file_path} ...")
        with open(file_path, "wb") as f:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, context=ctx) as response:
                with tqdm(total=response.length, unit="B") as pbar:
                    for chunk in iter(lambda: response.read(chunk_size), b""):
                        if not chunk:
                            break
                        pbar.update(chunk_size)
                        f.write(chunk)

    def download_url(
        self,
        url: str,
        path: Optional[str] = None,
        filename: Optional[str] = None,
        md5: Optional[str] = None,
    ) -> None:
        """Download a file from a url and place it in root."""
        if path is None:
            path = get_default_download_root()
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)

        if not filename:
            filename = os.path.basename(url)

        file_path = os.path.join(path, filename)

        # Check if the file is exists.
        if os.path.isfile(file_path):
            if not md5 or self.check_md5(file_path, md5):
                return file_path

        # Download the file.
        try:
            self.download_file(url, file_path)
        except (urllib.error.URLError, IOError) as e:
            if url.startswith("https"):
                url = url.replace("https", "http")
                try:
                    self.download_file(url, file_path)
                except (urllib.error.URLError, IOError):
                    # pylint: disable=protected-access
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.download_file(url, file_path)
                    ssl._create_default_https_context = ssl.create_default_context
            else:
                raise e

        return file_path

    def download_and_extract_archive(
        self,
        url: str,
        download_path: Optional[str] = None,
        extract_path: Optional[str] = None,
        filename: Optional[str] = None,
        md5: Optional[str] = None,
        remove_finished: bool = False,
    ) -> None:
        """Download and extract archive."""
        if download_path is None:
            download_path = get_default_download_root()
        download_path = os.path.expanduser(download_path)

        if not filename:
            filename = os.path.basename(url)

        self.download_url(url, download_path, filename, md5)

        archive = os.path.join(download_path, filename)
        self.extract_archive(archive, extract_path)

        if remove_finished:
            os.remove(archive)


def download_checkpoint(ckpt_url, save_dir="models/"):
    file_path = DownLoad().download_url(ckpt_url, path=save_dir)
    return file_path
