"""Public frozen-reader API.

The implementation lives in :mod:`visig_cr.reader_core`; keeping this package
boundary makes the public ``visig_cr.reader`` import stable.
"""

from ..reader_core import (
    ReaderOutputValidationError,
    VLMReader,
    evidence_package_sha256,
    image_to_data_url,
    llava_med_compact_response_format,
    llava_med_no_rag_response_format,
    reader_response_format,
)

__all__ = [
    "ReaderOutputValidationError",
    "VLMReader",
    "evidence_package_sha256",
    "image_to_data_url",
    "llava_med_compact_response_format",
    "llava_med_no_rag_response_format",
    "reader_response_format",
]
