"""
OCR Adapter Module
==================

Provides a unified interface for different OCR engines (EasyOCR and PaddleOCR).
This adapter pattern allows seamless switching between OCR backends without
changing the main processing logic.

Supported Engines:
- PaddleOCR: Faster, more accurate for jersey number detection
- EasyOCR: Maintained for backward compatibility

Usage:
    from utils.ocr_adapter import create_ocr_engine

    # Create PaddleOCR engine (default)
    ocr = create_ocr_engine(engine_type='paddleocr', use_gpu=True)

    # Or use EasyOCR
    ocr = create_ocr_engine(engine_type='easyocr', use_gpu=True)

    # Process image - both engines have the same interface
    results = ocr.readtext(image)  # Returns [(bbox, text, confidence), ...]
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def readtext(self, image: np.ndarray) -> List[Tuple[List, str, float]]:
        """
        Detect and recognize text in an image.

        Args:
            image: Input image as numpy array (BGR or grayscale)

        Returns:
            List of detections, where each detection is a tuple:
                (bbox_coords, text, confidence)

            bbox_coords: List of 4 [x, y] coordinates for the text region
                        [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text: Detected text string
            confidence: Confidence score (float, 0.0-1.0)
        """
        pass


class PaddleOCRAdapter(OCREngine):
    """
    Adapter for PaddleOCR to provide EasyOCR-compatible interface.

    PaddleOCR returns results in format: [[[bbox], (text, confidence)], ...]
    This adapter converts to EasyOCR format: [(bbox, text, confidence), ...]
    """

    def __init__(
        self,
        lang: str = 'en',
        use_gpu: bool = True,
        use_angle_cls: bool = True,
        det: bool = True,
        rec: bool = True,
        cls: bool = True,
        show_log: bool = False
    ):
        """
        Initialize PaddleOCR engine (supports v2.7 and v3.3+).

        Args:
            lang: Language code (default: 'en' for English)
            use_gpu: Whether to use GPU acceleration (default: True)
            use_angle_cls: Enable angle classification for rotated text (default: True)
                          Supported in v2.7, deprecated in v3.3.2
            det: Enable text detection (default: True, required for operation)
            rec: Enable text recognition (default: True, required for operation)
            cls: Enable text direction classification (default: True)
            show_log: Show PaddleOCR debug logs (default: False)
        """
        try:
            from paddleocr import PaddleOCR
            import paddleocr
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Install it with: pip install paddleocr"
            )

        # Detect PaddleOCR version
        version_str = paddleocr.__version__
        version_parts = version_str.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        # Initialize based on version
        if major >= 3:
            # v3.0+ API: uses 'device' parameter
            device = "gpu:0" if use_gpu else "cpu"
            try:
                self.ocr = PaddleOCR(
                    lang=lang,
                    device=device,
                    use_textline_orientation=use_angle_cls
                )
            except TypeError:
                # Fallback if use_textline_orientation not supported
                self.ocr = PaddleOCR(lang=lang, device=device)
        else:
            # v2.7 API: uses 'use_gpu' parameter
            try:
                self.ocr = PaddleOCR(
                    lang=lang,
                    use_gpu=use_gpu,
                    use_angle_cls=use_angle_cls
                )
            except TypeError:
                # Fallback for older versions
                self.ocr = PaddleOCR(lang=lang, use_gpu=use_gpu)

        self.engine_name = f'PaddleOCR v{version_str}'
        # Store det, rec, cls for use in readtext() method
        self.det = det
        self.rec = rec
        self.cls = cls

    def readtext(self, image: np.ndarray) -> List[Tuple[List, str, float]]:
        """
        Detect and recognize text using PaddleOCR.

        Args:
            image: Input image as numpy array

        Returns:
            List of (bbox, text, confidence) tuples in EasyOCR format
        """
        try:
            # PaddleOCR returns: [[[bbox], (text, confidence)], ...]
            result = self.ocr.ocr(image, det=self.det, rec=self.rec, cls=self.cls)

            # Handle empty or None results
            if not result or result[0] is None:
                return []

            # Convert PaddleOCR format to EasyOCR format
            converted = []
            for detection in result[0]:
                try:
                    # Extract components
                    bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text, confidence = detection[1]  # (text, confidence)

                    # Validate bbox structure
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue

                    # Ensure all bbox points are valid
                    bbox_valid = all(
                        isinstance(p, (list, tuple)) and len(p) == 2
                        for p in bbox
                    )
                    if not bbox_valid:
                        continue

                    # Convert confidence to float
                    confidence = float(confidence)

                    # Append in EasyOCR format
                    converted.append((bbox, text, confidence))

                except (IndexError, ValueError, TypeError):
                    # Skip malformed detections
                    continue

            return converted

        except Exception as e:
            # Log error but return empty list to continue processing
            print(f"[WARNING] PaddleOCR error: {e}")
            return []


class EasyOCRAdapter(OCREngine):
    """
    Adapter for EasyOCR (thin wrapper for backward compatibility).

    EasyOCR already returns results in the desired format:
    [(bbox, text, confidence), ...]
    """

    def __init__(self, lang: List[str] = None, gpu: bool = True):
        """
        Initialize EasyOCR engine.

        Args:
            lang: List of language codes (default: ['en'] for English)
            gpu: Whether to use GPU acceleration (default: True)
        """
        if lang is None:
            lang = ['en']

        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "EasyOCR is not installed. "
                "Install it with: pip install easyocr"
            )

        self.ocr = easyocr.Reader(lang, gpu=gpu)
        self.engine_name = 'EasyOCR'

    def readtext(self, image: np.ndarray) -> List[Tuple[List, str, float]]:
        """
        Detect and recognize text using EasyOCR.

        Args:
            image: Input image as numpy array

        Returns:
            List of (bbox, text, confidence) tuples
        """
        try:
            return self.ocr.readtext(image)
        except Exception as e:
            print(f"[WARNING] EasyOCR error: {e}")
            return []


def create_ocr_engine(
    engine_type: str = 'paddleocr',
    lang: str = 'en',
    use_gpu: bool = True,
    use_angle_cls: bool = True,
    det: bool = True,
    rec: bool = True,
    cls: bool = True,
    show_log: bool = False
) -> OCREngine:
    """
    Factory function to create OCR engine instances.

    Args:
        engine_type: Type of OCR engine ('paddleocr' or 'easyocr')
        lang: Language code for OCR (default: 'en')
        use_gpu: Whether to use GPU acceleration (default: True)
        use_angle_cls: Enable angle classification (PaddleOCR only, default: True)
        det: Enable text detection (PaddleOCR only, default: True)
        rec: Enable text recognition (PaddleOCR only, default: True)
        cls: Enable text direction classification (PaddleOCR only, default: True)
        show_log: Show debug logs (PaddleOCR only, default: False)

    Returns:
        OCREngine: An OCR engine instance with a readtext() method

    Raises:
        ValueError: If engine_type is not recognized
        ImportError: If required OCR library is not installed
    """
    if engine_type == 'paddleocr':
        return PaddleOCRAdapter(
            lang=lang,
            use_gpu=use_gpu,
            use_angle_cls=use_angle_cls,
            det=det,
            rec=rec,
            cls=cls,
            show_log=show_log
        )
    elif engine_type == 'easyocr':
        return EasyOCRAdapter(lang=[lang], gpu=use_gpu)
    else:
        raise ValueError(
            f"Unknown OCR engine: {engine_type}. "
            f"Supported engines: 'paddleocr', 'easyocr'"
        )
