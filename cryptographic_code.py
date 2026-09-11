#!/usr/bin/env python3
"""
UISP 80-bit 2D Visual Cryptographic Code (Single-file implementation)

- Grid: 8 rows x 10 columns (80 nodes)
- Representation: filled circle = 0, hollow ring = 1
- Orientation marker: solid L-shaped line along bottom and right edges (meeting at bottom-right)
- Mapping order: "Top to Bottom, Left to Right" -> column-major:
    for col in 0..9:
        for row in 0..7:
            bit_index = col * 8 + row
- Output: generate_code(hex_id, save_path)
- Input: read_code(image_path) -> returns decoded 20-char hex string (uppercase)

Usage: Requires OpenCV (cv2), numpy and requests
    pip install opencv-python-headless numpy requests

Backend integration:
- The two placeholder backend functions now perform HTTP requests. Configure the following
  environment variables in production before running the script:
    BACKEND_FETCH_URL   -> HTTP(S) GET endpoint that returns JSON with the user's UISP_ID
    BACKEND_VERIFY_URL  -> HTTP(S) POST endpoint that accepts JSON payload for verification
    BACKEND_API_TOKEN   -> (optional) Bearer token for Authorization header

- Expected JSON responses (example):
  Fetch (GET): { "uisp_id": "0123456789ABCDEF0123" }
  Verify (POST): { "verified": true }

Adjust JSON keys/field names to match your backend contract.
"""

import cv2
import numpy as np
import os
import math
import tempfile
import json
import time
from typing import Optional

# HTTP requests
try:
    import requests
    from requests.exceptions import RequestException
except Exception as e:
    requests = None
    RequestException = Exception

# ---- Constants ----
ROWS = 8
COLS = 10
BITS = ROWS * COLS  # 80
HEX_LEN = 20        # 20 hex chars * 4 = 80 bits

# Environment variables for backend endpoints and token
ENV_FETCH_URL = "BACKEND_FETCH_URL"
ENV_VERIFY_URL = "BACKEND_VERIFY_URL"
ENV_API_TOKEN = "BACKEND_API_TOKEN"

# Image and drawing defaults
DEFAULT_IMG_WIDTH = 1200
DEFAULT_IMG_HEIGHT = 1000
DEFAULT_MARGIN = 80  # pixels around grid
DEFAULT_RING_THICKNESS = 8
ORIENTATION_LINE_THICKNESS = 14
ORIENTATION_LINE_COLOR = (0, 0, 0)  # black
MARKER_COLOR = (0, 0, 0)  # black
BACKGROUND_COLOR = (255, 255, 255)  # white

# ---- Utility functions ----

def hex_to_bin80(hex_id: str) -> str:
    """Convert a 20-character hex string into an 80-character binary string.
       Pads with leading zeros if necessary. Raises ValueError on invalid lengths.
    """
    h = hex_id.strip().lstrip("0x").lower()
    if len(h) != HEX_LEN:
        raise ValueError(f"UISP_ID must be {HEX_LEN} hex characters (got {len(h)}).")
    try:
        i = int(h, 16)
    except ValueError as e:
        raise ValueError("UISP_ID contains non-hex characters") from e
    b = format(i, f'0{BITS}b')
    return b

def bin80_to_hex(bin_str: str) -> str:
    """Convert an 80-bit binary string to a 20-character hex string (uppercase)."""
    if len(bin_str) != BITS:
        raise ValueError(f"Binary string must be {BITS} bits.")
    i = int(bin_str, 2)
    hx = format(i, f'0{HEX_LEN}x').upper()
    return hx

def rotate_image_cv(img: np.ndarray, k: int) -> np.ndarray:
    """Rotate image by 90*k degrees clockwise using cv2.rotate for exact pixel mapping.
       k must be 0..3.
    """
    k = k % 4
    if k == 0:
        return img.copy()
    if k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    if k == 3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# ---- Main class ----

class VisualCryptoCode:
    def __init__(self,
                 rows: int = ROWS,
                 cols: int = COLS,
                 width: int = DEFAULT_IMG_WIDTH,
                 height: int = DEFAULT_IMG_HEIGHT,
                 margin: int = DEFAULT_MARGIN,
                 circle_radius: Optional[int] = None,
                 ring_thickness: int = DEFAULT_RING_THICKNESS,
                 orientation_thickness: int = ORIENTATION_LINE_THICKNESS):
        self.rows = rows
        self.cols = cols
        self.bits = rows * cols
        self.width = width
        self.height = height
        self.margin = margin
        self.ring_thickness = ring_thickness
        self.orientation_thickness = orientation_thickness

        # Cell size and circle radius calculation
        if circle_radius is None:
            usable_w = max(100, self.width - 2 * self.margin)
            usable_h = max(100, self.height - 2 * self.margin)
            cell_w = usable_w / self.cols
            cell_h = usable_h / self.rows
            cell = int(min(cell_w, cell_h))
            self.cell = cell
            self.circle_radius = max(6, int(cell * 0.4))
        else:
            self.circle_radius = circle_radius
            self.cell = None

    # ---------------- Encoding ----------------

    def generate_code(self, hex_id: str, save_path: str) -> str:
        """Create the code image from hex_id and save it to save_path.
           Returns the absolute path to the saved file.
        """
        bin_str = hex_to_bin80(hex_id)
        img = np.full((self.height, self.width, 3), BACKGROUND_COLOR, dtype=np.uint8)

        left = self.margin
        top = self.margin
        right = self.width - self.margin
        bottom = self.height - self.margin

        usable_w = right - left
        usable_h = bottom - top

        cell_w = usable_w / self.cols
        cell_h = usable_h / self.rows

        for col in range(self.cols):
            for row in range(self.rows):
                idx = col * self.rows + row  # column-major ordering
                bit = bin_str[idx]
                cx = int(left + (col + 0.5) * cell_w)
                cy = int(top + (row + 0.5) * cell_h)

                if bit == '0':
                    cv2.circle(img, (cx, cy), self.circle_radius, MARKER_COLOR, thickness=-1, lineType=cv2.LINE_AA)
                else:
                    cv2.circle(img, (cx, cy), self.circle_radius, MARKER_COLOR, thickness=self.ring_thickness, lineType=cv2.LINE_AA)

        l = int(left)
        t = int(top)
        r = int(right)
        b = int(bottom)
        thickness = int(self.orientation_thickness)

        cv2.rectangle(img, (r - thickness, t), (r, b + thickness), ORIENTATION_LINE_COLOR, thickness=-1)
        cv2.rectangle(img, (l - thickness, b - thickness), (r + thickness, b), ORIENTATION_LINE_COLOR, thickness=-1)

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        cv2.imwrite(save_path, img)
        return os.path.abspath(save_path)

    # ---------------- Decoding ----------------

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        return th

    def _find_orientation_rotation(self, img: np.ndarray) -> int:
        best_k = 0
        best_score = -1
        for k in range(4):
            rot = rotate_image_cv(img, k)
            h, w = rot.shape[:2]
            strip_w = max(8, int(0.08 * w))
            strip_h = max(8, int(0.08 * h))

            right_strip = rot[:, w - strip_w - 1: w]
            bottom_strip = rot[h - strip_h - 1: h, :]

            gray_r = cv2.cvtColor(right_strip, cv2.COLOR_BGR2GRAY)
            gray_b = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)
            _, tr = cv2.threshold(gray_r, 200, 255, cv2.THRESH_BINARY_INV)
            _, tb = cv2.threshold(gray_b, 200, 255, cv2.THRESH_BINARY_INV)
            prop_r = tr.mean() / 255.0
            prop_b = tb.mean() / 255.0

            corner_size = max(8, int(0.08 * min(w, h)))
            corner = rot[h - corner_size: h, w - corner_size: w]
            gray_c = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            _, tc = cv2.threshold(gray_c, 200, 255, cv2.THRESH_BINARY_INV)
            prop_c = tc.mean() / 255.0

            score = (prop_r + prop_b) * 0.6 + prop_c * 0.4

            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def read_code(self, image_path: str, debug: bool = False) -> str:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        k = self._find_orientation_rotation(img_bgr)
        img_norm = rotate_image_cv(img_bgr, k)

        bin_img = self._preprocess_image(img_norm)

        contours, hierarchy = cv2.findContours(bin_img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours or hierarchy is None:
            raise ValueError("No contours found while decoding.")

        hierarchy = hierarchy[0]
        candidates = []
        for i, cnt in enumerate(contours):
            hinfo = hierarchy[i]
            parent = hinfo[3]
            if parent != -1:
                continue
            area = cv2.contourArea(cnt)
            if area < 5.0:
                continue
            perim = cv2.arcLength(cnt, True)
            circularity = 0.0
            if perim > 1e-3:
                circularity = 4.0 * math.pi * area / (perim * perim)
            M = cv2.moments(cnt)
            if abs(M.get('m00', 0)) < 1e-6:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            has_child = (hinfo[2] != -1)
            candidates.append({
                'idx': i,
                'area': area,
                'perim': perim,
                'circularity': circularity,
                'cx': cx, 'cy': cy,
                'has_child': has_child,
                'contour': cnt
            })

        if len(candidates) < self.bits:
            raise ValueError(f"Not enough marker candidates found: {len(candidates)} (expected {self.bits})")

        candidates_sorted = sorted(candidates, key=lambda x: (-x['circularity'], x['area']))
        selected = candidates_sorted[:self.bits]

        if len(selected) != self.bits:
            raise ValueError(f"Could not isolate exactly {self.bits} markers (found {len(selected)}).")

        cxs = [c['cx'] for c in selected]
        cys = [c['cy'] for c in selected]
        min_x, max_x = min(cxs), max(cxs)
        min_y, max_y = min(cys), max(cys)

        pad_x = max(4, int(0.02 * (max_x - min_x + 1)))
        pad_y = max(4, int(0.02 * (max_y - min_y + 1)))
        min_x -= pad_x; min_y -= pad_y; max_x += pad_x; max_y += pad_y

        cell_w = (max_x - min_x) / float(self.cols)
        cell_h = (max_y - min_y) / float(self.rows)
        if cell_w <= 0 or cell_h <= 0:
            raise ValueError("Invalid grid geometry detected while decoding.")

        grid_bits = [['X' for _ in range(self.cols)] for __ in range(self.rows)]
        assigned = 0

        mapping = {}
        for c in selected:
            col_f = (c['cx'] - min_x) / cell_w
            row_f = (c['cy'] - min_y) / cell_h
            col = int(round(col_f))
            row = int(round(row_f))
            col = max(0, min(self.cols - 1, col))
            row = max(0, min(self.rows - 1, row))
            center_x = int(min_x + (col + 0.5) * cell_w)
            center_y = int(min_y + (row + 0.5) * cell_h)
            dist = math.hypot(c['cx'] - center_x, c['cy'] - center_y)
            key = (row, col)
            if key in mapping:
                prev = mapping[key]
                prev_dist = math.hypot(prev['cx'] - center_x, prev['cy'] - center_y)
                if dist < prev_dist:
                    mapping[key] = c
            else:
                mapping[key] = c

        for (row, col), c in mapping.items():
            grid_bits[row][col] = '1' if c['has_child'] else '0'
            assigned += 1

        if assigned < self.bits:
            occupied = set(mapping.keys())
            unused = [c for c in selected if (int(round((c['cy'] - min_y)/cell_h)), int(round((c['cx'] - min_x)/cell_w))) not in occupied]
            for c in unused:
                best_cell = None
                best_dist = float('inf')
                for r in range(self.rows):
                    for co in range(self.cols):
                        if (r, co) in occupied:
                            continue
                        center_x = int(min_x + (co + 0.5) * cell_w)
                        center_y = int(min_y + (r + 0.5) * cell_h)
                        dist = math.hypot(c['cx'] - center_x, c['cy'] - center_y)
                        if dist < best_dist:
                            best_dist = dist
                            best_cell = (r, co)
                if best_cell is not None:
                    r, co = best_cell
                    grid_bits[r][co] = '1' if c['has_child'] else '0'
                    occupied.add((r, co))
                    assigned += 1
                    if assigned >= self.bits:
                        break

        if assigned < self.bits:
            raise ValueError(f"After assignment, not all grid cells are filled (assigned {assigned}/{self.bits}).")

        bits_out = []
        for col in range(self.cols):
            for row in range(self.rows):
                v = grid_bits[row][col]
                if v not in ('0', '1'):
                    raise ValueError(f"Ambiguous cell at row {row}, col {col}")
                bits_out.append(v)

        bin_str = ''.join(bits_out)
        if len(bin_str) != self.bits:
            raise ValueError("Reconstructed binary length mismatch.")

        hex_out = bin80_to_hex(bin_str)
        return hex_out

# ---------------- Backend hooks (HTTP) ----------------

def _make_auth_headers() -> dict:
    """Helper: construct Authorization headers if BACKEND_API_TOKEN is set in env."""
    headers = {"Accept": "application/json"}
    token = os.environ.get(ENV_API_TOKEN)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_data_from_backend(retries: int = 2, timeout: float = 6.0) -> str:
    """Fetch UISP_ID from configured backend endpoint (HTTP GET).

    Environment variables used:
      BACKEND_FETCH_URL -> required. Example: https://api.example.com/uisp/id?user=123
      BACKEND_API_TOKEN -> optional bearer token

    Expected JSON response (example): { "uisp_id": "20HEXCHARS..." }

    Raises ValueError/RuntimeError on fatal errors.
    """
    if requests is None:
        raise RuntimeError("requests library is required for backend integration. Install with 'pip install requests'.")

    url = os.environ.get(ENV_FETCH_URL)
    if not url:
        raise ValueError(f"Environment variable {ENV_FETCH_URL} must be set to the backend fetch endpoint URL.")

    headers = _make_auth_headers()

    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            # Expect HTTP 200
            if resp.status_code != 200:
                # Try to include response body for debugging
                raise RuntimeError(f"Backend returned status {resp.status_code}: {resp.text}")
            data = resp.json()
            # Accept multiple possible keys to be flexible with backend
            for key in ("uisp_id", "uispId", "id", "uId"):
                if key in data:
                    uisp = data[key]
                    if isinstance(uisp, str) and len(uisp.strip()) == HEX_LEN:
                        return uisp.strip().upper()
                    # If hex is not exact length, still attempt to normalize
                    if isinstance(uisp, str):
                        cleaned = uisp.strip().replace("0x", "").upper()
                        if len(cleaned) == HEX_LEN:
                            return cleaned
                        # otherwise continue to error below
            raise RuntimeError(f"Backend response did not contain a valid '{HEX_LEN}'-char UISP_ID: {json.dumps(data)}")
        except RequestException as e:
            last_exc = e
            # simple backoff
            time.sleep(0.2 + attempt * 0.5)
            continue
    # All retries failed
    raise RuntimeError(f"Failed to fetch UISP_ID from backend after {retries+1} attempts: {last_exc}")


def send_to_backend_for_verification(scanned_hex: str, expected_hex: Optional[str] = None,
                                     retries: int = 2, timeout: float = 6.0) -> bool:
    """Send scanned_hex (and optionally expected_hex) to backend verification endpoint (HTTP POST).

    Environment variables used:
      BACKEND_VERIFY_URL -> required. Example: https://api.example.com/uisp/verify
      BACKEND_API_TOKEN -> optional bearer token

    Expected request JSON: { "scanned_hex": "...", "expected_hex": "..." }
    Expected response JSON: { "verified": true }  or { "success": true }

    Returns True if backend reports verification success, False otherwise.
    """
    if requests is None:
        raise RuntimeError("requests library is required for backend integration. Install with 'pip install requests'.")

    url = os.environ.get(ENV_VERIFY_URL)
    if not url:
        raise ValueError(f"Environment variable {ENV_VERIFY_URL} must be set to the backend verify endpoint URL.")

    headers = _make_auth_headers()
    headers["Content-Type"] = "application/json"

    payload = {"scanned_hex": scanned_hex}
    if expected_hex:
        payload["expected_hex"] = expected_hex

    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                raise RuntimeError(f"Backend verification returned {resp.status_code}: {body}")
            data = resp.json()
            if isinstance(data, dict):
                if data.get("verified") is True:
                    return True
                if data.get("success") is True:
                    return True
                st = data.get("status")
                if isinstance(st, str) and st.lower() in ("ok", "verified", "success"):
                    return True
            return False
        except RequestException as e:
            last_exc = e
            time.sleep(0.2 + attempt * 0.5)
            continue
    raise RuntimeError(f"Failed to contact verification backend after {retries+1} attempts: {last_exc}")

# ---- Execution block tying it together ----

if __name__ == '__main__':
    print("UISP Visual Cryptographic Code generator/reader starting...")

    try:
        uisp_id = fetch_data_from_backend()
    except Exception as e:
        print(f"[ERROR] Could not fetch UISP_ID from backend: {e}")
        raise

    print(f"[Step] Fetched UISP_ID from backend: {uisp_id}")

    vcc = VisualCryptoCode(width=1200, height=1000, margin=80, circle_radius=34, ring_thickness=10)

    tmp_dir = tempfile.gettempdir()
    out_path = os.path.join(tmp_dir, f"uisp_code_{uisp_id}.png")
    print(f"[Step] Generating code image at: {out_path}")
    try:
        saved = vcc.generate_code(uisp_id, out_path)
        print(f"[OK] Image saved to: {saved}")
    except Exception as e:
        print(f"[ERROR] Failed to generate image: {e}")
        raise

    print("[Step] Decoding the generated image...")
    try:
        decoded = vcc.read_code(out_path)
        print(f"[OK] Decoded UISP_ID: {decoded}")
    except Exception as e:
        print(f"[ERROR] Decoding failed: {e}")
        raise

    try:
        verified = send_to_backend_for_verification(decoded, expected_hex=uisp_id)
    except Exception as e:
        print(f"[ERROR] Verification request failed: {e}")
        raise

    if verified:
        print("[RESULT] Verification successful: scanned code matches backend UISP_ID.")
    else:
        print("[RESULT] Verification failed: scanned code DOES NOT match backend UISP_ID.")

    print("Done. The image file remains on disk for inspection.")
