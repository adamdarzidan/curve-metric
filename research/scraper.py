#!/usr/bin/env python3
"""
Hospital Privacy Notice PDF Processor
======================================
Downloads each hospital's privacy notice PDF, extracts clean English
paragraph-level text, and writes one .txt file per hospital where each
chunk is separated by a blank line.

Usage:
    pip install pdfplumber requests
    python process_privacy_pdfs.py

Outputs go to ./chunks/ directory.
Failures are logged to ./chunks/_failed.log
"""

import json
import os
import re
import sys
import time
import unicodedata
import requests
import pdfplumber

# ── Config ──────────────────────────────────────────────────────────────────
JSON_FILE       = "research/hospitals.json"
CHUNKS_DIR      = "research/docs/"
FAILED_LOG      = os.path.join(CHUNKS_DIR, "_failed.log")
MIN_CHUNK_WORDS = 8        # discard very short fragments
CHUNK_SEPARATOR = "\n\n"   # blank line between chunks in output file
REQUEST_TIMEOUT = 30       # seconds per PDF download
DELAY_BETWEEN   = 1.5      # seconds between requests (polite scraping)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchBot/1.0; "
        "+mailto:yourname@yourschool.edu)"
    )
}

# ── Regex patterns to strip ──────────────────────────────────────────────────
# Phone numbers  (xxx) xxx-xxxx | xxx-xxx-xxxx | 1-800-xxx-xxxx
RE_PHONE = re.compile(
    r"(\+?1[-.\s]?)?"
    r"(\(?\d{3}\)?[-.\s]?)"
    r"\d{3}[-.\s]\d{4}"
)

# Lines that are almost certainly non-English or junk
def is_mostly_non_english(text: str) -> bool:
    """Detect non-English or symbol-heavy text that should be filtered."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True  # No letters = probably junk
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    return non_ascii / len(letters) > 0.3  # More aggressive: >30% non-ASCII

def is_symbol_junk(text: str) -> bool:
    """Detect lines that are mostly symbols/non-printable characters."""
    if len(text) < 3:
        return True
    # Count printable ASCII characters
    printable = sum(1 for c in text if 32 <= ord(c) <= 126 or c.isspace())
    return printable / len(text) < 0.4  # Less than 40% ASCII printable

def has_english_words(text: str) -> bool:
    """Check if text contains recognizable English words."""
    # Common English words to look for
    common_words = {
        'the', 'and', 'of', 'to', 'in', 'is', 'you', 'for', 'that', 'with',
        'this', 'have', 'will', 'your', 'from', 'are', 'not', 'be', 'can',
        'has', 'or', 'as', 'by', 'an', 'on', 'we', 'it', 'may', 'information',
        'patient', 'health', 'care', 'medical', 'hospital', 'privacy', 'policy',
        'right', 'access', 'treatment', 'service', 'services', 'agreement'
    }
    words = text.lower().split()
    english_word_count = sum(1 for w in words if any(ew in w for ew in common_words))
    return english_word_count >= 1  # At least one recognizable English word

# Known non-content lines to skip (headers, footers, page numbers, form labels)
RE_SKIP = re.compile(
    r"^\s*("
    r"page\s*\d+(\s*of\s*\d+)?"        # "Page 3 of 7"
    r"|©.*"                              # copyright lines
    r"|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"  # dates like 01/01/2024
    r"|www\.\S+"                         # bare URLs
    r"|https?://\S+"                     # full URLs
    r"|fax:.*"                           # fax lines
    r"|tty:.*"                           # TTY lines
    r"|rev(ised|\.)\s*\d+"              # revision stamps
    r")\s*$",
    re.IGNORECASE,
)


def clean_text(raw: str) -> str:
    """Normalize unicode, remove phones, collapse whitespace, strip junk chars."""
    # Normalize unicode (ligatures, smart quotes, etc.)
    text = unicodedata.normalize("NFKC", raw)
    
    # Remove phone numbers
    text = RE_PHONE.sub("", text)
    
    # Strip excessive non-ASCII mathematical symbols, decorative chars, etc
    # Keep letters, numbers, common punctuation, and whitespace
    cleaned_chars = []
    for c in text:
        code = ord(c)
        # Keep: ASCII printable (32-126), common accented chars (192-255), whitespace
        if (32 <= code <= 126 or  # ASCII letters, numbers, punctuation, space
            192 <= code <= 255 or  # Latin extended-A (à, é, ñ, etc)
            c in '\n\r\t' or      # Whitespace
            c.isspace()):          # Other whitespace
            cleaned_chars.append(c)
        # Replace other unicode with space to maintain word boundaries
        elif code > 127:
            cleaned_chars.append(' ')
    
    text = ''.join(cleaned_chars)
    
    # Collapse excessive whitespace within line but preserve newlines
    lines = text.splitlines()
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in lines]
    return "\n".join(lines)


def extract_chunks(pdf_path: str) -> list[str]:
    """
    Open a PDF and return a list of clean text chunks.
    Aggressively filters non-English text, symbols, and encoding junk.
    """
    all_paragraphs: list[str] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_text_length = 0
            
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=4)
                if not page_text:
                    continue

                page_text = clean_text(page_text)
                total_text_length += len(page_text)
                lines = page_text.splitlines()

                current_block: list[str] = []

                for line in lines:
                    stripped = line.strip()

                    # Empty line = paragraph boundary
                    if not stripped:
                        if current_block:
                            paragraph = " ".join(current_block).strip()
                            all_paragraphs.append(paragraph)
                            current_block = []
                        continue

                    # Skip non-content lines
                    if RE_SKIP.match(stripped):
                        continue

                    # Skip symbol junk
                    if is_symbol_junk(stripped):
                        continue

                    # Skip non-English lines
                    if is_mostly_non_english(stripped):
                        continue
                    
                    # Only keep lines with recognizable English words
                    if not has_english_words(stripped):
                        continue

                    current_block.append(stripped)

                # Flush last block on page
                if current_block:
                    all_paragraphs.append(" ".join(current_block).strip())
    
    except Exception as e:
        print(f"    Error extracting text from PDF: {e}")
        return []

    # Merge short fragments and filter
    merged: list[str] = []
    for para in all_paragraphs:
        word_count = len(para.split())
        if word_count < MIN_CHUNK_WORDS and merged:
            # Append short fragment to previous chunk
            merged[-1] = merged[-1] + " " + para
        else:
            merged.append(para)

    # Final filtering: keep only chunks with enough English words
    chunks = [
        c for c in merged
        if len(c.split()) >= MIN_CHUNK_WORDS and has_english_words(c)
    ]

    return chunks


def download_pdf(url: str, dest_path: str) -> bool:
    """Download PDF to dest_path. Returns True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT,
                            stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
            # Some servers redirect to an HTML error page
            if b"<html" in resp.content[:200].lower():
                return False
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    ✗ Download error: {e}")
        return False


def process_hospital_documents(hospital: dict, base_dir: str = CHUNKS_DIR, tmp_dir: str = "tmp") -> None:
    """
    Process all PDFs listed under hospital['documents'].
    Writes each processed PDF text to a separate .txt file under:
        base_dir / hospital['directory'] / <pdf_name>.txt
    Aggressively filters junk, non-English text, and corrupted content.
    """
    if "directory" not in hospital or not hospital["directory"]:
        print(f"  Skipping hospital with no directory field: {hospital.get('name')}")
        return

    hospital_dir = os.path.join(base_dir, hospital["directory"])
    os.makedirs(hospital_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    for doc in hospital.get("documents", []):
        url = doc.get("url", "")
        if not url:
            print(f"  Skipping empty URL for {hospital['name']}")
            continue

        safe_pdf_name = re.sub(r"[^\w]", "_", doc.get("type", "doc"))
        pdf_filename = f"{safe_pdf_name}.pdf"
        pdf_path = os.path.join(tmp_dir, pdf_filename)

        print(f"  Downloading: {url}")
        if not download_pdf(url, pdf_path):
            print(f"    Download failed for {url}")
            continue

        # Check file size - skip obviously huge/corrupted files
        pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        if pdf_size_mb > 50:
            print(f"    Skipping huge PDF ({pdf_size_mb:.1f} MB - likely corrupted)")
            try:
                os.remove(pdf_path)
            except OSError:
                pass
            continue

        print(f"  Extracting text …")
        chunks = extract_chunks(pdf_path)
        
        if chunks:
            # Calculate final file size
            final_text = CHUNK_SEPARATOR.join(chunks)
            final_size_mb = len(final_text) / (1024 * 1024)
            
            # Skip if cleaned output is still huge (sign of junk)
            if final_size_mb > 10:
                print(f"    Skipping - cleaned output too large ({final_size_mb:.1f} MB, likely junk)")
            else:
                # Write chunks to a .txt file in hospital_dir
                txt_path = os.path.join(hospital_dir, f"{safe_pdf_name}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(final_text)
                print(f"    ✓ Saved {len(chunks)} chunks ({final_size_mb:.2f} MB) to {txt_path}")
        else:
            print(f"    ✗ No valid English text extracted from {url}")

        # Clean up PDF
        try:
            os.remove(pdf_path)
        except OSError:
            pass

        # polite scraping delay
        time.sleep(DELAY_BETWEEN)