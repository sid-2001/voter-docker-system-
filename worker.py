import os
import re
import gc
import cv2
import fitz
import uuid
import time
import socket
import logging
import threading
import numpy as np

from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCR
from pymongo import MongoClient, UpdateOne, ReturnDocument
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# CONFIG
# =========================================================

PDF_FOLDER = os.getenv("PDF_FOLDER", "./practical")
START_PAGE = int(os.getenv("START_PAGE", "2"))
PDF_DPI = int(os.getenv("PDF_DPI", "150"))

ROWS = 10
COLS = 3

LEFT_MARGIN = 35
TOP_MARGIN = 55
RIGHT_MARGIN = 35
BOTTOM_MARGIN = 20
GAP_X = 18
GAP_Y = 8
PADDING = 20

OCR_THREADS = int(os.getenv("OCR_THREADS", "4"))
DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "2000"))

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongodb:27017")
DB_NAME = os.getenv("DB_NAME", "srs_database")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "srs_2025")
JOBS_COLLECTION_NAME = os.getenv("JOBS_COLLECTION_NAME", "srs_pdf_jobs")

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
STALE_SECONDS = int(os.getenv("STALE_SECONDS", "300"))
HEARTBEAT_SECONDS = int(os.getenv("HEARTBEAT_SECONDS", "30"))

WORKER_ID = os.getenv("WORKER_ID", f"{socket.gethostname()}-{os.getpid()}")

# =========================================================
# LOGGING
# =========================================================

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("pdf_ocr_worker")
logger.setLevel(LOG_LEVEL)
logger.handlers.clear()

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

file_handler = RotatingFileHandler(LOG_DIR / f"worker-{WORKER_ID}.log", maxBytes=10_000_000, backupCount=5)
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

# =========================================================
# MONGODB
# =========================================================

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
jobs_collection = db[JOBS_COLLECTION_NAME]
stats_collection = db["srs_run_stats"]

collection.create_index([("record_uid", 1)], unique=True)
collection.create_index([("idcard_no", 1)])
collection.create_index([("assembly_name", 1)])
collection.create_index([("section_name", 1)])

jobs_collection.create_index([("pdf_name", 1)], unique=True)
jobs_collection.create_index([("status", 1), ("heartbeat_at", 1)])

# =========================================================
# OCR THREAD LOCAL
# =========================================================

os.environ["FLAGS_enable_pir_api"] = "0"
_thread_local = threading.local()

def get_ocr():
    if not hasattr(_thread_local, "ocr"):
        _thread_local.ocr = PaddleOCR(
            use_textline_orientation=False,
            lang="en",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    return _thread_local.ocr

# =========================================================
# HELPERS
# =========================================================

def now_utc():
    return datetime.now(timezone.utc)


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text):
    replacements = {
        "Fathors": "Father's",
        "Fathar": "Father",
        "Maie": "Male",
        "Femate": "Female",
        "Mals": "Male",
        "Femaie": "Female",
        "Housa": "House",
        "Narne": "Name",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def is_blank_image(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.mean(gray) > 245


def preprocess_image(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def paddle_ocr_text(image):
    processed = preprocess_image(image)
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    result = get_ocr().predict(processed_bgr)

    texts, scores = [], []
    for res in result:
        texts.extend(res.get("rec_texts", []))
        scores.extend(res.get("rec_scores", []))

    text = normalize_text(clean_text(" ".join(texts)))
    confidence = round(sum(scores) / len(scores), 4) if scores else 0
    return text, confidence


def extract_field(patterns, text):
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_text(match.group(1))
    return ""


def parse_record(text):
    voter = {"raw_text": text}
    voter["name"] = extract_field([r"Name\s*[:\-]?\s*(.*?)(?=Father|Husband|Mother|House|Age|Gender|$)"], text)

    father_name = extract_field([r"Father'?s?\s*Name\s*[:\-]?\s*(.*?)(?=House|Age|Gender|$)"], text)
    husband_name = extract_field([r"Husband'?s?\s*Name\s*[:\-]?\s*(.*?)(?=House|Age|Gender|$)"], text)
    mother_name = extract_field([r"Mother'?s?\s*Name\s*[:\-]?\s*(.*?)(?=House|Age|Gender|$)"], text)

    if father_name:
        voter["relation_name"], voter["relation"] = father_name, "F"
    elif husband_name:
        voter["relation_name"], voter["relation"] = husband_name, "H"
    elif mother_name:
        voter["relation_name"], voter["relation"] = mother_name, "M"
    else:
        voter["relation_name"], voter["relation"] = "", ""

    voter["house_no"] = extract_field([r"House\s*(?:Number|No\.?)\s*[:\-]?\s*(.*?)(?=Photo|Age|Gender|$)"], text)
    voter["age"] = extract_field([r"Age\s*[:\-]?\s*(\d+)"], text)

    gender = extract_field([r"(Male|Female)"], text).strip().lower()
    voter["sex"] = "M" if gender == "male" else ("F" if gender == "female" else gender)
    voter["idcard_no"] = extract_field([r"\b([A-Z]{3}[0-9]{6,10})\b"], text)
    return voter


def is_valid_record(voter):
    return sum(bool(voter.get(k)) for k in ("name", "age", "idcard_no")) >= 2


def generate_record_uid():
    return uuid.uuid4().hex


def save_to_database(voters):
    if not voters:
        return 0

    operations = [
        UpdateOne({"record_uid": voter["record_uid"]}, {"$setOnInsert": voter}, upsert=True)
        for voter in voters
    ]

    result = collection.bulk_write(operations, ordered=False)
    logger.info("DB UPSERTED: %s", result.upserted_count)
    return result.upserted_count


def process_cell(args):
    (img, width, height, row, col, cell_width, cell_height, assembly_name, section_name, pdf_name, page_number) = args

    x1 = LEFT_MARGIN + col * (cell_width + GAP_X)
    y1 = TOP_MARGIN + row * (cell_height + GAP_Y)
    x2 = x1 + cell_width
    y2 = y1 + cell_height

    cropped = img.crop((max(0, x1 - PADDING), max(0, y1 - PADDING), min(width, x2 + PADDING), min(height, y2 + PADDING)))
    if is_blank_image(cropped):
        return None

    text, confidence = paddle_ocr_text(cropped)
    if not text:
        return None

    voter = parse_record(text)
    if not is_valid_record(voter):
        return None

    current_time = now_utc()
    voter.update({
        "record_uid": generate_record_uid(),
        "pdf_file": pdf_name,
        "page_number": page_number,
        "row": row + 1,
        "column": col + 1,
        "confidence": confidence,
        "assembly_name": assembly_name,
        "section_name": section_name,
        "phone": None,
        "religon": None,
        "caste": None,
        "linked_member": None,
        "status": "unverified",
        "srs_status": "unauthorized",
        "verified_by": None,
        "created_by": "ocr_system",
        "updated_by": "ocr_system",
        "created_at": current_time,
        "updated_at": current_time,
    })
    return voter


def extract_page_metadata(img):
    width, height = img.size
    header_crop = img.crop((0, 0, width, int(height * 0.18)))
    text, _ = paddle_ocr_text(header_crop)

    assembly = extract_field([
        r"Assembly\s*Constituency\s*No\.?\s*and\s*Name\s*[:\-]?\s*(.*?)(?=Section|Part|Polling|$)",
        r"Assembly\s*Constituency\s*[:\-]?\s*(.*?)(?=Section|Part|Polling|$)",
        r"AC\s*[:\-]?\s*(.*?)(?=Section|Part|Polling|$)",
    ], text)

    section = extract_field([
        r"Section\s*No\.?\s*and\s*Name\s*[:\-]?\s*([0-9A-Za-z\-\s]+?)(?=\s+\d+\s+[A-Z]{3}\d+|Name\s*:|$)",
        r"([0-9]+\s*[-–]\s*[A-Za-z\s]+?)(?=\s+\d+\s+[A-Z]{3}\d+|Name\s*:|$)",
    ], text)

    section = re.sub(r"\b[A-Z]{3}\d+\b.*", "", section)
    section = re.sub(r"\s+", " ", section).strip()

    return {"assembly_name": assembly, "section_name": section}


def heartbeat_loop(pdf_name, stop_event):
    while not stop_event.is_set():
        jobs_collection.update_one(
            {"pdf_name": pdf_name, "status": "processing", "worker_id": WORKER_ID},
            {"$set": {"heartbeat_at": now_utc()}},
        )
        stop_event.wait(HEARTBEAT_SECONDS)


def process_pdf(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    logger.info("START PDF: %s", pdf_name)

    start = time.perf_counter()
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    page_images = {}
    for page_number in range(START_PAGE, total_pages - 1):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=PDF_DPI)
        page_images[page_number] = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    page_metadata = {}
    with ThreadPoolExecutor(max_workers=OCR_THREADS) as executor:
        futures = {executor.submit(extract_page_metadata, img): page for page, img in page_images.items()}
        for future in as_completed(futures):
            page = futures[future]
            try:
                page_metadata[page] = future.result()
            except Exception as exc:
                logger.exception("METADATA ERROR page=%s err=%s", page + 1, exc)
                page_metadata[page] = {"assembly_name": "", "section_name": ""}

    all_cell_args = []
    for page_number, img in page_images.items():
        metadata = page_metadata[page_number]
        width, height = img.size
        usable_width = width - LEFT_MARGIN - RIGHT_MARGIN
        usable_height = height - TOP_MARGIN - BOTTOM_MARGIN
        cell_width = (usable_width - ((COLS - 1) * GAP_X)) // COLS
        cell_height = (usable_height - ((ROWS - 1) * GAP_Y)) // ROWS

        for row in range(ROWS):
            for col in range(COLS):
                all_cell_args.append((
                    img, width, height, row, col, cell_width, cell_height,
                    metadata.get("assembly_name", ""), metadata.get("section_name", ""),
                    pdf_name, page_number + 1,
                ))

    inserted_total = 0
    batch_records = []
    with ThreadPoolExecutor(max_workers=OCR_THREADS) as executor:
        futures = [executor.submit(process_cell, args) for args in all_cell_args]
        for future in as_completed(futures):
            voter = future.result()
            if voter is not None:
                batch_records.append(voter)
                if len(batch_records) >= DB_BATCH_SIZE:
                    inserted_total += save_to_database(batch_records)
                    batch_records = []

    if batch_records:
        inserted_total += save_to_database(batch_records)

    elapsed = time.perf_counter() - start
    logger.info("DONE PDF: %s | pages=%s cells=%s records=%s time=%.1fs", pdf_name, len(page_images), len(all_cell_args), inserted_total, elapsed)
    return {
        "records": inserted_total,
        "pages": len(page_images),
        "cells": len(all_cell_args),
        "elapsed_seconds": round(elapsed, 3),
    }


def init_jobs_from_folder():
    files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    ops = []
    now = now_utc()
    for f in files:
        pdf_name = os.path.splitext(f)[0]
        ops.append(UpdateOne(
            {"pdf_name": pdf_name},
            {"$setOnInsert": {"pdf_name": pdf_name, "file_name": f, "status": "pending", "created_at": now, "updated_at": now}},
            upsert=True,
        ))
    if ops:
        jobs_collection.bulk_write(ops, ordered=False)
    logger.info("JOB QUEUE READY. total_pdfs=%s", len(files))


def claim_next_job():
    stale_before = now_utc() - timedelta(seconds=STALE_SECONDS)
    query = {
        "$or": [
            {"status": "pending"},
            {"status": "processing", "heartbeat_at": {"$lt": stale_before}},
        ]
    }
    update = {
        "$set": {
            "status": "processing",
            "worker_id": WORKER_ID,
            "started_at": now_utc(),
            "heartbeat_at": now_utc(),
            "updated_at": now_utc(),
        },
        "$inc": {"attempts": 1},
    }
    return jobs_collection.find_one_and_update(query, update, sort=[("updated_at", 1)], return_document=ReturnDocument.AFTER)


def mark_job_done(pdf_name, metrics):
    jobs_collection.update_one(
        {"pdf_name": pdf_name, "worker_id": WORKER_ID},
        {"$set": {"status": "done", "finished_at": now_utc(), "updated_at": now_utc(), "metrics": metrics}},
    )


def mark_job_failed(pdf_name, err):
    jobs_collection.update_one(
        {"pdf_name": pdf_name, "worker_id": WORKER_ID},
        {"$set": {"status": "failed", "last_error": str(err), "updated_at": now_utc()}},
    )


def log_global_stats():
    total_pdfs = jobs_collection.count_documents({})
    done_pdfs = jobs_collection.count_documents({"status": "done"})
    processing_pdfs = jobs_collection.count_documents({"status": "processing"})
    pending_pdfs = jobs_collection.count_documents({"status": "pending"})
    failed_pdfs = jobs_collection.count_documents({"status": "failed"})
    total_records = collection.count_documents({})

    logger.info(
        "GLOBAL STATS | done=%s/%s pending=%s processing=%s failed=%s total_records=%s",
        done_pdfs, total_pdfs, pending_pdfs, processing_pdfs, failed_pdfs, total_records,
    )
    stats_collection.insert_one({
        "worker_id": WORKER_ID,
        "timestamp": now_utc(),
        "total_pdfs": total_pdfs,
        "done_pdfs": done_pdfs,
        "pending_pdfs": pending_pdfs,
        "processing_pdfs": processing_pdfs,
        "failed_pdfs": failed_pdfs,
        "total_records": total_records,
    })


def main():
    logger.info("WORKER STARTED: %s", WORKER_ID)
    get_ocr()
    init_jobs_from_folder()

    while True:
        job = claim_next_job()
        if not job:
            log_global_stats()
            if jobs_collection.count_documents({"status": {"$in": ["pending", "processing"]}}) == 0:
                logger.info("NO MORE JOBS. EXITING.")
                break
            time.sleep(5)
            continue

        file_name = job["file_name"]
        pdf_name = job["pdf_name"]
        pdf_path = os.path.join(PDF_FOLDER, file_name)

        stop_event = threading.Event()
        hb_thread = threading.Thread(target=heartbeat_loop, args=(pdf_name, stop_event), daemon=True)
        hb_thread.start()

        try:
            metrics = process_pdf(pdf_path)
            mark_job_done(pdf_name, metrics)
            log_global_stats()
        except Exception as exc:
            logger.exception("PDF FAILED: %s", pdf_name)
            mark_job_failed(pdf_name, exc)
        finally:
            stop_event.set()
            hb_thread.join(timeout=2)
            gc.collect()


if __name__ == "__main__":
    main()
