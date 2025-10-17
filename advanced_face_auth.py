#!/usr/bin/env python3
# advanced_face_auth.py
# Host side for SNN+PUF biometric system (supports EE debug and 55 long10 replies)
from __future__ import annotations
import os, sys, time, json, csv, argparse, collections
from datetime import datetime
from pathlib import Path
import numpy as np

# optional libs
try:
    import cv2
    CV2_OK = True
except Exception:
    CV2_OK = False

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    import serial, serial.tools.list_ports
    SERIAL_OK = True
except Exception:
    SERIAL_OK = False

# ---------------- Config ----------------
DEVICE = 'cuda' if TORCH_OK and torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 160
MARGIN = 14
DATA_DIR = "data"
CSV_PATH = "auth_log.csv"
SERIAL_BAUD = 115200
SEND_COOLDOWN_S = 0.2
CONSISTENT_FRAMES = 3
SIM_THRESHOLD_MIN = 0.60
SIM_MARGIN = 0.05
SEND_UART_ON_GRANT = True
UART_CONTROL_PACKET = bytes([0xFE, 0x01])
CAPTURE_INTERVAL = 3.0
FORCE_AUTH_IF_SPIKE_HIGH = False
SPIKE_OVERRIDE_THRESH = 30000
PREFER_DEBUG = True  # you can prefer parsing EE debug fields when both present

# ---------------- Helpers ----------------
def lbp_checksum16_from_bgr(face_bgr):
    if face_bgr is None or getattr(face_bgr, 'size', 0) == 0:
        return 0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    face128 = cv2.resize(eq, (128, 128), interpolation=cv2.INTER_AREA)
    r = face128
    lbp = np.zeros_like(r, dtype=np.uint8)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        sh = np.roll(np.roll(r, dy, axis=0), dx, axis=1)
        lbp = ((lbp << 1) | (sh >= r).astype(np.uint8)) & 0xFF
    hist,_ = np.histogram(lbp, bins=256, range=(0,256))
    s = int(np.sum(hist.astype(np.int64) * np.arange(256, dtype=np.int64)))
    return s & 0xFFFF

def cos_sim(a,b):
    a = a / (np.linalg.norm(a)+1e-9)
    b = b / (np.linalg.norm(b)+1e-9)
    return float(np.dot(a,b))

# ---------------- Embedder ----------------
class Embedder:
    def __init__(self, device=DEVICE):
        self.device = device
        self.enabled = TORCH_OK and PIL_OK and CV2_OK
        if self.enabled:
            self.mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=MARGIN, post_process=True, device=device)
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            print("[embed] models loaded on", device)
        else:
            print("[embed] disabled (missing torch/PIL/opencv)")

    def embed_image_path(self, path):
        if not self.enabled: return None, None
        img = Image.open(path).convert('RGB')
        face = self.mtcnn(img)
        if face is None: return None, None
        with torch.no_grad():
            emb = self.model(face.unsqueeze(0).to(self.device)).detach().cpu().numpy().reshape(-1)
        emb = emb / (np.linalg.norm(emb)+1e-9)
        arr = ((face.permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
        face_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return emb, face_bgr

    def embed_pil(self, pil_img):
        if not self.enabled: return None, None
        face = self.mtcnn(pil_img)
        if face is None: return None, None
        with torch.no_grad():
            emb = self.model(face.unsqueeze(0).to(self.device)).detach().cpu().numpy().reshape(-1)
        emb = emb / (np.linalg.norm(emb)+1e-9)
        arr = ((face.permute(1,2,0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
        face_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return emb, face_bgr

# ---------------- Serial helpers ----------------
def autodetect_port():
    if not SERIAL_OK: return None
    ports = list(serial.tools.list_ports.comports())
    if not ports: return None
    for p in ports:
        desc = f"{p.device} {p.description}".lower()
        if 'usb' in desc or 'ftdi' in desc or 'serial' in desc:
            return p.device
    return ports[0].device

def open_serial(port_hint=None, baud=SERIAL_BAUD):
    if not SERIAL_OK:
        print("[serial] pyserial not installed")
        return None
    port = port_hint or autodetect_port()
    if not port:
        print("[serial] no port found")
        return None
    try:
        ser = serial.Serial(port, baud, timeout=0.5)
        print("[serial] opened", ser.port, "@", baud)
        return ser
    except Exception as e:
        print("serial open failed:", e)
        return None

def send_packet(ser, person_id, face16, timeout_s=1.5):
    if ser is None:
        return None, b""
    pkt = bytes([0xAA, person_id & 0x0F, (face16>>8)&0xFF, face16 & 0xFF])
    try:
        ser.reset_input_buffer(); ser.reset_output_buffer()
    except Exception:
        pass
    try:
        ser.write(pkt)
    except Exception as e:
        print("serial write failed:", e)
        return pkt, b""
    deadline = time.time() + timeout_s
    buf = bytearray()
    while time.time() < deadline and len(buf) < 2048:
        try:
            ch = ser.read(128)
        except Exception:
            ch = b""
        if ch:
            buf.extend(ch)
        else:
            time.sleep(0.004)
    # try a short extra read if we saw something
    if buf and (0x55 in buf or 0xEE in buf) and len(buf) < 10:
        endt = time.time() + 0.25
        while time.time() < endt:
            try:
                ch = ser.read(128)
            except:
                ch = b""
            if ch: buf.extend(ch)
            else: time.sleep(0.005)
    return pkt, bytes(buf)

# ---------------- Parsers ----------------
def parse_reply(buf: bytes):
    """Parse 0x55 long10 packet if present"""
    if not buf or 0x55 not in buf: return None
    i = buf.index(0x55); t = buf[i:]; L = len(t)
    if L >= 10 and t[0] == 0x55:
        _, idx, thr_hi, thr_lo, sp_hi, sp_lo, maj, auth, ch_hi, ch_lo = t[:10]
        thr_idx = int(idx) & 0x03
        return {"format":"long10",
                "thr_idx": thr_idx,
                "thr16": (thr_hi<<8)|thr_lo,
                "spike16": (sp_hi<<8)|sp_lo,
                "majresp": int(maj)&1,
                "auth": int(auth)&1,
                "challenge": (ch_hi<<8)|ch_lo,
                "raw_hex": t[:10].hex()}
    return None

def parse_debug_packet(buf: bytes):
    """Parse 0xEE debug packet (5 or 7 bytes) if present"""
    if not buf or 0xEE not in buf: return None
    i = buf.index(0xEE); t = buf[i:]; L = len(t)
    # new debug: 0xEE sh sl seed_hi seed_lo chi clo  (7 bytes)
    if L >= 7 and t[0] == 0xEE:
        try:
            _, sh, sl, s_hi, s_lo, chi, clo = t[:7]
            return {"marker":True,"spike16":(sh<<8)|sl,"seed16":(s_hi<<8)|s_lo,"challenge":(chi<<8)|clo,"raw_hex":t[:7].hex()}
        except Exception:
            pass
    # fallback older 5-byte debug
    if L >= 5 and t[0] == 0xEE:
        _, sh, sl, chi, clo = t[:5]
        return {"marker":True,"spike16":(sh<<8)|sl,"seed16":None,"challenge":(chi<<8)|clo,"raw_hex":t[:5].hex()}
    return None

# ---------------- CSV ----------------
def open_csv(path):
    new = not os.path.exists(path)
    f = open(path, 'a', newline='', encoding='utf-8')
    w = csv.writer(f)
    if new:
        header = ['timestamp','mode','frame_id','det_conf','recognized','similarity','rec_label','rec_name',
                  'packet_pid','face16_hex','spike16_hex','seed16_hex','thr16_hex','thr8_hex','thr_idx','challenge_hex',
                  'majresp','auth','tx_hex','rx_hex','rx_raw_hex','debug_raw_hex','raw_parsed_format','raw_parsed_json',
                  'image_path','facecrop_path','true_label']
        w.writerow(header); f.flush()
    return f, w

# ---------------- Live mode ----------------
def live_mode(args):
    embedder = Embedder() if TORCH_OK and PIL_OK and CV2_OK else None
    ser = open_serial(args.serial_port, args.serial_baud) if (args.serial_port or SERIAL_OK) else None
    csvf, csvw = open_csv(args.csv)

    # load enroll images
    enroll_map = {}
    enroll_root = Path(args.data_dir) / "enroll"
    if enroll_root.exists():
        for d in sorted(enroll_root.iterdir()):
            if d.is_dir():
                enroll_map[d.name] = []
                for img in sorted(d.glob("*")):
                    if img.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
                        if embedder and embedder.enabled:
                            emb, face_bgr = embedder.embed_image_path(str(img))
                        else:
                            emb, face_bgr = None, None
                        f16 = lbp_checksum16_from_bgr(face_bgr) if face_bgr is not None else 0
                        enroll_map[d.name].append((emb, f16, str(img)))

    # compute centroids
    centroids = {}; face16_mean = {}; avg_sim = {}
    for lbl, items in enroll_map.items():
        embs = [e for (e,f,p) in items if e is not None]
        if embs:
            embs = np.vstack(embs); cent = np.mean(embs, axis=0); cent = cent / (np.linalg.norm(cent)+1e-9)
            centroids[lbl] = cent
            face16_mean[lbl] = int(round(np.mean([f for (e,f,p) in items])))
            sims = [cos_sim(e, cent) for (e,f,p) in items if e is not None]
            avg_sim[lbl] = float(np.mean(sims)) if sims else 0.8
    per_user_thr = {lbl: max(SIM_THRESHOLD_MIN, avg_sim.get(lbl,0.75)-SIM_MARGIN) for lbl in centroids.keys()}
    global_thr = float(np.mean(list(per_user_thr.values()))) if per_user_thr else SIM_THRESHOLD_MIN
    print("[live] enrolled:", list(centroids.keys()), "global_thr=", global_thr)

    if not CV2_OK:
        print("OpenCV required for live mode"); return
    cap = cv2.VideoCapture(0)
    if not cap or not cap.isOpened():
        print("Cannot open camera"); return

    recent = collections.deque(maxlen=args.consistent_frames)
    last_granted = None
    last_send_time = 0.0
    last_periodic_send = 0.0
    frame_id = 0
    try:
        while True:
            ret, frame = cap.read(); frame_id += 1
            if not ret: time.sleep(0.01); continue
            best_lbl = None; best_sim = -10.0; rec_name = ""
            face16 = 0; recognized = False; face_bgr = None; face_pil = None
            if embedder and embedder.enabled:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                emb, face_bgr = embedder.embed_pil(pil)
                if emb is not None and centroids:
                    for lbl,cent in centroids.items():
                        s = cos_sim(emb, cent)
                        if s > best_sim: best_sim = s; best_lbl = lbl
                    recognized = (best_sim >= per_user_thr.get(best_lbl, global_thr)) and (best_lbl is not None)
                    rec_name = best_lbl
                face16 = lbp_checksum16_from_bgr(face_bgr) if face_bgr is not None else 0

            cv2.putText(frame, f"best={best_lbl} sim={best_sim:.3f} face16=0x{face16:04X}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if recognized else (0,120,255),2)
            cv2.imshow("Advanced Face Auth (live)", frame)
            recent.append((best_lbl, best_sim, face16, bool(recognized)))
            granted = False
            if len(recent) >= args.consistent_frames:
                tail = list(recent)[-args.consistent_frames:]
                if all(t[3] for t in tail) and all(t[0] == tail[0][0] for t in tail):
                    granted = True

            now = time.time()

            # Periodic capture send
            if now - last_periodic_send >= CAPTURE_INTERVAL:
                last_periodic_send = now
                if best_lbl is not None:
                    try:
                        pid = int(best_lbl) if (best_lbl is not None and str(best_lbl).isdigit()) else (abs(hash(str(best_lbl))) & 0x0F)
                    except:
                        pid = (abs(hash(str(best_lbl))) & 0x0F)
                    face16_val = int(face16) & 0xFFFF
                    pkt, rx = send_packet(ser, pid, face16_val) if ser else (None, b"")
                    rx_hex = rx.hex() if rx else ""
                    parsed = parse_reply(rx)
                    debug = parse_debug_packet(rx)
                    # robust pick logic (prefer debug if configured)
                    spike16_hex = ""; challenge_hex = ""; seed16_hex = ""; thr16_hex=""; thr8_hex=""; thr_idx_val=""; maj=""; auth=None
                    if PREFER_DEBUG:
                        if debug and debug.get('spike16') is not None: spike16_hex = f"0x{debug['spike16']:04X}"
                        elif parsed and parsed.get('spike16') is not None: spike16_hex = f"0x{parsed['spike16']:04X}"
                        if debug and debug.get('challenge') is not None: challenge_hex = f"0x{debug['challenge']:04X}"
                        elif parsed and parsed.get('challenge') is not None: challenge_hex = f"0x{parsed['challenge']:04X}"
                        if debug and debug.get('seed16') is not None: seed16_hex = f"0x{debug['seed16']:04X}"
                    else:
                        if parsed and parsed.get('spike16') is not None: spike16_hex = f"0x{parsed['spike16']:04X}"
                        elif debug and debug.get('spike16') is not None: spike16_hex = f"0x{debug['spike16']:04X}"
                        if parsed and parsed.get('challenge') is not None: challenge_hex = f"0x{parsed['challenge']:04X}"
                        elif debug and debug.get('challenge') is not None: challenge_hex = f"0x{debug['challenge']:04X}"
                        if parsed and parsed.get('thr16') is not None:
                            thr16_hex = f"0x{parsed['thr16']:04X}"
                    if parsed:
                        if parsed.get('thr16') is not None: thr16_hex = f"0x{parsed['thr16']:04X}"
                        thr8_hex = f"0x{(parsed.get('thr16') or 0)&0xFF:02X}" if parsed.get('thr16') is not None else ""
                        thr_idx_val = parsed.get('thr_idx') if parsed.get('thr_idx') is not None else ""
                        maj = parsed.get('majresp')
                        auth = parsed.get('auth')
                    if thr_idx_val is None: thr_idx_val = ""
                    if maj is None: maj = ""
                    if auth is None: auth = 0
                    effective_auth = auth
                    if FORCE_AUTH_IF_SPIKE_HIGH and debug and debug.get('spike16') and debug.get('spike16') > SPIKE_OVERRIDE_THRESH:
                        effective_auth = 1
                    tx_hex = pkt.hex() if pkt else ""
                    raw_parsed_hex = parsed.get('raw_hex') if parsed else rx_hex
                    debug_raw_hex = debug.get('raw_hex') if debug else ""
                    print("[PERIODIC] TX", tx_hex, "RX", rx_hex, "PARSED", parsed, "DEBUG", debug, "EFFECTIVE_AUTH", effective_auth)
                    safe_parsed = None
                    if parsed:
                        safe_parsed = {}
                        for k,v in parsed.items():
                            if isinstance(v, (bytes,bytearray)): safe_parsed[k] = v.hex()
                            else:
                                try:
                                    json.dumps(v)
                                    safe_parsed[k] = v
                                except:
                                    safe_parsed[k] = str(v)
                    csvw.writerow([
                        datetime.now().isoformat(), "periodic", frame_id, "", int(bool(recognized)), f"{best_sim:.4f}" if best_sim is not None else "",
                        best_lbl if best_lbl is not None else "", rec_name if rec_name is not None else "",
                        pid, f"0x{face16_val:04X}", spike16_hex, seed16_hex, thr16_hex, thr8_hex, thr_idx_val, challenge_hex,
                        maj, effective_auth, tx_hex, rx_hex, raw_parsed_hex, debug_raw_hex, parsed.get('format') if parsed else "",
                        json.dumps(safe_parsed) if safe_parsed else "", "", ""
                    ])
                    csvf.flush()

            # Normal grant/send logic
            if granted and last_granted != best_lbl:
                now_send = time.time()
                if now_send - last_send_time < SEND_COOLDOWN_S:
                    print("[throttle] skipped send (cooldown)")
                else:
                    try:
                        pid = int(best_lbl) if (best_lbl is not None and str(best_lbl).isdigit()) else (abs(hash(str(best_lbl))) & 0x0F)
                    except:
                        pid = (abs(hash(str(best_lbl))) & 0x0F)
                    face16_val = int(face16) & 0xFFFF
                    pkt, rx = send_packet(ser, pid, face16_val) if ser else (None, b"")
                    rx_hex = rx.hex() if rx else ""
                    parsed = parse_reply(rx)
                    debug = parse_debug_packet(rx)
                    spike16_hex = ""; challenge_hex = ""; seed16_hex = ""; thr16_hex=""; thr8_hex=""; thr_idx_val=""; maj=""; auth=None
                    if PREFER_DEBUG:
                        if debug and debug.get('spike16') is not None: spike16_hex = f"0x{debug['spike16']:04X}"
                        elif parsed and parsed.get('spike16') is not None: spike16_hex = f"0x{parsed['spike16']:04X}"
                        if debug and debug.get('challenge') is not None: challenge_hex = f"0x{debug['challenge']:04X}"
                        elif parsed and parsed.get('challenge') is not None: challenge_hex = f"0x{parsed['challenge']:04X}"
                        if debug and debug.get('seed16') is not None: seed16_hex = f"0x{debug['seed16']:04X}"
                    else:
                        if parsed and parsed.get('spike16') is not None: spike16_hex = f"0x{parsed['spike16']:04X}"
                        elif debug and debug.get('spike16') is not None: spike16_hex = f"0x{debug['spike16']:04X}"
                        if parsed and parsed.get('challenge') is not None: challenge_hex = f"0x{parsed['challenge']:04X}"
                        elif debug and debug.get('challenge') is not None: challenge_hex = f"0x{debug['challenge']:04X}"
                        if parsed and parsed.get('thr16') is not None:
                            thr16_hex = f"0x{parsed['thr16']:04X}"
                    if parsed:
                        if parsed.get('thr16') is not None: thr16_hex = f"0x{parsed['thr16']:04X}"
                        thr8_hex = f"0x{(parsed.get('thr16') or 0)&0xFF:02X}" if parsed.get('thr16') is not None else ""
                        thr_idx_val = parsed.get('thr_idx')
                        maj = parsed.get('majresp')
                        auth = parsed.get('auth')
                    if thr_idx_val is None: thr_idx_val = ""
                    if maj is None: maj = ""
                    if auth is None: auth = 0
                    effective_auth = auth
                    if FORCE_AUTH_IF_SPIKE_HIGH and debug and debug.get('spike16') and debug.get('spike16') > SPIKE_OVERRIDE_THRESH:
                        effective_auth = 1
                    tx_hex = pkt.hex() if pkt else ""
                    raw_parsed_hex = parsed.get('raw_hex') if parsed else rx_hex
                    debug_raw_hex = debug.get('raw_hex') if debug else ""
                    print("[UART] TX", tx_hex, "RX", rx_hex, "PARSED", parsed, "DEBUG", debug, "EFFECTIVE_AUTH", effective_auth)
                    safe_parsed = None
                    if parsed:
                        safe_parsed = {}
                        for k,v in parsed.items():
                            if isinstance(v, (bytes,bytearray)): safe_parsed[k] = v.hex()
                            else:
                                try:
                                    json.dumps(v)
                                    safe_parsed[k] = v
                                except:
                                    safe_parsed[k] = str(v)
                    csvw.writerow([
                        datetime.now().isoformat(), "live", frame_id, "", int(bool(recognized)), f"{best_sim:.4f}" if best_sim is not None else "",
                        best_lbl if best_lbl is not None else "", rec_name if rec_name is not None else "",
                        pid, f"0x{face16_val:04X}", spike16_hex, seed16_hex, thr16_hex, thr8_hex, thr_idx_val, challenge_hex,
                        maj, effective_auth, tx_hex, rx_hex, raw_parsed_hex, debug_raw_hex, parsed.get('format') if parsed else "",
                        json.dumps(safe_parsed) if safe_parsed else "", "", ""
                    ])
                    csvf.flush()
                    last_send_time = now_send
                    if effective_auth == 1:
                        print("[ACCESS] GRANTED")
                        if SEND_UART_ON_GRANT and ser:
                            try: ser.write(UART_CONTROL_PACKET)
                            except: pass
                    else:
                        print("[ACCESS] DENIED")
                last_granted = best_lbl

            k = cv2.waitKey(10) & 0xFF
            if k == ord('q'): break
            if k == ord('e'):
                # simple enrollment flow (save frame & face crop)
                if not TK_OK:
                    print("Enter enroll label: ", end='', flush=True)
                    try: label = input().strip()
                    except: label = None
                else:
                    # use tkinter if available
                    try:
                        import tkinter as tk
                        from tkinter import simpledialog
                        root = tk.Tk(); root.withdraw()
                        label = simpledialog.askstring("Enroll", "Enter enroll label (e.g. 1 or user1):", parent=root)
                        root.destroy()
                    except Exception:
                        label = None
                if not label:
                    print("Enroll cancelled")
                else:
                    dest_dir = Path(args.data_dir)/"enroll"/label; dest_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = str(dest_dir / f"enroll_{ts}.jpg")
                    face_path = str(dest_dir / f"face_{ts}.jpg")
                    try:
                        cv2.imwrite(img_path, frame)
                        if face_bgr is not None: cv2.imwrite(face_path, face_bgr)
                        else:
                            h,w = frame.shape[:2]; cx,cy = w//2,h//2; s = min(w,h)//4
                            crop = frame[max(0,cy-s):min(h,cy+s), max(0,cx-s):min(w,cx+s)]
                            cv2.imwrite(face_path, crop)
                        print("[enroll] saved", img_path, face_path)
                        if embedder and embedder.enabled:
                            emb, fb = embedder.embed_image_path(img_path)
                        else:
                            emb, fb = None, None
                        f16val = lbp_checksum16_from_bgr(fb) if fb is not None else 0
                        enroll_map.setdefault(label, []).append((emb, f16val, img_path))
                        # recompute centroids
                        centroids.clear(); face16_mean.clear(); avg_sim.clear()
                        for lbl, items in enroll_map.items():
                            embs = [e for (e,f,p) in items if e is not None]
                            if not embs: continue
                            embs = np.vstack(embs); cent = np.mean(embs, axis=0); cent = cent / (np.linalg.norm(cent)+1e-9)
                            centroids[lbl] = cent
                            face16_mean[lbl] = int(round(np.mean([f for (e,f,p) in items])))
                            sims = [cos_sim(e, cent) for (e,f,p) in items if e is not None]
                            avg_sim[lbl] = float(np.mean(sims)) if sims else 0.8
                        per_user_thr.update({lbl: max(SIM_THRESHOLD_MIN, avg_sim.get(lbl,0.75)-SIM_MARGIN) for lbl in centroids.keys()})
                        global_thr = float(np.mean(list(per_user_thr.values()))) if per_user_thr else SIM_THRESHOLD_MIN
                        print("[enroll] recomputed centroids:", list(centroids.keys()), "global_thr=", global_thr)
                    except Exception as e:
                        print("Enroll save error:", e)
            if k == ord(' '):
                # manual send
                if best_lbl is None:
                    print("[manual send] no recognized label; not sending")
                else:
                    try:
                        pid = int(best_lbl) if str(best_lbl).isdigit() else (abs(hash(str(best_lbl))) & 0x0F)
                    except:
                        pid = (abs(hash(str(best_lbl))) & 0x0F)
                    face16_val = int(face16) & 0xFFFF
                    print(f"[manual send] pid={pid} label={best_lbl} face16=0x{face16_val:04X}")
                    pkt, rx = send_packet(ser, pid, face16_val) if ser else (None, b"")
                    print("manual send rx:", rx.hex() if rx else "")
    finally:
        cap.release(); csvf.close()
        if ser:
            try: ser.close()
            except: pass
        cv2.destroyAllWindows()

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["live"], default="live")
    p.add_argument("--data_dir", default=DATA_DIR)
    p.add_argument("--serial_port", default=None)
    p.add_argument("--serial_baud", type=int, default=SERIAL_BAUD)
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--consistent_frames", type=int, default=CONSISTENT_FRAMES)
    args = p.parse_args()
    if args.mode == "live":
        live_mode(args)

if __name__ == "__main__":
    main()
