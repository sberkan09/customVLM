# src/utils/build_dataset.py
from datasets import load_dataset
from PIL import Image
from io import BytesIO
import os, json, time, random
import requests
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

# Thread-local session (keep-alive + retry)
_tls = {}

def get_session():
    sid = id(os.getpid())  # process id is enough for this script
    s = _tls.get(sid)
    if s is None:
        s = requests.Session()
        retries = Retry(
            total=5, connect=5, read=5, backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _tls[sid] = s
    return s

def fetch_image(url, timeout=10):
    s = get_session()
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

def _download_one(eg, out_images, split, idx, quality=95):
    """
    Tek örneği indirip diske yazar.
    Başarılıysa (image_path, caption) döner, yoksa None.
    """
    url = eg["url"]
    cap = eg["sentences"][0]
    path = os.path.join(out_images, f"{split}_{idx:07d}.jpg")

    # Resume: dosya varsa tekrar indirme
    if os.path.exists(path):
        return {"image_path": path, "caption": cap}

    try:
        img = fetch_image(url)
        img.save(path, quality=quality)
        return {"image_path": path, "caption": cap}
    except Exception as e:
        # Basit throttle: yoğun hata varsa azıcık bekle
        time.sleep(0.2 + random.random() * 0.3)
        return None

def build_coco_subset(root="data/coco-mini", split="train", pct="2%", workers=16):
    os.makedirs(root, exist_ok=True)

    # Dataset (Karpathy split; train/validation/test/restval mevcut)
    hf_split = split if split != "validation" else "validation"
    ds = load_dataset("yerevann/coco-karpathy", split=f"{hf_split}[:{pct}]")

    out_images = os.path.join(root, f"{split}_imgs")
    os.makedirs(out_images, exist_ok=True)
    index_path = os.path.join(root, f"{split}.jsonl")

    # Paralel indirme
    jobs = []
    results = []
    download_fn = partial(_download_one, out_images=out_images, split=split)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, eg in enumerate(ds):
            jobs.append(ex.submit(download_fn, eg, idx=i))

        for fut in tqdm(as_completed(jobs), total=len(jobs), desc=f"downloading {split}"):
            rec = fut.result()
            if rec is not None:
                results.append(rec)

    # Yaz (tek seferde)
    with open(index_path, "w", encoding="utf-8") as fw:
        for rec in results:
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote: {index_path} (ok={len(results)}/{len(ds)})")

if __name__ == "__main__":
    # İstersen workers sayısını makinene göre artır/azalt (8-32 arası tipik)
    build_coco_subset(split="train", pct="2%", workers=16)
    build_coco_subset(split="validation", pct="2%", workers=16)
