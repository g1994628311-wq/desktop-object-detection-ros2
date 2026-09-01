#!/usr/bin/env python3
"""Materialize the manually reviewed P02 capture sessions and V3 annotations."""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/P02"
V3 = ROOT / "data/v3"
NAMES = ("laptop", "keyboard", "cup")
COLORS = ((0, 220, 255), (255, 210, 0), (255, 60, 180))

# Boxes are manual normalized xyxy annotations made from direct visual inspection.
# Each tuple is (class_id, x1, y1, x2, y2).
RECORDS = [
    ("063edffc766c9c88e3e6052307d47ee8.jpg", "S02", []),
    ("06828ef342c72bd8cc1594e18971dc07.jpg", "S01", [(0,.18,.03,.82,.99)]),
    ("0a1b790dfc132f4e719620a4343ae757.jpg", "S01", [(1,.10,.34,.98,.65)]),
    ("13e05118d087818a00ecbc0721d53dd9.jpg", "S02", [(2,.27,.13,.78,.91)]),
    ("1a032d554b686d34efeb246b4a20533b.jpg", "S02", [(0,.00,.02,1.00,.98),(2,.68,.29,.99,.60)]),
    ("1b5eace105eb980d51b055add2163356.jpg", "S02", [(0,.00,.00,1.00,1.00),(1,.00,.34,.99,.73)]),
    ("1c64dd6e0321c3e521652bf2f8597b67.jpg", "S02", [(0,.00,.01,1.00,.94)]),
    ("201afe31a2011ea273dbb371f55f283a.jpg", "S02", []),
    ("22ec2152ad1c7bc3a117f10609e5c11f.jpg", "S02", [(0,.00,.00,1.00,.92)]),
    ("2dc45f19034d304028dc06e3c540fbab.jpg", "S02", [(2,.33,.13,.91,.63)]),
    ("2e863fcfb00ff099457402654c33dd62.jpg", "S02", [(1,.02,.08,.99,.93)]),
    ("3b04be77ca5764cf55f1567130014bf6.jpg", "S02", [(1,.25,.03,.76,.97)]),
    ("40bb7946a44c355dcf9ab84dd2d95bc2.jpg", "S01", [(0,.00,.05,.93,.99)]),
    ("44d1fb7431b8be7d34a4e6602f26e899.jpg", "S02", [(2,.39,.03,.78,.97)]),
    ("492aec4eb8d6408158b603642fcc6341.jpg", "S02", [(0,.78,.00,1.00,.99),(1,.08,.00,.58,.82),(2,.00,.63,.45,1.00)]),
    ("50ed179a8b30dbf60f2139d778456807.jpg", "S02", [(2,.48,.02,.99,.86)]),
    ("5610d1f35786a278343ffb3d03085e4c.jpg", "S02", [(1,.20,.00,.77,.90)]),
    ("569c29a100ac20c7f21b15726211c15a.jpg", "S02", [(0,.00,.00,1.00,1.00)]),
    ("56fd26338fe813d13144a6be998aed45.jpg", "S02", [(0,.90,.02,1.00,.99),(2,.06,.08,.85,.91)]),
    ("5fbdcfb1510bb64c0c4d9cb8bf1ee9fe.jpg", "S02", [(1,.03,.06,.99,.89)]),
    ("62586da0905e055af454a59546426067.jpg", "S02", []),
    ("63e9090af1ccfd91aee0021d9b240f4d.jpg", "S02", [(0,.02,.00,.91,.99),(1,.00,.42,.94,1.00)]),
    ("64d2e4ebd2c8aaac49f046128b822c13.jpg", "S03", [(1,.12,.23,.94,.80)]),
    ("6c500cf88c1ca8d0b81f091c488592b0.jpg", "S02", [(1,.38,.00,.99,.40)]),
    ("6e574b5bc7755b57bd02e284add340df.jpg", "S02", [(0,.01,.01,.98,.99)]),
    ("7a9f7fc241e02bc9a29be72acb13d018.jpg", "S02", [(0,.04,.02,.96,.98)]),
    ("7d4576ef888f12f6af6c304b7996c233.jpg", "S02", [(0,.00,.00,1.00,1.00),(1,.00,.27,.99,.67)]),
    ("85d07550dab016dd3f8ed9189c0e1466.jpg", "S04", [(2,.23,.10,.81,.91)]),
    ("86cc4af0e8c0ad7ba39359d2e6d4e823.jpg", "S01", [(2,.31,.02,.76,.97)]),
    ("8bbb3eff2d6ab9a3c313df5080ac7fc9.jpg", "S02", [(1,.06,.33,.96,.99)]),
    ("906b7bbdef3122cac13ab2409de0b3d4.jpg", "S02", [(1,.14,.25,.88,.87)]),
    ("919b1a40b930945d9a144dd9dceaa984.jpg", "S03", [(2,.35,.25,.71,.78)]),
    ("9c2beb8d6b9ef24fcf038c1f88fb7499.jpg", "S02", [(0,.33,.20,.76,.69),(2,.13,.02,.41,.99)]),
    ("9fd0bd9ddcea1b4a21c3d235d4219f72.jpg", "S03", []),
    ("a5010fb92c045407bdc48bfcfc04948f.jpg", "S04", [(2,.25,.08,.76,.91)]),
    ("a6899db1ed5857d4d8d75baef90f6c01.jpg", "S01", [(2,.31,.03,.72,.97)]),
    ("a6e33c8f6359df5acb47717c750be6b3.jpg", "S03", []),
    ("aa38b37a8ce1d3521eb813acc93e3f8b.jpg", "S02", [(1,.03,.02,.99,.99)]),
    ("ab98064a9255f9401f27c56f7c4089c6.jpg", "S01", [(1,.08,.38,.99,.79)]),
    ("ae8774c2184a0c941ec2ad6bd0d66825.jpg", "S05", []),
    ("b2ac8aaf85689f636b704fd58d895c87.jpg", "S02", [(0,.03,.01,.98,.99)]),
    ("b7286c5024db3980a9cb6b155aa5630f.jpg", "S02", [(0,.07,.01,.93,.99)]),
    ("b96d4d51a2263a3b489afd8367bb0c21.jpg", "S02", [(0,.00,.00,.99,.48),(0,.00,.49,.99,1.00),(2,.00,.16,.30,.92)]),
    ("b97135205f43ece509a6442a70cad1b5.jpg", "S04", [(0,.08,.02,.92,.99)]),
    ("bc1dc77d2be7873ae3082c04f2f94490.jpg", "S02", [(0,.00,.02,1.00,.95)]),
    ("bc75f08443cfd14d7489aff4b3ee4cfc.jpg", "S01", [(0,.19,.02,.82,.99)]),
    ("bd22a715288e45afb82bc4a9499f3b78.jpg", "S02", []),
    ("c3f44dd20bfddc58deab0221257e5612.jpg", "S03", []),
    ("c4390e296a64f33e51fe7720361220f4.jpg", "S04", [(0,.05,.06,.94,.91)]),
    ("c6d613f40222cf7e8dbc2d98ea367a16.jpg", "S02", [(1,.02,.04,.99,.87)]),
    ("cabde072fddb68a713038b53424fbf00.jpg", "S02", [(0,.02,.00,.97,.99)]),
    ("cd6e92d3fdc47b1bd40b33fcbc0f4cf0.jpg", "S03", []),
    ("cef117600ae4b7800558361a1fdacd03.jpg", "S02", []),
    ("cf11969d4cb065d98d33258a7f721462.jpg", "S04", [(0,.06,.03,.94,.98)]),
    ("cf5f358af4399e58684a57c78995dbf7.jpg", "S01", [(2,.31,.02,.74,.98)]),
    ("cfbf7b5bf3d95aa2fdd23b86327c12dd.jpg", "S02", [(0,.05,.02,.94,.98)]),
    ("d4ef61170b3e1b0a02a33ddb873c0895.jpg", "S02", [(0,.00,.02,1.00,.95)]),
    ("d70e274f373398831e26b469f15ef636.jpg", "S02", [(2,.00,.03,.79,.97)]),
    ("dc900062f1ae5b0fc1ef13f5e342e7fa.jpg", "S01", [(1,.10,.35,.99,.66)]),
    ("e2a142a60e5a1f840ce99bf1cab8108f.jpg", "S02", []),
    ("e5e4ae8017db486a038775b4a12f6345.jpg", "S02", [(0,.00,.00,.75,.99),(0,.72,.00,1.00,.99),(2,.00,.00,.52,.75)]),
    ("e8fec9115dcce42788401eafc9c5b0e6.jpg", "S02", [(1,.25,.02,.75,.98)]),
    ("ece4d0dacebbadd1489b0ca5e142dc37.jpg", "S02", [(0,.00,.02,1.00,.94)]),
    ("ed33e5668f6dcd4468bc4c0cf1be1aad.jpg", "S01", [(2,.30,.02,.73,.98)]),
    ("fa9bd1435373a25c45d82ae790ed009e.jpg", "S02", [(2,.05,.03,.77,.98)]),
    ("fd129547b2329bd56a74ab8a88dc7542.jpg", "S02", [(0,.18,.00,.79,.99),(0,.79,.00,1.00,.99),(1,.00,.25,.67,.99),(2,.00,.00,.46,.70)]),
]

SCENES = {
    "S01": ("light_lab_desk", "light laboratory desk, Dell monitor, shared MacBook/full keyboard/black thermos"),
    "S02": ("wood_desk_bookshelf", "wood-grain desk and bookshelf, shared laptops/compact keyboard/cups"),
    "S03": ("tile_floor", "tiled floor capture with keyboard, mug, and hard negatives"),
    "S04": ("low_light_room", "distinct low-light room and dark wood/bed surface"),
    "S05": ("window_folding_table", "window-side folding table hard-negative capture"),
}

INSTANCE_IDS = {
    "S01": {0:"p02_laptop_macbook_01", 1:"p02_keyboard_fullsize_01", 2:"p02_cup_black_thermos_01"},
    "S02": {0:"p02_laptop_macbook_01|p02_laptop_lenovo_01", 1:"p02_keyboard_compact_01", 2:"p02_cup_green_mug_01|p02_cup_black_thermos_01"},
    "S03": {1:"p02_keyboard_compact_01", 2:"p02_cup_green_mug_01"},
    "S04": {0:"p02_laptop_macbook_01", 2:"p02_cup_green_mug_01"},
    "S05": {},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_type(boxes):
    classes = sorted({b[0] for b in boxes})
    return "NEG" if not classes else ("MIX" if len(classes) > 1 else ("LAP", "KEY", "CUP")[classes[0]])


def draw_preview(src: Path, dst: Path, boxes):
    with Image.open(src) as opened:
        im = ImageOps.exif_transpose(opened).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(14, round(min(w, h) / 45)))
    line = max(3, round(min(w, h) / 240))
    for cls, x1, y1, x2, y2 in boxes:
        xy = (round(x1*w), round(y1*h), round(x2*w), round(y2*h))
        draw.rectangle(xy, outline=COLORS[cls], width=line)
        label = NAMES[cls]
        bb = draw.textbbox((xy[0], xy[1]), label, font=font, stroke_width=2)
        draw.rectangle((bb[0]-3, bb[1]-2, bb[2]+3, bb[3]+2), fill=(0,0,0))
        draw.text((xy[0], xy[1]), label, fill=COLORS[cls], font=font, stroke_width=1, stroke_fill=(0,0,0))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=92)


def make_contact(paths, dst, title):
    thumb_w, thumb_h = 320, 250
    cols = 4
    rows = (len(paths)+cols-1)//cols
    sheet = Image.new("RGB", (cols*thumb_w, 50+rows*thumb_h), (28,28,28))
    draw = ImageDraw.Draw(sheet)
    draw.text((12,12), title, fill="white", font=ImageFont.load_default(size=22))
    for i, path in enumerate(paths):
        with Image.open(path) as opened: im = ImageOps.exif_transpose(opened).convert("RGB")
        im.thumbnail((thumb_w-10, thumb_h-35))
        x=(i%cols)*thumb_w+(thumb_w-im.width)//2; y=50+(i//cols)*thumb_h
        sheet.paste(im,(x,y)); draw.text(((i%cols)*thumb_w+6,y+thumb_h-30),path.stem,fill="white")
    dst.parent.mkdir(parents=True, exist_ok=True); sheet.save(dst,quality=90)


def main():
    raw_files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
    if len(raw_files) != 66 or {p.name for p in raw_files} != {r[0] for r in RECORDS}:
        raise SystemExit("P02 raw inventory differs from reviewed 66-image inventory")
    before = {p.name: sha256(p) for p in raw_files}
    for p in raw_files:
        with Image.open(p) as im: im.verify()
    if len(set(before.values())) != len(before): raise SystemExit("exact duplicate raw images found")

    for base in (V3/"user/images/P02", V3/"user/labels/P02", V3/"user/previews/P02"):
        if base.exists(): shutil.rmtree(base)
    counters=Counter(); rows=[]; session_previews=defaultdict(list)
    for raw_name, session, boxes in RECORDS:
        raw=RAW/raw_name; typ=image_type(boxes); counters[(session,typ)]+=1
        stem=f"P02_{session}_{typ}_{counters[(session,typ)]:04d}"
        image=V3/f"user/images/P02/{session}/{stem}{raw.suffix.lower()}"
        label=V3/f"user/labels/P02/{session}/{stem}.txt"
        preview=V3/f"user/previews/P02/{session}/{stem}.jpg"
        image.parent.mkdir(parents=True,exist_ok=True); label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(raw,image)
        lines=[]
        for cls,x1,y1,x2,y2 in boxes:
            lines.append(f"{cls} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
        label.write_text("\n".join(lines)+("\n" if lines else ""),encoding="utf-8")
        draw_preview(image,preview,boxes); session_previews[session].append(preview)
        counts=Counter(b[0] for b in boxes); present="|".join(NAMES[c] for c in sorted(counts))
        ids=[]
        for cls in sorted(counts):
            candidates=INSTANCE_IDS[session].get(cls,"").split("|")
            if counts[cls] == 1:
                # Choose the visually applicable item where a session has two candidates.
                if cls==0 and session=="S02":
                    ids.append("p02_laptop_lenovo_01" if raw_name in {"1b5eace105eb980d51b055add2163356.jpg","1c64dd6e0321c3e521652bf2f8597b67.jpg","22ec2152ad1c7bc3a117f10609e5c11f.jpg","7d4576ef888f12f6af6c304b7996c233.jpg","9c2beb8d6b9ef24fcf038c1f88fb7499.jpg","b7286c5024db3980a9cb6b155aa5630f.jpg","bc1dc77d2be7873ae3082c04f2f94490.jpg","d4ef61170b3e1b0a02a33ddb873c0895.jpg","ece4d0dacebbadd1489b0ca5e142dc37.jpg"} else "p02_laptop_macbook_01")
                elif cls==2 and session=="S02": ids.append("p02_cup_black_thermos_01" if raw_name in {"44d1fb7431b8be7d34a4e6602f26e899.jpg","492aec4eb8d6408158b603642fcc6341.jpg","9c2beb8d6b9ef24fcf038c1f88fb7499.jpg","d70e274f373398831e26b469f15ef636.jpg"} else "p02_cup_green_mug_01")
                else: ids.extend(candidates[:1])
            else: ids.extend(candidates[:counts[cls]])
        scene,reason=SCENES[session]
        rel=lambda p:p.relative_to(ROOT).as_posix()
        rows.append({"source":"P02_user","raw_path":rel(raw),"original_path":rel(raw),"original_session":"unassigned","v2_session":"","capture_session":session,"scene_id":scene,"v3_filename":image.name,"v3_path":rel(image),"image_path":rel(image),"label_path":rel(label),"preview_path":rel(preview),"image_type":typ,"v3_type":typ,"target_classes_present":present,"scenario":"hard_negative" if not boxes else ("multi_target" if typ=="MIX" else "basic"),"object_instance_ids":"|".join(ids),"status":"included","laptop_instances":counts[0],"keyboard_instances":counts[1],"cup_instances":counts[2],"sha256":before[raw_name],"annotation_source":"manual_visual_P02_from_scratch","review_status":"second_visual_review_approved"})
    for session,paths in session_previews.items():
        make_contact(paths,V3/f"user/previews/session_contact_sheets/P02_{session}.jpg",f"P02 {session} - {SCENES[session][0]}")

    manifest=V3/"manifests/user_image_manifest.csv"
    with manifest.open(encoding="utf-8-sig",newline="") as f: old=[r for r in csv.DictReader(f) if r["source"]!="P02_user"]; fields=f.seek(0) or next(csv.reader(f))
    # Re-read the stable header because DictReader consumed the stream.
    fields=list(rows[0].keys()) if not old else list(old[0].keys())
    with manifest.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(old+rows)

    summary=V3/"manifests/p02_session_summary.csv"
    sf=["session","images","laptop_instances","keyboard_instances","cup_instances","negative_images","unique_laptop_instances","unique_keyboard_instances","unique_cup_instances","scene_description"]
    with summary.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sf); w.writeheader()
        for session in SCENES:
            rs=[r for r in rows if r["capture_session"]==session]
            uniq={n:set() for n in NAMES}
            for r in rs:
                for ident in r["object_instance_ids"].split("|"):
                    for n in NAMES:
                        if f"_{n}_" in ident: uniq[n].add(ident)
            w.writerow({"session":f"P02_{session}","images":len(rs),"laptop_instances":sum(int(r["laptop_instances"]) for r in rs),"keyboard_instances":sum(int(r["keyboard_instances"]) for r in rs),"cup_instances":sum(int(r["cup_instances"]) for r in rs),"negative_images":sum(r["v3_type"]=="NEG" for r in rs),"unique_laptop_instances":len(uniq["laptop"]),"unique_keyboard_instances":len(uniq["keyboard"]),"unique_cup_instances":len(uniq["cup"]),"scene_description":SCENES[session][1]})

    ds=V3/"manifests/dataset_summary.csv"
    with ds.open(encoding="utf-8-sig",newline="") as f: dsrows=list(csv.DictReader(f)); dsfields=f.seek(0) or []
    dsfields=["section","images","positive_images","negative_images","objects","laptop_instances","keyboard_instances","cup_instances","user_images","coco_images","unique_capture_sessions","unique_physical_instances"]
    p01=[r for r in old if r["source"]=="P01_user"]
    combined=p01+rows
    def contributor_stats(section, records):
        physical={i for r in records for i in re.split(r"[;|]",r["object_instance_ids"]) if i and any(i.startswith(f"{n}_") or f"_{n}_" in i for n in NAMES)}
        result={"section":section,"images":len(records),"positive_images":sum(r["v3_type"]!="NEG" for r in records),"negative_images":sum(r["v3_type"]=="NEG" for r in records),"laptop_instances":sum(int(r["laptop_instances"]) for r in records),"keyboard_instances":sum(int(r["keyboard_instances"]) for r in records),"cup_instances":sum(int(r["cup_instances"]) for r in records),"user_images":len(records),"coco_images":0,"unique_capture_sessions":len({(r["source"],r["capture_session"]) for r in records}),"unique_physical_instances":len(physical)}
        result["objects"]=sum(int(result[f"{n}_instances"]) for n in NAMES)
        return result
    replacements={s["section"]:s for s in (contributor_stats("P01_USER",p01),contributor_stats("P02_USER",rows),contributor_stats("USER",combined))}
    dsrows=[r for r in dsrows if r["section"] not in {"P01_USER","P02_USER"}]
    for row in dsrows:
        if row["section"]=="USER": row.update({k:str(v) for k,v in replacements["USER"].items()})
    user_index=next(i for i,r in enumerate(dsrows) if r["section"]=="USER")
    dsrows[user_index:user_index]=[{k:str(v) for k,v in replacements["P01_USER"].items()},{k:str(v) for k,v in replacements["P02_USER"].items()}]
    with ds.open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=dsfields); w.writeheader(); w.writerows(dsrows)

    after={p.name:sha256(p) for p in raw_files}
    if before!=after: raise SystemExit("raw P02 bytes changed")
    print(f"P02 images={len(rows)} positives={sum(r['v3_type']!='NEG' for r in rows)} negatives={sum(r['v3_type']=='NEG' for r in rows)}")
    print("objects",{n:sum(int(r[f'{n}_instances']) for r in rows) for n in NAMES})

if __name__ == "__main__": main()
