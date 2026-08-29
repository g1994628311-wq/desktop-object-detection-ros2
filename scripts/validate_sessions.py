#!/usr/bin/env python3
"""Validate V2 split coverage, duplicate assignment, and capture-group leakage."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; V2=ROOT/'data/v2'
rows=list(csv.DictReader((V2/'manifests/image_manifest.csv').open(encoding='utf-8')))
expected={f"data/v2/images/P01/{r['new_session']}/{r['new_filename']}" for r in rows}; owner={}; errors=[]
for split in ('train','val','test'):
 for line in (V2/'splits'/f'{split}.txt').read_text(encoding='utf-8').splitlines():
  if line in owner: errors.append('duplicate assignment: '+line)
  owner[line]=split
if set(owner)!=expected: errors.append(f"coverage mismatch missing={len(expected-set(owner))} extra={len(set(owner)-expected)}")
for session in sorted({r['new_session'] for r in rows}):
 assigned={owner.get(f"data/v2/images/P01/{r['new_session']}/{r['new_filename']}") for r in rows if r['new_session']==session}
 if len(assigned)!=1: errors.append(f'capture-group leakage {session}: {assigned}')
print(f'images={len(rows)} sessions={len(set(r["new_session"] for r in rows))} errors={len(errors)}')
for e in errors: print('ERROR:',e)
raise SystemExit(bool(errors))
