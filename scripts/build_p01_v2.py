#!/usr/bin/env python3
"""Rebuild P01 V2 from raw images after visual capture-session audit.

No detector is invoked.  Existing V1 labels are visually reused; S06 labels are
hand-transcribed during direct visual review.
"""
from __future__ import annotations
import csv, shutil
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw/P01'; V1=ROOT/'data/v1/labels/P01'; V2=ROOT/'data/v2'
EXCLUDED_RAW={'data/raw/P01/S06/152bd349e15c850ec32b6c0c64ad0305.jpg'}
NAMES=('mouse','keyboard','laptop','cup','headphones'); COLORS=[(0,220,255),(255,180,0),(0,255,80),(255,70,70),(190,80,255)]
# These source images form one light laboratory-table capture sequence.  All
# other pre-S06 P01 photos form the continuous curved-wood-desk capture group.
LAB={
'S01/P01_S01_CUP_0001.jpg','S01/P01_S01_CUP_0002.jpg','S01/P01_S01_CUP_0003.jpg','S01/P01_S01_CUP_0004.jpg','S01/P01_S01_LAP_0005.jpg','S01/P01_S01_LAP_0006.jpg','S01/P01_S01_LAP_0007.jpg','S01/P01_S01_MIX_0008.jpg','S01/P01_S01_MIX_0009.jpg','S01/P01_S01_MIX_0010.jpg','S01/P01_S01_MOU_0011.jpg',
'S02/P01_S02_CUP_0001.jpg','S02/P01_S02_CUP_0002.jpg','S02/P01_S02_CUP_0007.jpg','S02/P01_S02_LAP_0003.jpg','S02/P01_S02_LAP_0004.jpg','S02/P01_S02_LAP_0005.jpg','S02/P01_S02_LAP_0006.jpg','S02/P01_S02_MIX_0008.jpg','S02/P01_S02_MIX_0009.jpg','S02/P01_S02_MIX_0010.jpg','S02/P01_S02_MIX_0019.jpg','S02/P01_S02_MIX_0020.jpg','S02/P01_S02_MOU_0011.jpg','S02/P01_S02_MOU_0012.jpg',
'S03/P01_S03_CUP_0020.jpg','S03/P01_S03_CUP_0021.jpg','S03/P01_S03_MIX_0023.jpg'}
FLOOR={'070ec69193ffcf6ef778f5053b3ff55a.jpg','182866919097fea3596f20d6095cb60f.jpg','2347c0379a4047b4edd106386320d3cf.jpg','25720402ad40ca28bc072ec1fd5c8acd.jpg','262caa9c7a97ec0ab3f623c8460ac6e2.jpg','30369bb8428bcc5532ee4084c7f8d5ad.jpg','63e77d00ff7615130df5b6e0c90c7e91.jpg','c347d893b6e2351548edad6f2f18bacd.jpg'}
DOLL={'152bd349e15c850ec32b6c0c64ad0305.jpg','1c13548f0cf6b47b964126751e53105f.jpg','1c8abf840dc70b2b3687bde1ceb38f23.jpg','1e0600e7c44ad48876c05e9125ad2520.jpg','6003090d371cf667f7fd64bc5910d414.jpg','e6fa3360513e8b78544b1c302d74edef.jpg'}
# class, x1, y1, x2, y2 in original pixels.  Every old-S06 image was
# re-annotated by direct visual inspection; no detector or old S06 box is used.
S06_PX={
'070ec69193ffcf6ef778f5053b3ff55a.jpg':[(4,680,270,1070,640),(2,455,610,1260,1130)],
'07a4fcdb46ba0f699f835d4c362f798f.jpg':[(1,0,0,735,420),(4,0,465,380,920),(3,665,285,1080,920)],
'0c4091061b3f911528c74ee8a0a13813.jpg':[(4,360,125,1185,700),(3,230,755,710,1110)],
'152bd349e15c850ec32b6c0c64ad0305.jpg':[(0,400,875,605,1100),(3,170,650,325,805),(3,1010,720,1130,890),(3,1100,850,1210,1010),(4,1100,470,1435,1010)],
'15c1d618223cab16abf21d30acedcfb7.jpg':[(3,380,495,950,1000)],
'182866919097fea3596f20d6095cb60f.jpg':[(2,465,295,1125,790),(4,545,835,940,1210)],
'1c13548f0cf6b47b964126751e53105f.jpg':[(2,100,280,1150,710),(3,1380,480,1645,760)],
'1c8abf840dc70b2b3687bde1ceb38f23.jpg':[(2,330,120,1080,930),(3,1310,690,1515,910),(0,1510,1110,1706,1279)],
'1e0600e7c44ad48876c05e9125ad2520.jpg':[(2,250,0,980,730),(3,1160,500,1340,730),(0,1240,905,1415,1205)],
'2347c0379a4047b4edd106386320d3cf.jpg':[(2,400,290,1165,1000)],
'25720402ad40ca28bc072ec1fd5c8acd.jpg':[(2,220,340,1190,810)],
'262caa9c7a97ec0ab3f623c8460ac6e2.jpg':[(4,660,160,1380,1130)],
'2abf39d30f1fd9e7332fbaf3fb7e0293.jpg':[(3,595,410,1010,820),(4,160,820,750,1490)],
'30369bb8428bcc5532ee4084c7f8d5ad.jpg':[(2,310,270,1165,930)],
'3736b8184c4e5ae86a53c08df1eb5b40.jpg':[(0,885,345,1165,805)],
'458679203d547b54a2408788f64fa32c.jpg':[(4,370,460,905,1030),(3,1040,310,1390,805)],
'4b52f9b2d3bbf260db733263069e2bd5.jpg':[(3,300,320,740,710),(3,890,270,1320,845)],
'55a64229d4b70b14eaeaae8418ef43c8.jpg':[(3,620,90,1150,900)],
'6003090d371cf667f7fd64bc5910d414.jpg':[(2,125,380,1045,885),(3,1310,550,1515,800),(0,1480,1050,1706,1279)],
'63e77d00ff7615130df5b6e0c90c7e91.jpg':[(2,420,240,1270,1020)],
'8d3289b47cfd62eb6f250d551ae709c9.jpg':[(4,380,80,1090,1040)],
'9a1f8e1aa6cc90036fd74b8500ca3b38.jpg':[(4,470,220,1100,1090)],
'a458f532fd147800cde7936bdab6d914.jpg':[(4,290,260,1050,1150)],
'a741c501622cc8725b1a691b54bd7a46.jpg':[(4,300,270,970,700),(3,1020,0,1410,580),(3,790,510,1240,1030)],
'ac25ecc08fbfefea8baad76c978a49e8.jpg':[(1,590,20,1690,870),(3,410,190,840,590),(3,100,580,590,990)],
'b150b341062f4d1cce7a9e6a6d6d14ac.jpg':[(4,930,410,1680,980)],
'c347d893b6e2351548edad6f2f18bacd.jpg':[(2,420,540,965,1255)],
'cfeca885a8620a3819ac7cf516f376f2.jpg':[(0,700,510,950,960)],
'dae5116440b5495b96b577e4364c1f0b.jpg':[(1,400,360,1280,710)],
'e6fa3360513e8b78544b1c302d74edef.jpg':[(2,360,300,1320,890)],
'e7da96fd1d60bb638400c8ba11332895.jpg':[(4,350,300,1080,850),(3,340,890,770,1300)],
'f3d6f5085991a685e1ba6dc5be6e36c9.jpg':[(3,250,420,750,760),(3,790,530,1300,900)],
'fec971af1db779aed9628e268518e38d.jpg':[(4,150,0,970,790),(3,970,80,1530,560)]}
def normalise_px(image, boxes):
 W,H=Image.open(image).size
 return [(c,(x1+x2)/(2*W),(y1+y2)/(2*H),(x2-x1)/W,(y2-y1)/H) for c,x1,y1,x2,y2 in boxes]
def typ(b):
 ids={x[0] for x in b}; return 'NEG' if not ids else ('MIX' if len(ids)>1 else ('MOU','KEY','LAP','CUP','HDP')[next(iter(ids))])
def read(p): return [tuple([int(x.split()[0])]+[float(v) for v in x.split()[1:]]) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def write(p,b): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(''.join(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c,x,y,w,h in b),encoding='utf-8')
def draw_preview(src,dst,boxes):
 im=Image.open(src).convert('RGB'); d=ImageDraw.Draw(im); f=ImageFont.load_default(size=22); W,H=im.size
 for c,x,y,w,h in boxes:
  box=((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H); d.rectangle(box,outline=COLORS[c],width=5); d.text((box[0]+3,max(0,box[1]+3)),NAMES[c],fill=COLORS[c],font=f,stroke_width=1,stroke_fill='black')
 dst.parent.mkdir(parents=True,exist_ok=True); im.save(dst,quality=92)
def session_for(old,name):
 if old!='S06': return 'S01' if f'{old}/{name}' in LAB else 'S02'
 if name in FLOOR:return 'S03'
 if name in DOLL:return 'S05'
 if name=='ac25ecc08fbfefea8baad76c978a49e8.jpg':return 'S02'
 return 'S04'
def scene(s): return {'S01':'light_lab_table','S02':'curved_wood_desk','S03':'floor_tile','S04':'simpsons_desk','S05':'doll_desk'}[s]
def instances(s,boxes):
 ids=[]
 for c in sorted({b[0] for b in boxes}):
  base=NAMES[c]
  suffix={'S01':'lab_01','S02':'wood_01','S03':'lenovo_floor_01','S04':'simpsons_01','S05':'doll_01'}[s]
  if c==2 and s in {'S01','S02','S03'}: suffix='lenovo_01'
  ids.append(f'{base}_{suffix}')
 return ';'.join(ids)
def main():
 if V2.exists(): shutil.rmtree(V2)
 records=[]
 for olddir in sorted(RAW.glob('S*')):
  for image in sorted(olddir.glob('*.jpg')):
   old=olddir.name; boxes=normalise_px(image,S06_PX[image.name]) if old=='S06' else read(V1/old/f'{image.stem}.txt'); s=session_for(old,image.name)
   if image.relative_to(ROOT).as_posix() in EXCLUDED_RAW: continue
   records.append({'src':image,'old':old,'session':s,'boxes':boxes,'source':'new_from_old_s06' if old=='S06' else 'existing_old','scenario':'negative' if not boxes else ('multi' if len({b[0] for b in boxes})>1 else 'basic')})
 records.sort(key=lambda r:(r['session'],typ(r['boxes']),r['src'].as_posix())); seq=Counter()
 for r in records:
  t=typ(r['boxes']); seq[(r['session'],t)]+=1
  offset=1 if (r['session']=='S05' and t=='MIX') else 0
  r['name']=f"P01_{r['session']}_{t}_{seq[(r['session'],t)]+offset:04d}.jpg"
  for root in (V2/'images/P01',V2/'previews/P01'):(root/r['session']).mkdir(parents=True,exist_ok=True)
  shutil.copy2(r['src'],V2/'images/P01'/r['session']/r['name']); write(V2/'labels/P01'/r['session']/r['name'].replace('.jpg','.txt'),r['boxes']); draw_preview(r['src'],V2/'previews/P01'/r['session']/r['name'],r['boxes'])
 manifests=V2/'manifests'; manifests.mkdir(parents=True,exist_ok=True)
 fields=['original_path','old_session','new_session','new_filename','source_type','scenario','primary_class','object_instance_ids','scene_id','capture_group_reason','label_status','review_status']
 with (manifests/'image_manifest.csv').open('w',newline='',encoding='utf-8') as h:
  w=csv.DictWriter(h,fields);w.writeheader()
  for r in records:w.writerow({'original_path':r['src'].relative_to(ROOT).as_posix(),'old_session':r['old'],'new_session':r['session'],'new_filename':r['name'],'source_type':r['source'],'scenario':r['scenario'],'primary_class':NAMES[r['boxes'][0][0]] if len(r['boxes'])==1 else ('negative' if not r['boxes'] else 'mixed'),'object_instance_ids':instances(r['session'],r['boxes']),'scene_id':scene(r['session']),'capture_group_reason':f"Visual audit: {scene(r['session'])} background, objects, and camera context form one capture group.",'label_status':'manual_reannotation' if r['source']=='new_from_old_s06' else 'reused_visually_checked','review_status':'approved'})
 with (manifests/'excluded_images.csv').open('w',newline='',encoding='utf-8') as h:
  w=csv.writer(h); w.writerow(['original_path','original_filename','original_session','status','reason','decision_source'])
  for p in sorted(EXCLUDED_RAW): w.writerow([p,Path(p).name,'S06','excluded','manual_quality_rejection','user_manual_review'])
 (manifests/'manual_review.csv').write_text('file,issue,recommended_action\n',encoding='utf-8')
 with (manifests/'object_instances.csv').open('w',newline='',encoding='utf-8') as h:
  w=csv.writer(h);w.writerow(['new_session','object_instance_id','scene_id'])
  for s in range(1,6):
   rs=[r for r in records if r['session']==f'S{s:02d}']; ids=sorted({x for r in rs for x in instances(r['session'],r['boxes']).split(';') if x})
   for i in ids:w.writerow([f'S{s:02d}',i,scene(f'S{s:02d}')])
 with (manifests/'session_summary.csv').open('w',newline='',encoding='utf-8') as h:
  w=csv.writer(h);w.writerow(['session','images','objects','scene','source_old_sessions'])
  for s in range(1,6):
   rs=[r for r in records if r['session']==f'S{s:02d}'];w.writerow([f'S{s:02d}',len(rs),sum(len(r['boxes']) for r in rs),scene(f'S{s:02d}'),';'.join(sorted({r['old'] for r in rs}))])
 # 72.5/15/12.5; no capture session crosses a split.
 split={'S01':'train','S02':'train','S03':'test','S04':'val','S05':'test'}; paths=defaultdict(list)
 for r in records: paths[split[r['session']]].append(r)
 sp=V2/'splits';sp.mkdir(parents=True,exist_ok=True)
 for name in ('train','val','test'):(sp/f'{name}.txt').write_text('\n'.join(f"data/v2/images/P01/{r['session']}/{r['name']}" for r in paths[name])+'\n',encoding='utf-8')
 with (manifests/'split_summary.csv').open('w',newline='',encoding='utf-8') as h:
  w=csv.writer(h);w.writerow(['split','sessions','images','objects','unique_object_instances'])
  for name in ('train','val','test'):
   rs=paths[name];w.writerow([name,';'.join(sorted({r['session'] for r in rs})),len(rs),sum(len(r['boxes']) for r in rs),len({x for r in rs for x in instances(r['session'],r['boxes']).split(';') if x})])
 # Materialise YOLO inputs locally; only YAML is tracked.
 y=V2/'yolo';(y/'images').mkdir(parents=True,exist_ok=True);(y/'labels').mkdir(parents=True,exist_ok=True)
 for name,rs in paths.items():
  for r in rs:
   (y/'images'/name).mkdir(parents=True,exist_ok=True);(y/'labels'/name).mkdir(parents=True,exist_ok=True)
   shutil.copy2(V2/'images/P01'/r['session']/r['name'],y/'images'/name/r['name']);shutil.copy2(V2/'labels/P01'/r['session']/r['name'].replace('.jpg','.txt'),y/'labels'/name/r['name'].replace('.jpg','.txt'))
 (y/'dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(NAMES)),encoding='utf-8')
 # Contact sheets use final annotated previews, completing the second visual QA artefact.
 contacts=V2/'previews/session_contact_sheets';contacts.mkdir(parents=True,exist_ok=True)
 for s in sorted({r['session'] for r in records}):
  rs=[r for r in records if r['session']==s];cw,ch=280,220; cols=4; sheet=Image.new('RGB',(cols*cw,((len(rs)+cols-1)//cols)*ch),'white');d=ImageDraw.Draw(sheet);f=ImageFont.load_default(size=16)
  for i,r in enumerate(rs):
   im=Image.open(V2/'previews/P01'/s/r['name']).convert('RGB');im.thumbnail((cw-8,ch-28));x=(i%cols)*cw+4;y=(i//cols)*ch+22;sheet.paste(im,(x+(cw-8-im.width)//2,y));d.text((x,(i//cols)*ch+3),r['name'],fill='black',font=f)
  sheet.save(contacts/f'{s}.jpg',quality=90)
 print(f'images={len(records)} objects={sum(len(r["boxes"]) for r in records)} previews={len(records)} sessions=5')
if __name__=='__main__':main()
