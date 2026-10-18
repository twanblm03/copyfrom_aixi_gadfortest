import sys,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
root=Path('/mzy/copyfrom_aixi_gadfortest');sys.path.insert(0,str(root))
from util.sam_person import read_frame
c=json.loads((root/'configs/sam_person/qc_resolved.yaml').read_text());out=root/'outputs/sam_person/20261018_v1/smoke_gallery'
rows=[json.loads(l) for l in (root/'outputs/sam_person/20261018_v1/smoke_manifest.jsonl').read_text().splitlines()]
for row in rows:
 masks,meta=read_frame(c['cache_root'],c['cache_version'],row)
 rgb=np.asarray(Image.open(row['source_image_path']).convert('RGB'))
 sheet=Image.new('RGB',(1536,260*math.ceil(len(masks)/4)),'white')
 for j,(actor,mask) in enumerate(zip(row['actors'],masks)):
  box=actor['source_box_xyxy'];x1,y1,x2,y2=box
  dx=max(20,(x2-x1)*.2);dy=max(20,(y2-y1)*.15)
  crop=(max(0,int(x1-dx)),max(0,int(y1-dy)),min(rgb.shape[1],int(x2+dx)),min(rgb.shape[0],int(y2+dy)))
  original=Image.fromarray(rgb.copy());draw=ImageDraw.Draw(original)
  for a in row['actors']:draw.rectangle(a['source_box_xyxy'],outline='gray',width=2)
  draw.rectangle(box,outline='red',width=3)
  painted=rgb.astype(float).copy();painted[mask]=painted[mask]*.5+np.array([0,255,0])*.5
  overlay=Image.fromarray(painted.astype('uint8'));ImageDraw.Draw(overlay).rectangle(box,outline='red',width=3)
  cell=Image.new('RGB',(384,260),'white')
  for k,img in enumerate((original,overlay)):
   img=img.crop(crop);img.thumbnail((188,234));cell.paste(img,(k*192+(192-img.width)//2,24+(234-img.height)//2))
  ImageDraw.Draw(cell).text((3,3),'row%d Q=%.3f'%(actor['actor_row'],meta['records'][j]['predicted_quality']),fill='black')
  sheet.paste(cell,((j%4)*384,(j//4)*260))
 name='contact_%d_%d_%d.jpg'%(row['video_id'],row['clip_id'],row['source_frame_index']);sheet.save(out/name);print(name)
