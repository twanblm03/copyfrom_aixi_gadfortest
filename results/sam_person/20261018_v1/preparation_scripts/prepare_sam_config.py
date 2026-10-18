import sys,json,hashlib
from pathlib import Path
root=Path('/mzy/copyfrom_aixi_gadfortest');sys.path.insert(0,str(root))
from util.sam_person import file_hash,digest,canonical,SCHEMA
from tools.sam_person_masks import version
repo=Path('/mzy/cafe_runs/sam-probe/python_packages')
ckpt=Path('/mzy/cafe_runs/sam-probe/checkpoints/sam_vit_b_01ec64.pth')
source={str(p.relative_to(repo)):file_hash(p) for p in sorted(repo.rglob('*.py'))}
out=root/'outputs/sam_person/20261018_v1'
c=dict(schema_version=SCHEMA,project_root=str(root),dataset_adapter='real dataloader.dataloader.read_dataset + CafeDataset.load_samples/select_frames',protocol='place_train_val_b0_repro',data_root='/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/cafe',tracks_path='/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset/cafe/gt_tracks.pkl',source_frame_mapping='clip-relative floor(segment centers)*framesEach; absolute video frame mapping unavailable',actor_box_source='given GT actor tracklets gt_tracks.pkl, matching B0; visible vs amodal semantics not independently documented',box_coordinate_convention='normalized_cxcywh_to_original_pixel_xyxy',sam_python='/share/apps/anaconda3/bin/python',gad_python='/share/apps/anaconda3/envs/torch171/bin/python',sam_repo=str(repo),sam_family='sam_v1_vit_b',sam_commit='dca509fe793f601edb92606367a655c15ac00fdf',sam_commit_evidence='installed direct_url.json archive path; source content hash locked',sam_source_hash=digest(canonical(source)),checkpoint=str(ckpt),checkpoint_sha256=file_hash(ckpt),model_config='original SAM build_sam_vit_b',sam_config_hash=file_hash(repo/'segment_anything/build_sam.py'),device='cuda:0',physical_gpu_ids=[4,5,6,7],generation_gpu_uuid='GPU-d2b35225-e4f2-e4d6-e4e1-d289490a03ab',precision='fp32',prompt_policy='box_only',multimask_output=False,mask_threshold=0.,postprocess_options={'raw_binary':True,'clip_to_box':False,'largest_component':False,'nms':False,'custom_cuda_extension':'not required by original SAM'},cache_root=str(out/'cache'),cache_version='',sample_seed=20261018,per_split_limits={'train':120,'val':80},max_frames=200,max_actor_prompts=3000,max_gpu_minutes=60,generation_ledger=str(out/'generation_budget.json'),train_mask_mode='sam_roi',empty_eps=1e-6,fallback_policy='ordinary ROI only for explicit raw failure/empty or feature ROI mean<=eps; invalid actors unchanged',image_hw=[720,1280],feature_hw=[23,40],independent_pixel_gt=None)
c['sampling_split_manifest']='/mzy/cafe_runs/heatmap/h1_match_full_seed1_resume_epoch7_20261013_193136/run/data_manifest.json'
c['sampling_split_manifest_sha256']=file_hash(c['sampling_split_manifest'])
c['cache_version']=version(c)
(root/'configs/sam_person/qc_resolved.yaml').write_text(json.dumps(c,indent=2)+'\n')
(out/'sam_source_hashes.json').write_text(json.dumps(source,indent=2)+'\n')
print(c['cache_version'],c['checkpoint_sha256'])
