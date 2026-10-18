from pathlib import Path
import json,hashlib
out=Path('/mzy/copyfrom_aixi_gadfortest/outputs/sam_person/20261018_v1')
c=json.loads((out/'paired/B0/config_actual.json').read_text())['config'];expected=c['code_hash']
source=out/'source_snapshot';files={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*.py')}
def replace(path,a,b):
 assert files[path].count(a.encode())==1,(path,a[:80]);files[path]=files[path].replace(a.encode(),b.encode(),1)
replace('models/sam_roi.py',', sample_valid=None):','):')
replace('models/sam_roi.py', '''    # The constant-preserving form of the same linear interpolation avoids
    # RoIAlign(ones)=1 +/- one ulp. TF32 can amplify that rounding in fc_emb.
    # This is spatial filtering, with no new pooling or feature normalization.
    weights = (1-roi_align((1-masks.detach()).reshape(count, 1, h, w).to(rois),
                          [box[None] for box in boxes_feature_xyxy])).clamp(0, 1)
    if sample_valid is not None:
        # Preserve B0's whole-sample zero extrapolation at image boundaries.
        weights=weights*sample_valid[:,None].to(weights.dtype)
''','''    weights = roi_align(masks.detach().reshape(count, 1, h, w).to(rois),
                        [box[None] for box in boxes_feature_xyxy])
''')
replace('models/b0_repro.py',', sample_valid=valid_samples)',')')
replace('tests/sam_person/test_contract.py', '''        reference=roi*align(masks.reshape(3,1,7,13),[b[None] for b in boxes])
        torch.testing.assert_allclose(changed[:2],reference[:2],rtol=1e-6,atol=5e-7)
''','')
replace('tools/sam_person_masks.py','''        roi_weights = (1-align(1-grid[:,None],[b[None] for b in boxes_feature])).clamp(0,1)[:,0]
        bins=torch.arange(5,dtype=boxes_feature.dtype)
        sx=boxes_feature[:,0,None]+(bins[None]+.5)*(boxes_feature[:,2]-boxes_feature[:,0])[:,None]/5-.5
        sy=boxes_feature[:,1,None]+(bins[None]+.5)*(boxes_feature[:,3]-boxes_feature[:,1])[:,None]/5-.5
        sample_valid=((sy>=0)&(sy<=c['feature_hw'][0]-1))[:,:,None] & ((sx>=0)&(sx<=c['feature_hw'][1]-1))[:,None,:]
        roi_weights=roi_weights*sample_valid.to(roi_weights.dtype)
''','''        roi_weights = align(grid[:,None],[b[None] for b in boxes_feature])[:,0]
''')
replace('tools/run_sam_person_pilot.py','''                        if stage=='overfit' and record['steps']==3:
                            atomic_json(out/'technical_smoke.json',dict(variant=cli.variant,steps=3,
                                status='passed',finite_loss=True,finite_gradients=True,
                                init_hash=init_hash,code_hash=c['code_hash'],
                                sample_order_hash=digest(canonical(order)),integrated_in_overfit=True))
''','')
manifest={name:hashlib.sha256(text).hexdigest() for name,text in files.items()}
body=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();actual=hashlib.sha256(body).hexdigest()
assert actual==expected,(actual,expected)
target=out/'source_archives'/expected
for name,text in files.items():
 p=target/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(text)
(target/'source_manifest.json').write_bytes(body)
(target/'reconstruction_provenance.json').write_text(json.dumps(dict(method='reverse the five documented numerical-fix changes from the next source snapshot; manifest SHA256 exactly matches failed run config_actual.json',expected=expected,actual=actual,verified=True),indent=2)+'\n')
print('archived exact failed-run source',actual)
