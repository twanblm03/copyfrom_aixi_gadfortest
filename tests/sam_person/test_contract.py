import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torchvision.ops import RoIAlign
from models import build_model
from models.sam_roi import filter_actor_rois
from util.sam_person import (SCHEMA, canonical, digest, file_hash, read_frame,
    write_frame, transform_geometry, feature_masks, MaskReader)


def args_for(mode='disabled'):
    args=argparse.Namespace(**json.loads((ROOT/'configs/heatmap/b0.json').read_text()))
    args.train_mask_mode=mode
    return args


class CacheGeometry(unittest.TestCase):
    def test_real_annotation_parser_bounded_equals_default(self):
        from dataloader.cafe import cafe_read_annotations
        ann=dict(framesCount=35,framesEach=6,figures=[
            dict(id=41,label='group1',shapes=[dict(frame=102,coordinates=[[1,2],[7,12]])],
                 attributes=[dict(value=dict(key='Queueing'))]),
            dict(id=93,label='individual',shapes=[dict(frame=102,coordinates=[[9,3],[15,12]])],attributes=[])])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'1'/'2';path.mkdir(parents=True)
            (path/'ann.json').write_text(json.dumps(ann))
            default=cafe_read_annotations(temp,['1'],6)
            bounded=cafe_read_annotations(temp,['1'],6,clip_ids=[(1,2)])
            self.assertEqual(set(default),set(bounded))
            for key,value in default[(1,2)].items():
                if torch.is_tensor(value):self.assertTrue(torch.equal(value,bounded[(1,2)][key]),key)
                else:self.assertEqual(value,bounded[(1,2)][key])

    def test_provider_reuses_image_and_preserves_actor_order_and_axes(self):
        from tools.sam_person_masks import Provider
        class Predictor:
            calls=0
            def set_image(self,image,image_format):
                self.calls+=1;self.hw=image.shape[:2]
                self.assert_rgb=image_format=='RGB'
            def predict(self,box,multimask_output):
                mask=np.zeros((1,)+self.hw,bool)
                x1,y1,x2,y2=box.astype(int);mask[0,y1:y2,x1:x2]=True
                return mask,np.array([.8],np.float32),np.zeros((1,2,2))
        provider=Provider.__new__(Provider);provider.torch=torch;provider.c=dict(device='cpu')
        provider.predictor=Predictor()
        image=np.zeros((7,13,3),np.uint8);provider.encode(image)
        first,_,_=provider.predict([1,1,4,5],(7,13))
        second,_,_=provider.predict([8,2,12,6],(7,13))
        self.assertEqual(provider.predictor.calls,1);self.assertTrue(provider.predictor.assert_rgb)
        self.assertTrue(first[1,1]);self.assertFalse(first[2,8]);self.assertTrue(second[2,8])
        with self.assertRaises(ValueError):provider.predict([1,1,4,5],(13,7))

    def test_non_square_roundtrip_identity_and_invalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'image.bin';source.write_bytes(b'image fingerprint')
            row=dict(schema_version=SCHEMA,dataset='CAFE',protocol='place',split='train',
                video_id=1,clip_id=2,source_frame_index=3,source_image_key='1/2/3',
                source_image_path=str(source),source_image_hash=file_hash(source),original_hw=[7,13],
                actors=[dict(actor_uid='1/2/row0',source_box_xyxy=[0,0,8,7])])
            mask=np.zeros((1,7,13),bool);mask[0,1:6,2:8]=True
            records=[dict(actor_uid='1/2/row0',mask_checksum=digest(mask[0].tobytes()),raw_mask_valid=True)]
            write_frame(temp,'v1',row,mask,records)
            restored,_=read_frame(temp,'v1',row,True)
            np.testing.assert_array_equal(restored,mask)
            changed=copy.deepcopy(row);changed['actors'][0]['source_box_xyxy'][0]=1
            with self.assertRaises(ValueError):read_frame(temp,'v1',changed)
            changed=copy.deepcopy(row);changed['actors'][0]['actor_uid']='other/video/row0'
            with self.assertRaises(ValueError):read_frame(temp,'v1',changed)
            with self.assertRaises(FileNotFoundError):read_frame(temp,'model_changed',row)
            source.write_bytes(b'changed image')
            with self.assertRaises(ValueError):read_frame(temp,'v1',row,True)
            npz=next(Path(temp).rglob('*.npz'));npz.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):read_frame(temp,'v1',row)

    def test_shared_resize_crop_flip_and_soft_grid(self):
        masks=np.zeros((2,9,17),bool);masks[0,2:7,3:8]=True;masks[1,1:4,11:15]=True
        image=np.zeros((9,17,3),np.uint8);image[...,0]=masks[0]*255;image[...,1]=masks[1]*255
        boxes=np.array([[3,2,8,7],[11,1,15,4]],np.float32)
        resized,b,m=transform_geometry(image,boxes,masks,(18,34),(2,2,32,18),True)
        np.testing.assert_array_equal(m[0],resized[...,0]>127)
        np.testing.assert_allclose(b[0],[16,2,26,12])
        flipped,bf,mf=transform_geometry(image,boxes,masks,(9,17),flip=True)
        restored,br,mr=transform_geometry(flipped,bf,mf,(9,17),flip=True)
        np.testing.assert_array_equal(restored,image);np.testing.assert_allclose(br,boxes)
        np.testing.assert_array_equal(mr,masks)
        grid=feature_masks(masks,(18,34),(3,5))
        self.assertTrue(((grid>0)&(grid<1)).any())
        self.assertGreater(float(grid[0].sum()),0)


class RoIContract(unittest.TestCase):
    def setUp(self):torch.set_num_threads(4)

    def test_ones_fallback_causality_actor_permutation_and_gradients(self):
        align=RoIAlign((5,5),1.,1,True)
        boxes=torch.tensor([[0.,0.,13.,7.],[1.,1.,12.,6.],[-4.,-4.,1.,1.]])
        f=torch.randn(3,4,7,13,requires_grad=True)
        roi=align(f,[b[None] for b in boxes])
        valid=torch.tensor([[[True,True,False]]]);raw=torch.ones_like(valid)
        ones=torch.ones(1,1,3,7,13)
        same,stats=filter_actor_rois(roi,boxes,ones,valid,raw,align)
        torch.testing.assert_allclose(same,roi,rtol=0,atol=0)
        masks=ones.clone();masks[:,:,:,:,6:]=0
        changed,_=filter_actor_rois(roi,boxes,masks,valid,raw,align)
        reference=roi*align(masks.reshape(3,1,7,13),[b[None] for b in boxes])
        torch.testing.assert_allclose(changed[:2],reference[:2],rtol=1e-6,atol=5e-7)
        self.assertFalse(torch.equal(changed[:2],roi[:2]))
        shifted=masks.flip(-1)
        shifted_roi,_=filter_actor_rois(roi,boxes,shifted,valid,raw,align)
        self.assertFalse(torch.equal(changed[:2],shifted_roi[:2]))
        perm=torch.tensor([1,2,0]);inverse=perm.argsort()
        permuted,_=filter_actor_rois(roi[perm],boxes[perm],masks[:,:,perm],valid[:,:,perm],raw[:,:,perm],align)
        torch.testing.assert_allclose(permuted[inverse],changed,rtol=0,atol=0)
        changed.square().sum().backward()
        self.assertTrue(torch.isfinite(f.grad).all());self.assertGreater(float(f.grad.abs().sum()),0)
        fallback,stats=filter_actor_rois(roi.detach(),boxes,torch.zeros_like(ones),valid,raw,align)
        self.assertEqual(stats['feature_grid_empty'],2);self.assertTrue(torch.isfinite(fallback).all())
        torch.testing.assert_allclose(fallback,roi.detach(),rtol=0,atol=0)
        fallback,stats=filter_actor_rois(roi.detach(),boxes,ones,valid,~raw,align)
        self.assertEqual(stats['raw_failed'],2)

    def test_disabled_original_full_model_loss_export_and_rng(self):
        # Execute the pre-SAM B0 revision as a fixed reference. The user had
        # no changes to this file at entry; original diff is preserved separately.
        original=subprocess.check_output(['git','show','33ca78cffd46d3bc50b7d3515ede28f70de842be:models/b0_repro.py'],cwd=ROOT).decode()
        module=types.ModuleType('models._sam_test_original');module.__package__='models'
        exec(compile(original,'committed_b0_repro.py','exec'),module.__dict__)
        torch.manual_seed(51)
        reference=module.GADTRB0Repro(args_for()).eval();rng=torch.get_rng_state().clone()
        torch.manual_seed(51)
        model,criterion=build_model(args_for());model.eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in reference.state_dict().items():
            self.assertTrue(torch.equal(value,model.state_dict()[key]),key)
        torch.manual_seed(51)
        enabled,_=build_model(args_for('sam_roi'));enabled.eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        images=torch.randn(1,5,3,64,96)
        boxes=torch.tensor([.5,.5,.4,.6]).repeat(1,5,14,1)
        dummy=torch.zeros(1,14,dtype=torch.bool);valid=torch.ones(1,5,14,dtype=torch.bool)
        with torch.no_grad():
            expected=reference(images,boxes,dummy,actor_frame_valid=valid)
            actual=model(images,boxes,dummy,actor_frame_valid=valid,actor_masks=torch.tensor(float('nan')))
            ones=enabled(images,boxes,dummy,actor_frame_valid=valid,
                         actor_masks=torch.ones(1,5,14,2,3),raw_mask_valid=valid)
        errors={key:float((expected[key]-ones[key]).abs().max()) for key in expected}
        print('full-model ones max errors',errors,flush=True)
        for key in expected:
            torch.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=0)
            torch.testing.assert_allclose(ones[key],expected[key],rtol=0,atol=0)
        members=torch.zeros(1,1,14);members[0,0,:2]=1
        target=dict(actions=torch.zeros(1,14,dtype=torch.long),activities=torch.zeros(1,1,dtype=torch.long),
            members=members,membership=torch.tensor([[0,0]+[-1]*12]),boxes=boxes[0],actor_frame_valid=valid[0])
        losses_ref=criterion(expected,[copy.deepcopy(target)])
        losses_new=criterion(actual,[copy.deepcopy(target)])
        for key in losses_ref:torch.testing.assert_allclose(losses_ref[key],losses_new[key],rtol=0,atol=0)
        from util.group_export import make_txt
        with tempfile.TemporaryDirectory() as temp:
            info=[dict(vid=1,sid=2,fid=[0,6,12,18,24])]
            a,b=Path(temp)/'a.txt',Path(temp)/'b.txt'
            make_txt(boxes,info,expected,{1:1},str(a),args_for())
            make_txt(boxes,info,actual,{1:1},str(b),args_for())
            self.assertEqual(a.read_bytes(),b.read_bytes())
        masks=torch.ones(1,5,14,2,3);masks[...,0]=0
        enabled.train()
        out=enabled(images,boxes,dummy,actor_frame_valid=valid,actor_masks=masks,raw_mask_valid=valid)
        losses=criterion(out,[copy.deepcopy(target)])
        total=sum(losses[k]*criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
        total.backward()
        for prefix in ('backbone.','fc_emb.'):
            grads=[p.grad for name,p in enabled.named_parameters() if name.startswith(prefix) and p.grad is not None]
            self.assertTrue(grads);self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
        enabled.eval()
        with torch.no_grad():
            fixed=enabled(images,boxes,dummy,actor_frame_valid=valid,actor_masks=masks,raw_mask_valid=valid)
            altered=copy.deepcopy(target);altered['activities'][:]=4;altered['members'].zero_()
            criterion(fixed,[altered])
            after=enabled(images,boxes,dummy,actor_frame_valid=valid,actor_masks=masks,raw_mask_valid=valid)
        for key in fixed:torch.testing.assert_allclose(after[key],fixed[key],rtol=0,atol=0)
        # Complete padded frame + no GT groups uses the unchanged B0 guards;
        # new mask processing must neither create NaNs nor invent actors.
        none_valid=torch.zeros_like(valid)
        with torch.no_grad():
            empty=enabled(images,torch.zeros_like(boxes),torch.ones_like(dummy),
                actor_frame_valid=none_valid,actor_masks=torch.zeros_like(masks),raw_mask_valid=none_valid)
        self.assertTrue(all(torch.isfinite(v).all() for v in empty.values()))
        empty_target=dict(actions=torch.full((1,14),7,dtype=torch.long),
            activities=torch.zeros((1,0),dtype=torch.long),members=torch.zeros(1,0,14),
            membership=torch.full((1,14),-1,dtype=torch.long),boxes=torch.zeros_like(boxes[0]),
            actor_frame_valid=none_valid[0])
        empty_losses=criterion(empty,[empty_target])
        self.assertTrue(all(torch.isfinite(v).all() for v in empty_losses.values()))


if __name__=='__main__':unittest.main()
