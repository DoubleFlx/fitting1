"""Seed45 parameter-only Source adapter; frozen full-resolution RGB operator."""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ultralytics.data.dataset import YOLODataset

import m1_routea_photometric_realization_core_v1 as core
import m1_ss0_seed45_v4_source_inputs as trace
from protocol_b_m1_data_isolation import install_m1_data_isolation_guard

ROOT = Path(__file__).resolve().parents[2]
SS1 = ROOT / 'project/results/SS1_S45_CPU_V1'
SS2 = ROOT / 'project/results/SS2_S45_G_ADAPTER_V1'
SNAPSHOT = ROOT / 'project/audits/M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_20260808_120520/CODE_SNAPSHOT'
core.configure_q2_snapshot(SNAPSHOT)  # Import before worker RNG setup.


def sha(path):
    return core.sha256_file(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def active(uid, epoch):
    assert 72 <= epoch <= 100
    key = f'M1_SS0_SOURCE_INTRADOMAIN_V1|ACTIVE|45|{epoch}|{uid}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') / float(1 << 64) < .5



_SS3_PALETTE_IMPLEMENTATION_ANNOUNCED = False


def _ss3_exact_deployment_palette(rgb):
    """Build the exact RGB palette, counts and inverse map."""
    global _SS3_PALETTE_IMPLEMENTATION_ANNOUNCED
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PaletteImage requires uint8 HxWx3 RGB")
    height, width = image.shape[:2]
    packed = ((image[..., 0].astype(np.uint32) << 16)
              | (image[..., 1].astype(np.uint32) << 8)
              | image[..., 2].astype(np.uint32))
    dense_counts = np.bincount(packed.reshape(-1))
    support_size = len(dense_counts)
    unique = np.flatnonzero(dense_counts).astype(np.uint32)
    counts = dense_counts[unique].copy()
    del dense_counts
    lookup = np.empty(support_size, dtype=np.int32)
    lookup[unique] = np.arange(len(unique), dtype=np.int32)
    inverse = lookup[packed.reshape(-1)]
    del lookup, packed
    palette = np.stack([((unique >> 16) & 255).astype(np.uint8),
                        ((unique >> 8) & 255).astype(np.uint8),
                        (unique & 255).astype(np.uint8)], axis=1)
    srgb = palette.astype(np.float64) / 255.0
    y_prime = srgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    ell = np.log(y_prime + 1e-6)
    median_ell = core.inverse_ecdf_integer_counts(ell, counts, 0.50)
    if not _SS3_PALETTE_IMPLEMENTATION_ANNOUNCED:
        import os
        print(f"SS3_EXACT_INTEGER_PALETTE_V1 pid={os.getpid()} pixel_math_unchanged=True", flush=True)
        _SS3_PALETTE_IMPLEMENTATION_ANNOUNCED = True
    return core.PaletteImage(height=int(height), width=int(width),
        palette_uint8=palette, inverse_flat=inverse,
        counts=counts.astype(np.int64, copy=False), cell_memberships=[],
        srgb=srgb, y_prime=y_prime, ell=ell, median_ell=median_ell)


def render_rgb(rgb, theta_hex):
    """No solver, RNG draws, resize, or descriptor/geometry refit."""
    theta = np.frombuffer(bytes.fromhex(theta_hex), dtype='<f8').copy()
    assert theta.shape == (3,) and np.isfinite(theta).all()
    core.configure_q2_snapshot(SNAPSHOT)
    with threadpool_limits(limits=1):
        palette = _ss3_exact_deployment_palette(rgb)
        continuous, _, _, _ = palette.continuous_output(theta)
        deployed = np.rint(continuous * 255.).astype(np.uint8)
        return np.ascontiguousarray(deployed[palette.inverse_flat].reshape(rgb.shape))


def formal_sha(bgr):
    return hashlib.sha256(core.ARRAY_HASH_PREFIX + b'uint8' + str(tuple(bgr.shape)).encode('ascii') + bgr.tobytes()).hexdigest()


def load_lookup():
    jobs = {j['uid']: j for j in read(SS1 / 'FROZEN_JOBS.json')}
    rows = read(SS2 / 'FULL_SOURCE_PARAMETER_MANIFEST.json')
    units = {r['uid']: r for r in read(SS1 / 'FINAL_PARAMETER_INDEX.json')['results']}
    lookup = {}
    for row in rows:
        job = jobs[row['uid']]
        record = dict(row, image_path=job['image_path'], image_sha256=job['image_sha256'], shape=job['shape'], unit=units.get(row['uid']))
        stem = Path(job['image_path']).stem
        assert stem not in lookup
        lookup[stem] = record
    assert len(lookup) == 17637
    return lookup


class ParameterRenderer:
    def __init__(self, max_bytes=256 * 1024**2):
        self.max_bytes = max_bytes
        self.cache = OrderedDict()
        self.cache_bytes = 0

    def __call__(self, record, view):
        key = (record['uid'], view, record[view + '_theta_f8_hex'])
        assert record[view + '_status'] == 'COMPLETE' and key[2]
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        path = Path(record['image_path']).resolve()
        assert path.is_relative_to((ROOT / 'datasets/dataset_final_v2_1/source/train/images').resolve())
        assert sha(path) == record['image_sha256'], 'SOURCE_IMAGE_HASH_DRIFT'
        core.configure_q2_snapshot(SNAPSHOT)
        rgb, _ = core.decode_q2_rgb(path)
        assert list(rgb.shape) == record['shape']
        output = render_rgb(rgb, key[2])
        bgr = np.ascontiguousarray(output[:, :, ::-1])
        if view == 'S':
            unit = record['unit']
            assert sha(unit['path']) == unit['sha256'], 'UNIT_RESULT_HASH_DRIFT'
            result = read(unit['path'])
            assert formal_sha(bgr) == result['science']['output_array_sha256'], 'S_DEPLOYMENT_DRIFT'
        if bgr.nbytes <= self.max_bytes:
            while self.cache_bytes + bgr.nbytes > self.max_bytes:
                _, evicted = self.cache.popitem(last=False)
                self.cache_bytes -= evicted.nbytes
            self.cache[key] = bgr
            self.cache_bytes += bgr.nbytes
        return bgr.copy()


class ParameterSourceDataset(YOLODataset):
    def __init__(self, *args, source_method, parameter_lookup, audit_dir, source_epoch=72, **kwargs):
        assert source_method in ('DISABLED', 'S', 'G')
        assert 72 <= source_epoch <= 100
        self.source_method = source_method
        self.source_epoch = source_epoch
        self.parameter_lookup = parameter_lookup
        self.audit_dir = Path(audit_dir)
        self.ss0_components = []
        self.renderer = ParameterRenderer()
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        guard = install_m1_data_isolation_guard(phase='SS2_PARAMETER_' + self.source_method, audit_dir=self.audit_dir, allow_target_dev_images=False)
        trace.install_trace()
        trace._TRACE, trace._PRE_HSV = [], []
        self.ss0_components = []
        before = trace.rng_sha()
        try:
            result = super().__getitem__(index)
            assert len(trace._PRE_HSV) == 1, 'CANONICAL_HSV_INVOCATION_COUNT'
            result['ss0_pre_hsv'] = torch.from_numpy(np.ascontiguousarray(trace._PRE_HSV[0][:, :, ::-1].transpose(2, 0, 1)))
            result['ss0_audit'] = dict(components=self.ss0_components, geometry=trace._TRACE, rng_before=before, rng_after=trace.rng_sha())
            guard.export_process_audit()
            return result
        finally:
            trace._TRACE = trace._PRE_HSV = None

    def load_image(self, i, *args, **kwargs):
        original = super().load_image(i, *args, **kwargs)
        stem = Path(self.im_files[i]).stem
        record = self.parameter_lookup[stem]
        uid = record['uid']
        enabled = self.source_method in ('S', 'G')
        flag = active(uid, self.source_epoch)
        view = 'S' if self.source_method == 'S' else 'G'
        status = record[view + '_status']
        apply = enabled and flag and status == 'COMPLETE'
        buffer_sha = hashlib.sha256(original[0].tobytes()).hexdigest()
        result = original
        if apply:
            bgr = self.renderer(record, view)
            assert bgr.shape[:2] == tuple(original[1])
            h, w = original[2]
            if bgr.shape[:2] != (h, w):
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            assert not np.shares_memory(bgr, original[0])
            result = (bgr, original[1], original[2])
        assert hashlib.sha256(original[0].tobytes()).hexdigest() == buffer_sha, 'ORIGINAL_BUFFER_POLLUTED'
        self.ss0_components.append(dict(uid=uid, schedule_stem=stem, index=int(i), active=flag, applied=apply, status=status if enabled else 'DISABLED',
            theta_f8_hex=record[view + '_theta_f8_hex'] if enabled else None, original_hw=list(original[1]), resized_hw=list(original[2]), buffer_size=len(self.buffer),
            original_resized_sha256=buffer_sha, output_resized_sha256=hashlib.sha256(result[0].tobytes()).hexdigest(), original_buffer_unchanged=True))
        return result
