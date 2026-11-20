from __future__ import annotations

from pathlib import Path


ROOT = Path(r"D:\CV_YOLO11_ProtocolB_Formal")
OUT = ROOT / "project" / "audits" / "M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_20260808_120520"
CODE = OUT / "CODE_SNAPSHOT"
RAW = OUT / "raw"
FINAL_ZIP = ROOT / "M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_UNIFIED.zip"
PRIOR_STOP_EVIDENCE_DIR = (
    ROOT
    / "project"
    / "audits"
    / "M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_STOP_DEPENDENCE_IMPLEMENTATION_UNAVAILABLE_20260808"
)
PRIOR_STOP_EVIDENCE_ZIP = (
    PRIOR_STOP_EVIDENCE_DIR
    / "M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_STOP_DEPENDENCE_IMPLEMENTATION_UNAVAILABLE_UNIFIED.zip"
)
PRIOR_STOP_EVIDENCE_ZIP_SHA256 = "e3ad530dfcb9b31f173c97e10fad9cd856da4978f7805f023e9449892af2de4b"

TASK_NAME = "M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1"
ALLOWED_METADATA = ROOT / "project" / "audit_inputs" / "ALLOWED_METADATA_SOURCES.json"
EXPECTED_ALLOWLIST_SHA256 = "a8e82b8916b9d45d01e7736f53fd82c86d0fb839907b86905b8aa5bf1a3b9c30"
EXPECTED_SOURCE_SANITIZED_SHA256 = "36767cf32982d788ac3c3ba1c7baef934349ed08ba40da2650126041273edba4"
EXPECTED_TARGET_SANITIZED_SHA256 = "24426506f2fba321ca1c98f593d459dca3e9e76d030c0303299eacd1ad76b29f"

WHITELIST_COLUMNS = ["uid", "image_path", "image_sha256", "capture_event_id", "location_id"]
SOURCE_WHITELIST = OUT / "SOURCE_STYLE_AUDIT_WHITELIST.csv"
TARGET_WHITELIST = OUT / "TARGET_ADAPT_STYLE_AUDIT_WHITELIST.csv"
SOURCE_EXPECTED_ROWS = 17637
TARGET_EXPECTED_ROWS = 3272
TARGET_EXPECTED_LOCATIONS = 17
TARGET_EXPECTED_EVENTS = 2373

B2_RUN = (
    ROOT
    / "project"
    / "runs"
    / "final_v2_1"
    / "protocol_b_b2_tsda_s410b_core_e71_e100_20260724_232600"
)
B2_CHECKPOINT = B2_RUN / "full_state" / "epoch_100_full_state.pt"
B2_CHECKPOINT_SHA256 = "3330a9c7efb618e56433904745441846607ddd5c7060bc9559ceb152a5e68430"
B2_EMA_CANONICAL_SHA256 = "bfbd0b37bbe5498f352838e9883844eae1dfc257401f1c9f7647636b2f6b2d0d"
B2_RAW_CANONICAL_SHA256 = "0d9c019995afc096c8a7c47a72d71d436699b2400e7a9ce2c479f5d82d2b0cc8"
B2_RUN_CONTRACT = B2_RUN / "B2_run_contract.json"
B2_RUN_CONTRACT_SHA256 = "8512d1f7a6272756d56652dfe845cdf5c1011e201ddac82a14905171f3daa8ea"

PROJECT_SCRIPTS = ROOT / "project" / "scripts"
FROZEN_B2_CODE = {
    "protocol_b_pseudo.py": (
        PROJECT_SCRIPTS / "protocol_b_pseudo.py",
        "be97be4428e8df0e6412dbe3f1ba990f7214814059fcee06a80cc1caeebc44dd",
    ),
    "protocol_b_target_schedule.py": (
        PROJECT_SCRIPTS / "protocol_b_target_schedule.py",
        "a2612a8c6c68e319a066c94cb9a34dbd201d4772a18440dc6970cccca5655101",
    ),
    "protocol_b_b2_model.py": (
        PROJECT_SCRIPTS / "protocol_b_b2_model.py",
        "42d2e9b18d49a93796e56e1d238e3651eb708829748873235446f8a33669e4b7",
    ),
    "protocol_b_full_state.py": (
        PROJECT_SCRIPTS / "protocol_b_full_state.py",
        "1c24c2bc4dad0bebf3a3c362af5cb3ddce6707100bd491c2f49538ae9658546f",
    ),
}

HPL = {
    "imgsz": 640,
    "conf": 0.65,
    "iou": 0.65,
    "max_det": 20,
    "area_min_strict": 0.001,
    "aspect_min_inclusive": 0.2,
    "aspect_max_inclusive": 5.0,
    "single_class_nc": 1,
    "required_class": 0,
    "classes_filter": None,
    "agnostic_nms": False,
    "teacher_eval": True,
    "teacher_inference_mode": True,
    "teacher_amp": True,
    "decode_policy": "cv2_then_pillow_truncated_readonly_v1",
    "letterbox": {
        "new_shape": [640, 640],
        "auto": False,
        "scale_fill": False,
        "scaleup": True,
        "center": True,
        "stride": 32,
        "padding_value": 114,
        "interpolation": "cv2.INTER_LINEAR",
        "color_order_after_format": "RGB",
    },
    "mask_rectangle_rasterization": "floor(x1/y1), ceil(x2/y2), clip to decoded extent, half-open slices",
}

BASE_DESCRIPTORS = [
    "tone_ln_y_q10",
    "tone_ln_y_q50",
    "tone_ln_y_q90",
    "contrast_global_q90_minus_q10",
    "contrast_local_grid8_median_q90_minus_q10",
    "chroma_cstar_q50",
    "chroma_cstar_q90",
]
AXES = {
    "TONE": BASE_DESCRIPTORS[0:3],
    "CONTRAST": BASE_DESCRIPTORS[3:5],
    "CHROMA": BASE_DESCRIPTORS[5:7],
}
AXIS_ORDER = ["TONE", "CONTRAST", "CHROMA"]
STATES = ["WHOLE", "HPL_MASK", "RANDOM_MASK"]

LUMA_COEFFICIENTS = [0.2126, 0.7152, 0.0722]
LUMA_LOG_EPSILON = 1e-6
SCALE_EPSILON = 1e-6
DEGENERATE_IQR_THRESHOLD = 1e-8
PIXEL_QUANTILE_ALGORITHM = "inverse empirical CDF; no interpolation"
EVENT_QUANTILE_ALGORITHM = "inverse empirical CDF; no interpolation"

SRGB_TO_XYZ_D65 = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
]
D65_REFERENCE_WHITE = [0.95047, 1.0, 1.08883]
LAB_REFERENCE_TOLERANCE_XYZ = 1e-7
LAB_REFERENCE_TOLERANCE_LAB = 1e-4
LAB_REFERENCE_VECTORS = [
    {"name": "black", "rgb": [0.0, 0.0, 0.0], "xyz": [0.0, 0.0, 0.0], "lab": [0.0, 0.0, 0.0]},
    {"name": "white", "rgb": [1.0, 1.0, 1.0], "xyz": [0.95047, 1.0, 1.08883], "lab": [100.0, 0.0, 0.0]},
    {
        "name": "red",
        "rgb": [1.0, 0.0, 0.0],
        "xyz": [0.4124564, 0.2126729, 0.0193339],
        "lab": [53.24079414, 80.09245960, 67.20319652],
    },
    {
        "name": "green",
        "rgb": [0.0, 1.0, 0.0],
        "xyz": [0.3575761, 0.7151522, 0.1191920],
        "lab": [87.73472235, -86.18271642, 83.17932050],
    },
    {
        "name": "blue",
        "rgb": [0.0, 0.0, 1.0],
        "xyz": [0.1804375, 0.0721750, 0.9503041],
        "lab": [32.29701093, 79.18751985, -107.86016175],
    },
]

B_GATE1_BOOT = 2000
B_NULL = 2000
B_FG_BOOT = 2000
B_GMM = 500
B_DCOR_PERM = 2000
B_LOCATION_PERM = 10000
B_PAIR_BOOT = 2000
WITHIN_EVENT_MAX_PAIRS = 4
MIN_WITHIN_EVENT_EVENTS = 100
MIN_FOREGROUND_COMPLETE_CASE_EVENT_FRACTION = 0.80
FOREGROUND_ALPHA_AXIS = 0.05 / 3.0
COMPATIBLE_REFERENCES_PER_ANCHOR = 8

GMM_PARAMETERS = {
    "K_candidates": [1, 2, 3],
    "covariance_type": "full",
    "init_params": "kmeans",
    "n_init": 20,
    "max_iter": 500,
    "tol": 1e-4,
    "reg_covar": 1e-6,
    "random_state": 20260808,
    "bootstrap_success_requires_all_three_candidates_converged": True,
}
ROUTE_B_THRESHOLDS = {
    "gmm_success_rate_min": 0.95,
    "k_ge_2_frequency_min": 0.80,
    "median_ari_min": 0.75,
    "mode_event_fraction_min": 0.10,
    "mode_event_count_min": 100,
    "mode_location_quantile": 0.99,
}

RNG_NAMESPACES = [
    "M1_Q2_WITHIN_EVENT_PAIR_V1",
    "M1_Q2_PSEUDODOMAIN_NULL_V1",
    "M1_Q2_GMM_BOOTSTRAP_V1",
    "M1_Q2_DCOR_PERM_V1",
    "M1_Q2_LOCATION_PERM_V1",
    "M1_Q2_COMPATIBLE_PAIR_V1",
    "M1_Q2_PAIR_BOOTSTRAP_V1",
    "M1_Q2_RANDOM_MASK_V1",
]
RNG_SUBKEYS = {
    "gate1_event_bootstrap": "M1_Q2_WITHIN_EVENT_PAIR_V1|GATE1_EVENT_BOOTSTRAP",
    "foreground_event_bootstrap": "M1_Q2_RANDOM_MASK_V1|FOREGROUND_EVENT_BOOTSTRAP",
}

DEPENDENCY_FREEZE = {
    "dcor": "0.6",
    "scikit-learn": "1.7.1",
    "pyarrow": "20.0.0",
    "dcor_function": "dcor.distance_correlation",
    "dcor_estimator": "package default biased distance correlation",
    "dcor_known_values": {
        "identical_nonconstant": 1.0,
        "constant_argument": 0.0,
        "x_0_1_2_3_y_0_1_4_9": 0.9684641640120555,
        "absolute_tolerance": 1e-12,
    },
}

IMPLEMENTATION_DECISIONS_FROZEN_BEFORE_DATA_RUN = {
    "all_descriptor_pixel_quantiles": PIXEL_QUANTILE_ALGORITHM,
    "all_event_and_bootstrap_quantiles": EVENT_QUANTILE_ALGORITHM,
    "within_event_pair_selection": "rank all unordered UID pairs by stateless hash; retain first four",
    "gate1_bootstrap_subkey": RNG_SUBKEYS["gate1_event_bootstrap"],
    "foreground_bootstrap_subkey": RNG_SUBKEYS["foreground_event_bootstrap"],
    "location_event_sensitivity_weight": "each location total 1/L; events within location equal",
    "image_weighted_sensitivity": "each image equal within domain",
    "hpl_mask_gate2_threshold": (
        "recompute same B_NULL location-stratified S-S/T-T cross-axis max-null on paired complete-case "
        "HPL-mask event prototypes; original WHOLE Gate2 status is not refit"
    ),
    "matched_random_equal_change": "Delta_RAND >= Delta_FG",
    "gmm_bootstrap_failure": "any K candidate exception or nonconvergence makes replicate failed",
    "gmm_bic_tie": "strict numerical equality selects smaller K",
    "foreground_bootstrap": "independent event resampling by domain; paired WHOLE/HPL/RANDOM states share counts",
    "compatible_reference_sampler": (
        "stateless hash-index draws with deterministic rejection of self/duplicates; fixed eight; no O(N^2) pair enumeration"
    ),
    "location_loo_source_pseudonull_reuse": (
        "Source split assignments/statistics are reused because Source data and stateless Source keys are unchanged"
    ),
}


def audit_contract() -> dict:
    return {
        "experiment": TASK_NAME,
        "status": "FROZEN_BEFORE_IMAGE_READ_OR_STATISTICAL_RUN",
        "HORIZON": 100,
        "HORIZON_SEARCH_PERMANENTLY_CLOSED": True,
        "scientific_question": (
            "Does Target Adapt exhibit stable and cross-domain-relevant photometric support that remains robust "
            "under the preregistered correlation, location, and foreground sensitivity analyses?"
        ),
        "scientific_design": "FROZEN",
        "statistical_implementation": "FROZEN",
        "axes": AXIS_ORDER,
        "base_descriptors": BASE_DESCRIPTORS,
        "states": STATES,
        "luma": {
            "term": "sRGB-domain luma proxy",
            "coefficients": LUMA_COEFFICIENTS,
            "natural_log": True,
            "epsilon": LUMA_LOG_EPSILON,
        },
        "lab": {
            "path": "sRGB inverse companding -> linear RGB -> XYZ D65 -> CIELab D65",
            "matrix": SRGB_TO_XYZ_D65,
            "reference_white": D65_REFERENCE_WHITE,
            "reference_vectors": LAB_REFERENCE_VECTORS,
            "xyz_tolerance": LAB_REFERENCE_TOLERANCE_XYZ,
            "lab_tolerance": LAB_REFERENCE_TOLERANCE_LAB,
            "library": "NumPy explicit implementation",
        },
        "quantiles": {
            "pixel": PIXEL_QUANTILE_ALGORITHM,
            "weighted_event": EVENT_QUANTILE_ALGORITHM,
        },
        "standardization": {
            "domain_weights": {"Source": 0.5, "Target Adapt": 0.5},
            "scale_epsilon": SCALE_EPSILON,
            "degenerate_threshold": DEGENERATE_IQR_THRESHOLD,
            "full_data_fit_once": True,
            "refit_in_loo_or_mask": False,
        },
        "bootstrap_and_permutation_counts": {
            "B_GATE1_BOOT": B_GATE1_BOOT,
            "B_NULL": B_NULL,
            "B_FG_BOOT": B_FG_BOOT,
            "B_GMM": B_GMM,
            "B_DCOR_PERM": B_DCOR_PERM,
            "B_LOCATION_PERM": B_LOCATION_PERM,
            "B_PAIR_BOOT": B_PAIR_BOOT,
        },
        "foreground": {
            "complete_case_fraction_min": MIN_FOREGROUND_COMPLETE_CASE_EVENT_FRACTION,
            "bonferroni_axes": 3,
            "one_sided_alpha_axis": FOREGROUND_ALPHA_AXIS,
            "masked_pixels": "excluded, never filled",
        },
        "gmm": GMM_PARAMETERS,
        "route_b_thresholds": ROUTE_B_THRESHOLDS,
        "rng_namespaces": RNG_NAMESPACES,
        "rng_subkeys": RNG_SUBKEYS,
        "dependency_freeze": DEPENDENCY_FREEZE,
        "hpl": HPL,
        "implementation_decisions_frozen_before_data_run": IMPLEMENTATION_DECISIONS_FROZEN_BEFORE_DATA_RUN,
        "result_space": ["A", "B", "C", "D", "E", "STOP_*"],
        "training_authorized": False,
        "augmentation_authorized": False,
        "tsva_implementation_authorized": False,
    }
