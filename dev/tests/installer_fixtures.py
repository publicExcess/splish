"""What the installer tests share: model and draft repositories, selections,
Hub errors, and FakeHub, the stand-in for the Hugging Face Hub that the
upstream and GGUF tests install from. The legacy package tests
(test_models.py) mock huggingface_hub's functions directly: the frozen legacy
installer calls them with other arguments (token, repo_type,
force_download, and model_info without a timeout)."""

import hashlib
import json
import shutil
from types import SimpleNamespace
from unittest import mock

import httpx
import numpy as np
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import RepoSibling

from dev.tests import fixture_files
from install import families, hub, models

DENSE = families.named("Qwen3.8-27B")
MOE = families.named("Qwen3.6-35B-A3B")
MODEL = "mlx-community/Qwen3.8-27B-4bit"
# The commit the main branch of every family's draft repository names.
DRAFT_COMMIT = "d" * 40
# The image preprocessing Splash implements (server/images.py).
PROCESSOR = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5] * 3,
    "image_std": [0.5] * 3,
}


def text_config(family, **changes):
    return dict(family.signature) | changes


def mlx_target(root, family, *, changes=None):
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "text_config": text_config(family, **(changes or {})),
        "quantization": {"bits": 4, "group_size": 64},
    }
    (root / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (root / name).write_text("{}")
    return root


def draft_dir(root, family, **changes):
    """A DFlash2 release of family's draft: its configuration, stating the
    draft signature with changes (dotted keys), and weights."""
    root.mkdir(parents=True, exist_ok=True)
    config = {}
    for key, value in (dict(family.draft.signature) | changes).items():
        *objects, name = key.split(".")
        node = config
        for part in objects:
            node = node.setdefault(part, {})
        node[name] = list(value) if isinstance(value, tuple) else value
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors").write_bytes(b"draft")
    return root


def selection(root, model=MODEL, **options):
    """model's selection under root/models, text only unless options say."""
    return models.Selection.of(
        root / "models",
        model,
        revision=options.get("revision"),
        language_only=options.get("language_only", True),
        draft_model=options.get("draft_model"),
    )


def http_error(status):
    request = httpx.Request("GET", "https://huggingface.co/api/models/owner/model")
    return HfHubHTTPError(
        f"{status} Client Error.\n\nRevision Not Found for url: {request.url}.",
        response=httpx.Response(status, request=request),
    )


def listing(directory):
    return sorted(
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    )


class FakeHub:
    """The Hub as huggingface_hub presents it to the installer: repositories
    whose branches name commits, and downloads that fill the test's Hub cache
    with snapshots of only the files requested. It patches huggingface_hub for
    the test's duration."""

    def __init__(self, test, cache):
        self.cache = cache
        self.remote = cache.parent / "remote"
        self.branches = {}
        # What the next Hub request raises: resolution, or downloads.
        self.failure = self.download_failure = None
        self.requests, self.downloads, self.range_reads = [], [], []
        for name, replacement in (
            ("huggingface_hub.constants.HF_HUB_CACHE", str(cache)),
            ("huggingface_hub.constants.HF_HUB_OFFLINE", False),
            ("huggingface_hub.get_token", lambda: None),
            ("huggingface_hub.HfApi", lambda **options: self),
            ("huggingface_hub.snapshot_download", self.snapshot_download),
            ("huggingface_hub.hf_hub_download", self.hf_hub_download),
            ("huggingface_hub.HfFileSystem", lambda: self),
            ("huggingface_hub.try_to_load_from_cache", self.try_to_load_from_cache),
        ):
            patch = mock.patch(name, replacement)
            patch.start()
            test.addCleanup(patch.stop)

    def publish(self, repo_id, commit, build, branch="main"):
        build(self.remote / repo_id / commit)
        self.branches[repo_id, branch] = commit

    def snapshot(self, repo_id, commit):
        return self.cache / hub.folder_name(repo_id) / "snapshots" / commit

    def model_info(self, repo_id, *, revision=None, files_metadata, timeout):
        self.requests.append((repo_id, revision))
        assert files_metadata and timeout == hub.HUB_TIMEOUT
        if self.failure:
            raise self.failure
        commit = revision
        if not models.is_hex_digest(revision, 40):
            commit = self.branches.get((repo_id, revision or "main"))
        root = self.remote / repo_id / str(commit)
        if not root.is_dir():
            raise http_error(404)
        return SimpleNamespace(
            sha=commit,
            siblings=[
                RepoSibling(
                    rfilename=name,
                    size=(root / name).stat().st_size,
                    blob_id=hashlib.sha1((root / name).read_bytes()).hexdigest(),
                )
                for name in listing(root)
            ],
        )

    def fetch(self, repo_id, name, revision):
        if self.download_failure:
            raise self.download_failure
        path = self.snapshot(repo_id, revision) / name
        # As huggingface_hub does, a cached file is returned as it is.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.remote / repo_id / revision / name, path)
            self.downloads.append(f"{repo_id}/{name}")
        return path

    def snapshot_download(self, repo_id, *, revision, allow_patterns, max_workers):
        for name in allow_patterns:
            self.fetch(repo_id, name, revision)
        return str(self.snapshot(repo_id, revision))

    def hf_hub_download(self, repo_id, filename, *, revision):
        return str(self.fetch(repo_id, filename, revision))

    def try_to_load_from_cache(self, repo_id, filename, *, revision):
        path = self.snapshot(repo_id, revision) / filename
        return str(path) if path.is_file() else None

    def open(self, path, mode, *, revision, block_size):
        owner, name, filename = path.split("/", 2)
        self.range_reads.append(path)
        return (self.remote / owner / name / revision / filename).open(mode)


def fake_hub(test, cache, *, target=DENSE, commit="a" * 40):
    """A FakeHub publishing MODEL at commit on main and every family's
    draft repository at DRAFT_COMMIT on main."""
    fake = FakeHub(test, cache)
    fake.publish(MODEL, commit, lambda p: mlx_target(p, target))
    for family in families.FAMILIES:
        fake.publish(family.draft.repo, DRAFT_COMMIT, lambda p: draft_dir(p, family))
    return fake


def cached_snapshot(cache, repo_id, commit, build):
    """repo_id at commit as a download leaves it in the Hub cache: a snapshot
    of only the files downloaded."""
    return build(cache / hub.folder_name(repo_id) / "snapshots" / commit)


def pins(cache):
    return sorted(ref.name for ref in cache.glob("*/refs/splash/*/*"))


# Qwen3.8-27B's base vision tower (families.BASE_VISION), which the tests
# publish at its pinned commit.
BASE = families.BASE_VISION[DENSE.name]
# A tower of one block in the layout the native loader binds
# (VisionLoader.cpp), small enough to compare value by value.
VISION_CONFIG = {
    "depth": 1,
    "hidden_size": 8,
    "num_heads": 2,
    "intermediate_size": 12,
    "out_hidden_size": 16,
    "patch_size": 2,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "num_position_embeddings": 4,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [],
}
# Each tower tensor's MLX name (without vision_tower.) and GGUF name, as
# VisionLoader.cpp binds them; the patch embedding is split per frame.
TOWER_NAMES = {
    "patch_embed.proj.weight": ("v.patch_embd.weight", "v.patch_embd.weight.1"),
    "patch_embed.proj.bias": ("v.patch_embd.bias",),
    "pos_embed.weight": ("v.position_embd.weight",),
    **{
        f"{mlx}.{kind}": (f"{name}.{kind}",)
        for mlx, name in (
            ("blocks.0.norm1", "v.blk.0.ln1"),
            ("blocks.0.attn.qkv", "v.blk.0.attn_qkv"),
            ("blocks.0.attn.proj", "v.blk.0.attn_out"),
            ("blocks.0.norm2", "v.blk.0.ln2"),
            ("blocks.0.mlp.linear_fc1", "v.blk.0.ffn_up"),
            ("blocks.0.mlp.linear_fc2", "v.blk.0.ffn_down"),
            ("merger.norm", "v.post_ln"),
            ("merger.linear_fc1", "mm.0"),
            ("merger.linear_fc2", "mm.2"),
        )
        for kind in ("weight", "bias")
    },
}


def tower(seed=0):
    """The base tower's tensors by MLX name, as float32 arrays of BF16
    values. Each one's first value is one F16 holds only as a subnormal."""
    c = VISION_CONFIG
    h, p, i = c["hidden_size"], c["patch_size"], c["intermediate_size"]
    merged = h * c["spatial_merge_size"] ** 2
    shapes = {
        "patch_embed.proj.weight": (h, 2, p, p, 3),
        "patch_embed.proj.bias": (h,),
        "pos_embed.weight": (c["num_position_embeddings"], h),
        "blocks.0.norm1.weight": (h,),
        "blocks.0.norm1.bias": (h,),
        "blocks.0.attn.qkv.weight": (3 * h, h),
        "blocks.0.attn.qkv.bias": (3 * h,),
        "blocks.0.attn.proj.weight": (h, h),
        "blocks.0.attn.proj.bias": (h,),
        "blocks.0.norm2.weight": (h,),
        "blocks.0.norm2.bias": (h,),
        "blocks.0.mlp.linear_fc1.weight": (i, h),
        "blocks.0.mlp.linear_fc1.bias": (i,),
        "blocks.0.mlp.linear_fc2.weight": (h, i),
        "blocks.0.mlp.linear_fc2.bias": (h,),
        "merger.norm.weight": (h,),
        "merger.norm.bias": (h,),
        "merger.linear_fc1.weight": (merged, merged),
        "merger.linear_fc1.bias": (merged,),
        "merger.linear_fc2.weight": (c["out_hidden_size"], merged),
        "merger.linear_fc2.bias": (c["out_hidden_size"],),
    }
    assert shapes.keys() == TOWER_NAMES.keys()
    rng = np.random.default_rng(seed)
    tensors = {}
    for name, shape in shapes.items():
        values = rng.normal(0, 0.05, shape).astype(np.float32)
        values.flat[0] = 3e-6
        bf16 = values.view(np.uint32) & np.uint32(0xFFFF0000)
        tensors["vision_tower." + name] = bf16.view(np.float32)
    return tensors


def base_vision_repo(root, tensors):
    """The base tower's repository: config.json stating VISION_CONFIG, the
    processor configuration and BASE.shard, holding the tower in BF16 beside
    a language model tensor."""
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "text_config": text_config(DENSE),
        "vision_config": VISION_CONFIG,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "preprocessor_config.json").write_text(json.dumps(PROCESSOR))
    shard = {
        name: (
            list(values.shape),
            "BF16",
            (values.view(np.uint32) >> 16).astype("<u2").tobytes(),
        )
        for name, values in tensors.items()
    }
    shard["language_model.lm_head.weight"] = ([1], "U8", b"\0")
    fixture_files.write_safetensors(root / BASE.shard, shard)
    return root


def projector(path, tensors, *, flush=False):
    """An mmproj of tensors as llama.cpp converts a tower: weights F16,
    biases and norms F32, the patch embedding one [output, channel, row,
    column] F16 tensor per frame. flush rounds F16 subnormals to zero, as a
    converter may."""
    c = VISION_CONFIG
    grid = int(c["num_position_embeddings"] ** 0.5)
    metadata = {
        "general.architecture": "clip",
        "clip.projector_type": "qwen3vl_merger",
        "clip.use_gelu": True,
        "clip.vision.block_count": c["depth"],
        "clip.vision.embedding_length": c["hidden_size"],
        "clip.vision.attention.head_count": c["num_heads"],
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.feed_forward_length": c["intermediate_size"],
        "clip.vision.projection_dim": c["out_hidden_size"],
        "clip.vision.patch_size": c["patch_size"],
        "clip.vision.spatial_merge_size": c["spatial_merge_size"],
        "clip.vision.image_size": grid * c["patch_size"],
        "clip.vision.is_deepstack_layers": [False] * c["depth"],
        "clip.vision.image_mean": [0.5] * 3,
        "clip.vision.image_std": [0.5] * 3,
    }
    table = []
    for name, names in TOWER_NAMES.items():
        values = tensors["vision_tower." + name]
        frames = (
            [values[:, frame].transpose(0, 3, 1, 2) for frame in (0, 1)]
            if len(names) == 2
            else [values]
        )
        for gguf_name, frame in zip(names, frames, strict=True):
            if frame.ndim > 1:
                data = np.ascontiguousarray(frame).astype("<f2")
                if flush:
                    data[np.abs(data) < 2.0**-14] = 0
                kind = 1
            else:
                data, kind = frame.astype("<f4"), 0
            table.append((gguf_name, list(reversed(frame.shape)), kind, data.tobytes()))
    return fixture_files.write_gguf(path, metadata, table)
