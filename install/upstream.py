"""Install supported upstream models: a target repository and the draft
trained for its architecture, as an assembly (assembly.py).

The Hub owns downloads and snapshots; this module inspects the target,
selects its components and decides when to follow the Hub, and the native
source adapters own tensor validation and preparation. A target is
identified by its own metadata (a GGUF header or an MLX config), read before
any weight download, and paired with the draft trained for its family
(families.py); repository names play no part. Every start follows the
target's revision, then its draft's, with one Hub request each, and
publishes a new commit's assembly atomically; the installed assembly starts
when the Hub cannot answer or the new commit cannot be installed.
"""

from __future__ import annotations

import json
import math
import os
import struct
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

if __package__:
    from . import assembly, families, gguf, hub, legacy, models
else:
    import assembly
    import families
    import gguf
    import hub
    import legacy
    import models

# The tokenizer files an MLX target may supply, linked when present.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
)


@dataclass(frozen=True)
class Target:
    """What the target repository supplies, known before any weight download."""

    # The record's target_format: "mlx-affine" or "gguf".
    format: str
    # The record's vision_format: "none", "safetensors" or "gguf".
    vision_format: str
    # The configuration that identifies the target's family.
    config: dict
    # Assembly path -> repository file, for every path a repository file
    # serves (the layout in assembly.py). A GGUF's configuration and tokenizer
    # are derived from it at installation instead.
    files: dict[str, str]
    # The family's base vision tower (families.BASE_VISION), when the target
    # ships none Splash can use: its repository at the pinned commit, and the
    # assembly paths its files serve.
    base: hub.Repository | None = None
    base_files: dict[str, str] = field(default_factory=dict)


def _root_ggufs(files):
    """The GGUF files at a repository's root, named *.gguf as the native
    loader finds a target; subfolders (split BF16, MTP heads) never count."""
    return sorted(n for n in files if "/" not in n and n.endswith(".gguf"))


def _projector_named(name):
    """Whether a root GGUF is named as a vision projector: mmproj-*.gguf, or
    MODEL-mmproj-*.gguf as Prism ML names theirs."""
    return "mmproj" in Path(name).stem.lower()


def select_gguf(files, variant):
    """The target GGUF among a repository's root files, and whether it was
    taken by its -VARIANT ending from several, which the caller reports.
    :VARIANT names the file whose name is the model name all of them share,
    then -VARIANT (X-Q4_K_M for Q4_K_M, not X-UD-Q4_K_M); failing that, the
    only file whose name ends in -VARIANT. A repository of one GGUF names no
    variant apart from its model, so its one match needs no report. Without
    :VARIANT, the only target GGUF. Anything else is an error listing the
    candidates."""
    candidates = [n for n in _root_ggufs(files) if not _projector_named(n)]
    if variant is None:
        if len(candidates) == 1:
            return candidates[0], False
        choice = "select a GGUF with OWNER/REPO:VARIANT"
    else:
        parts = [Path(name).stem.split("-") for name in candidates]
        shared = len(os.path.commonprefix(parts))
        exact = [
            name
            for name, words in zip(candidates, parts, strict=True)
            if "-".join(words[shared:]).lower() == variant.lower()
        ]
        if len(exact) == 1:
            return exact[0], False
        suffix = "-" + variant.lower()
        ending = [n for n in candidates if Path(n).stem.lower().endswith(suffix)]
        if len(ending) == 1:
            return ending[0], len(candidates) > 1
        choice = "no single GGUF matches :" + variant
    listed = ", ".join(candidates) or "none"
    raise models.ModelError(f"{choice} (files in the repository root: {listed})")


class NoUsableProjector(models.ModelError):
    """A GGUF repository without a BF16 or F32 vision projector; projectors
    lists each root GGUF named mmproj: (name, header, architecture, the type
    names of its tensors)."""

    def __init__(self, message, projectors):
        super().__init__(message)
        self.projectors = projectors


def select_vision(repo):
    """The name and header of the GGUF repository's vision projector, chosen
    by content among its root GGUF files named mmproj, whatever the publisher
    calls them: a clip model whose weights are BF16, or F32, which
    preparation converts only where every value is exact; BF16 is preferred.
    The tower runs in BF16 and preparation never rounds a weight: F16 has a
    narrower exponent than BF16, so an F16 projector has already rounded
    small weights, as a quantized one has. Each header costs a few range
    requests."""
    usable, found, named = {"BF16": [], "F32": []}, [], []
    for name in filter(_projector_named, _root_ggufs(repo.files)):
        with repo.open(name) as stream:
            header = gguf.Metadata(stream, tensors=True)
        architecture = header.values.get("general.architecture")
        types = {
            gguf.TENSOR_TYPES.get(kind, f"type {kind}")
            for kind in header.tensors.values()
        }
        named.append((name, header, architecture, types))
        found.append(f"{name} ({architecture}: {', '.join(sorted(types))})")
        if architecture == "clip" and types and types <= {"BF16", "F32"}:
            usable["BF16" if "BF16" in types else "F32"].append((name, header))
    for precision, projectors in usable.items():
        if len(projectors) == 1:
            return projectors[0]
        if projectors:
            raise models.ModelError(
                f"several {precision} vision projectors, "
                + ", ".join(name for name, _ in projectors)
                + ", describe no single tower; use --language-only to serve text only"
            )
    raise NoUsableProjector(
        "the GGUF repository has no BF16 or F32 vision projector ("
        + ("; ".join(found) or "no GGUF named mmproj")
        + "); use --language-only to serve text only",
        named,
    )


def inspect_target(repo, variant, language_only):
    """What the target repository supplies, from metadata alone: a GGUF
    header read by range requests, or an MLX config and shard index. A
    :VARIANT names a GGUF, and so does a repository without a safetensors
    checkpoint."""
    if variant is not None or not any(n.endswith(".safetensors") for n in repo.files):
        return _gguf_target(repo, variant, language_only)
    return _mlx_target(repo, language_only)


def _gguf_target(repo, variant, language_only):
    name, by_ending = select_gguf(repo.files, variant)
    if by_ending:
        print(
            f"No GGUF is named for :{variant} alone; using {name}, the only one "
            f"whose name ends in -{variant}.",
            flush=True,
        )
    with repo.open(name) as stream:
        header = gguf.Metadata(stream, tensors=True)
    # The family bounds the layers whose tensors the screening lists.
    families.family_for(gguf.model_config(header))
    gguf.require_loadable(header)
    files = {"target/" + name: name}
    vision_header = None
    if not language_only:
        try:
            files[assembly.GGUF_VISION], vision_header = select_vision(repo)
        except NoUsableProjector as error:
            config = gguf.model_config(header)
            base, base_files = _gguf_base_vision(repo, config, error)
            print(f"Selected {name} from {repo.name}.", flush=True)
            return Target("gguf", "safetensors", config, files, base, base_files)
        _validate_processor(gguf.processor_config(vision_header))
    config = gguf.model_config(header, vision_header)
    print(f"Selected {name} from {repo.name}.", flush=True)
    return Target("gguf", "none" if language_only else "gguf", config, files)


def _mlx_target(repo, language_only):
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    if not language_only:
        required.add("preprocessor_config.json")
    if missing := required - repo.files:
        # --language-only drops the processor requirement and no other.
        hint = (
            "; use --language-only to serve text only"
            if missing == {"preprocessor_config.json"}
            else ""
        )
        raise models.ModelError(
            f"target repository {repo.name} is missing: {', '.join(sorted(missing))}. "
            "Configuration, tokenizer and processor must come from the target "
            f"repository{hint}."
        )
    config = models.read_json(repo.file("config.json"))
    # MLX states its quantization under "quantization"; a transformers
    # quantization_config alone describes another method (GPTQ, AWQ, ...).
    quant = config.get("quantization")
    if (
        not isinstance(quant, dict)
        or quant.get("mode", "affine") != "affine"
        or quant.get("bits") != 4
        or quant.get("group_size") != 64
    ):
        raise models.ModelError(
            "this model requires an MLX affine 4-bit/group-64 checkpoint or a supported GGUF"
        )
    files = {
        path: "config.json"
        for path in ("config.json", "target/config.json", "tokenizer/config.json")
    }
    files |= {"tokenizer/" + n: n for n in TOKENIZER_FILES if n in repo.files}
    if not language_only:
        _validate_processor(models.read_json(repo.file("preprocessor_config.json")))
        shards = _weight_files(repo, "vision_tower.")
        if not shards:
            no_tower = f"{repo.name} has no vision tower"
            base, base_files = _mlx_base_vision(repo, config, no_tower)
            files |= {"target/" + n: n for n in _weight_files(repo)}
            return Target("mlx-affine", "safetensors", config, files, base, base_files)
        files["vision/config.json"] = "config.json"
        files |= {"vision/" + n: n for n in shards}
    files |= {"target/" + n: n for n in _weight_files(repo)}
    return Target(
        "mlx-affine", "none" if language_only else "safetensors", config, files
    )


def _weight_files(repo, prefix=""):
    """The checkpoint's shards holding a tensor whose name starts with prefix."""
    if "model.safetensors.index.json" in repo.files:
        index = models.read_json(repo.file("model.safetensors.index.json"))
        weights = index.get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise models.ModelError("invalid safetensors shard index")
        if not all(isinstance(name, str) for name in weights.values()):
            raise models.ModelError("invalid safetensors shard filename")
        names = {file for tensor, file in weights.items() if tensor.startswith(prefix)}
    elif "model.safetensors" in repo.files:
        # One file: its header says whether it holds such a tensor.
        holds = not prefix or any(
            tensor.startswith(prefix)
            for tensor in _safetensors_tensors(repo, "model.safetensors")
        )
        names = {"model.safetensors"} if holds else set()
    else:
        raise models.ModelError("model has no safetensors checkpoint")
    if not all(
        name in repo.files and "/" not in name and name.endswith(".safetensors")
        for name in names
    ):
        raise models.ModelError("checkpoint has missing or unsupported shards")
    return names


def _safetensors_tensors(repo, name):
    """The tensor names in a safetensors file's header, read on demand,
    without a download."""
    with repo.open(name) as stream:
        header, _ = _safetensors_header(stream, name)
    return set(header) - {"__metadata__"}


def _safetensors_header(stream, name):
    """A safetensors header, its length (8 bytes, little-endian) then a JSON
    object, and the offset of the data its offsets count from."""
    size = int.from_bytes(stream.read(8), "little")
    # The native checkpoint reader's bound on one header.
    if not 2 <= size <= 1 << 20:
        raise models.ModelError(f"invalid safetensors header in {name}")
    try:
        header = json.loads(stream.read(size))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise models.ModelError(f"invalid safetensors header in {name}") from error
    if not isinstance(header, dict):
        raise models.ModelError(f"invalid safetensors header in {name}")
    return header, 8 + size


def _validate_processor(config):
    """Splash prepares images one way (server/images.py): 16-pixel patches in
    two temporal slices, merged 2x2 and normalized to [-1, 1]. A processor
    configuration asking for another is rejected before any download; it is
    not installed, since nothing reads it."""
    expected = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise models.ModelError("unsupported vision preprocessing configuration")


# The vision_config fields the native model inspection reads
# (ModelDescriptor.mm), which a target's must state as its base tower's do.
VISION_CONFIG_KEYS = (
    "depth",
    "hidden_size",
    "num_heads",
    "intermediate_size",
    "out_hidden_size",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "in_channels",
    "num_position_embeddings",
    "hidden_act",
    "deepstack_visual_indexes",
)
# GGML tensor types an unquantized projector stores, compared with the base.
F32, F16, BF16 = 0, 1, 30
# F16's smallest normal and largest finite magnitudes: below the first, F16
# underflows to subnormals or zero; beyond the second, it overflows.
F16_NORMAL, F16_MAX = 2.0**-14, 65504.0


def _base_tower(config):
    """The family and base vision tower (families.BASE_VISION) of a target's
    configuration, or None: no base is known for another family, or for an
    architecture no family has."""
    try:
        family = families.family_for(config)
    except models.ModelError:
        return None
    base = families.BASE_VISION.get(family.name)
    return base and (family.name, base)


def _base_repository(base):
    """The base tower's repository at its pinned commit: its cached snapshot
    when that holds every file used, without a Hub request, else as the Hub
    lists it. Its processor configuration is checked as a target's is."""
    used = {"config.json", "preprocessor_config.json", base.shard}
    repo = None
    if hub.snapshot(base.repo, base.revision).is_dir():
        repo = hub.Repository.cached(base.repo, base.revision)
    if repo is None or not used <= repo.files:
        repo = hub.Repository.resolve(base.repo, base.revision)
    _validate_processor(models.read_json(repo.file("preprocessor_config.json")))
    return repo


def _base_files(base):
    """Assembly path -> base repository file: its configuration and the shard
    holding the tower, served as an MLX tower is."""
    return {"vision/config.json": "config.json", "vision/" + base.shard: base.shard}


def _use_base(found, base, family):
    """Report, on one line, that the base tower serves the target's images."""
    print(
        f"vision: {found}; using {family}'s base tower "
        f"({base.repo}@{base.revision[:7]}). "
        "Use --language-only to serve without vision.",
        flush=True,
    )


def _mlx_base_vision(repo, config, no_tower):
    """The base tower of an MLX target that ships none, whose configuration
    states the base tower's vision_config; else the error of a target without
    a tower."""
    hint = "; use --language-only to serve text only"
    if (known := _base_tower(config)) is None:
        raise models.ModelError(no_tower + hint)
    family, base = known
    base_repo = _base_repository(base)
    expected = models.read_json(base_repo.file("config.json")).get("vision_config")
    stated = config.get("vision_config")
    if not isinstance(stated, dict) or not isinstance(expected, dict):
        raise models.ModelError(f"{no_tower}, and states no vision_config{hint}")
    if differences := [
        key for key in VISION_CONFIG_KEYS if stated.get(key) != expected.get(key)
    ]:
        raise models.ModelError(
            f"{no_tower}, and its vision_config is not {family}'s "
            f"({', '.join(differences)} differ){hint}"
        )
    _use_base(f"{repo.name} ships no vision tower", base, family)
    return base_repo, _base_files(base)


def _gguf_base_vision(repo, config, error):
    """The base tower of a GGUF target whose repository ships no vision
    projector, or only unquantized ones (F16 among them) that each equal the
    base tower within F16 rounding (_projector_difference); else error, the
    reason selection found no projector to use."""
    if (known := _base_tower(config)) is None:
        raise error
    family, base = known
    if not error.projectors:
        base_repo = _base_repository(base)
        _use_base(f"{repo.name} ships no vision tower", base, family)
        return base_repo, _base_files(base)
    candidates = [
        name
        for name, _, architecture, types in error.projectors
        if architecture == "clip" and types and types <= {"F16", "F32", "BF16"}
    ]
    if not candidates:
        raise error
    base_repo = _base_repository(base)
    depth = models.read_json(base_repo.file("config.json"))["vision_config"]["depth"]
    hint = "; use --language-only to serve text only"
    for name in candidates:
        print(f"vision: comparing {name} with {family}'s base tower.", flush=True)
        if difference := _projector_difference(repo, name, base_repo, base, depth):
            raise models.ModelError(
                f"{str(error).removesuffix(hint)}; {name} is not {family}'s "
                f"base tower ({difference}){hint}"
            )
    _use_base(
        f"{repo.name} ships its vision tower only as {', '.join(candidates)}, "
        f"which equals {family}'s base tower within F16 rounding",
        base,
        family,
    )
    return base_repo, _base_files(base)


def _tower_tensors(depth):
    """Each tensor of a tower of depth blocks: its MLX name and the GGUF names
    of its values, as VisionLoader.cpp binds them. The MLX patch embedding,
    one Conv3d weight, is one GGUF tensor per temporal frame."""
    modules = [
        (f"blocks.{i}.{mlx}", f"v.blk.{i}.{name}")
        for i in range(depth)
        for mlx, name in (
            ("norm1", "ln1"),
            ("attn.qkv", "attn_qkv"),
            ("attn.proj", "attn_out"),
            ("norm2", "ln2"),
            ("mlp.linear_fc1", "ffn_up"),
            ("mlp.linear_fc2", "ffn_down"),
        )
    ]
    modules += [
        ("merger.norm", "v.post_ln"),
        ("merger.linear_fc1", "mm.0"),
        ("merger.linear_fc2", "mm.2"),
    ]
    tensors = [
        ("patch_embed.proj.weight", ("v.patch_embd.weight", "v.patch_embd.weight.1")),
        ("patch_embed.proj.bias", ("v.patch_embd.bias",)),
        ("pos_embed.weight", ("v.position_embd.weight",)),
    ]
    tensors += [
        (f"{mlx}.{kind}", (f"{name}.{kind}",))
        for mlx, name in modules
        for kind in ("weight", "bias")
    ]
    return [("vision_tower." + mlx, names) for mlx, names in tensors]


def _gguf_tensor_table(stream):
    """Each tensor of the GGUF at the stream's start: name -> (shape in
    numpy's order, GGML type, offset of its data in the file), read through
    the bounded metadata reader."""
    count = struct.unpack("<Q", stream.read(16)[8:])[0]
    stream.seek(0)
    header = gguf.Metadata(stream)
    # The reader stops at the tensor table and bounds its count.
    table = {}
    for _ in range(count):
        name = header.string()
        rank = header.scalar("I")
        if rank > gguf.Metadata.MAX_DIMENSIONS or name in table:
            raise models.ModelError("invalid GGUF tensor table")
        dims = struct.unpack(f"<{rank}Q", header.read(8 * rank))
        table[name] = (tuple(reversed(dims)), header.scalar("I"), header.scalar("Q"))
    alignment = header.values.get("general.alignment", 32)
    if type(alignment) is not int or alignment <= 0:
        raise models.ModelError("invalid GGUF alignment")
    start = header.consumed + -header.consumed % alignment
    return {
        name: (shape, kind, start + at) for name, (shape, kind, at) in table.items()
    }


def _projector_difference(repo, name, base_repo, base, depth):
    """Why the projector name in repo is not the base tower, or None when it
    is: it holds exactly the tower's tensors, of the base's shapes, and each
    value is the base's BF16 value as F16 rounds it (to nearest, ties to
    even), F32 or BF16 holds it, except where F16 underflows (both below
    F16's smallest normal, of one sign or zero) or overflows (infinite or
    F16's largest finite, of one sign). Both are read tensor by tensor, from
    the Hub cache or by range requests, in the projector's order, and the
    first difference ends the comparison."""
    import numpy as np

    tensors = _tower_tensors(depth)
    with repo.open(name) as projector, base_repo.open(base.shard) as checkpoint:
        table = _gguf_tensor_table(projector)
        header, data = _safetensors_header(checkpoint, base.shard)
        expected = {gguf_name for _, names in tensors for gguf_name in names}
        if extra := sorted(set(table) - expected):
            return f"{len(extra)} tensor(s) the tower does not have, {extra[0]} first"
        if missing := sorted(expected - set(table)):
            return f"{len(missing)} tower tensor(s) missing, {missing[0]} first"
        for mlx, names in sorted(tensors, key=lambda t: table[t[1][0]][2]):
            entry = header.get(mlx)
            if not isinstance(entry, dict) or entry.get("dtype") != "BF16":
                raise models.ModelError(f"the base tower has no BF16 {mlx}")
            shape = tuple(entry["shape"])
            begin, end = entry["data_offsets"]
            if end - begin != 2 * math.prod(shape):
                raise models.ModelError(f"invalid base tower tensor {mlx}")
            checkpoint.seek(data + begin)
            bits = np.frombuffer(checkpoint.read(end - begin), "<u2")
            values = (bits.astype("<u4") << 16).view("<f4").reshape(shape)
            # The MLX patch embedding is [output, frame, row, column, channel];
            # each GGUF frame is [output, channel, row, column].
            frames = (
                [values[:, frame].transpose(0, 3, 1, 2) for frame in (0, 1)]
                if len(names) == 2
                else [values]
            )
            for gguf_name, base_values in zip(names, frames, strict=True):
                found, kind, offset = table[gguf_name]
                if found != base_values.shape:
                    return (
                        f"{gguf_name} is {list(found)}, not {list(base_values.shape)}"
                    )
                size = {F32: 4, F16: 2, BF16: 2}[kind] * math.prod(found)
                projector.seek(offset)
                raw = projector.read(size)
                if len(raw) != size:
                    raise models.ModelError(f"truncated {name}")
                if not _equals_rounded(
                    np.ascontiguousarray(base_values).ravel(), raw, kind
                ):
                    return f"{gguf_name} differs"
    return None


def _equals_rounded(values, raw, kind):
    """Whether raw, little-endian values of GGML type kind, holds values (F32,
    from BF16): exactly as F32 or BF16, or as F16 rounds them, within F16's
    underflow and overflow."""
    import numpy as np

    if kind == F32:
        return np.array_equal(np.frombuffer(raw, "<u4"), values.view("<u4"))
    if kind == BF16:
        return np.array_equal(np.frombuffer(raw, "<u2"), values.view("<u4") >> 16)
    found = np.frombuffer(raw, "<f2")
    with np.errstate(over="ignore"):
        differs = found.view("<u2") != values.astype("<f2").view("<u2")
    if not differs.any():
        return True
    base, found = values[differs], found[differs].astype("<f4")
    same_sign = (found == 0) | (np.signbit(found) == np.signbit(base))
    underflow = (np.abs(base) < F16_NORMAL) & (np.abs(found) < F16_NORMAL)
    overflow = (np.abs(base) > F16_MAX) & (np.isinf(found) | (np.abs(found) == F16_MAX))
    return bool((same_sign & (underflow | overflow)).all())


def prepare(selection):
    """Start selection's model, installing what it needs.

    Every start resolves the target's revision, then the default branch of
    its draft's repository, with one Hub request each (hub.Repository.resolve
    decides when none is made). The installed assembly of those commits
    starts; a new commit is installed and published atomically; when the Hub
    cannot answer, or a new commit cannot be installed, the verified
    installation starts instead. A legacy Splash package is installed by
    legacy.prepare."""
    kind = models.installation_kind(selection.link)
    if kind == models.PACKAGE:
        legacy.prepare(selection)
        return
    installed = None
    if kind == models.ASSEMBLY:
        try:
            installed = assembly.verify(selection.link)
        except (models.ModelError, OSError) as error:
            print(f"Reinstalling {selection.model}: {error}", flush=True)
    installed_commit = installed and installed["sources"]["target"]["revision"]
    target = hub.Repository.resolve(
        selection.repo_id,
        selection.revision,
        installation=selection.link,
        installed=installed_commit,
    )
    if installed is None and _is_legacy_package(target, selection.model):
        legacy.prepare(selection)
    elif installed is not None and target.revision == installed_commit:
        _start_installed(selection, target, installed)
    else:
        _install_commit(selection, target, installed)


def _is_legacy_package(repo, model):
    """Whether the target repository is a legacy Splash package, whose
    manifest.json names a package format."""
    if "manifest.json" not in repo.files:
        return False
    with hub.as_model_errors(f"cannot install {model}"):
        manifest = models.read_json(repo.file("manifest.json"))
    return legacy.is_package_manifest(manifest)


def _start_installed(selection, target, installed):
    """Start the verified installation of the target's commit, assembled again
    first when its draft moved or this release changes it (_changes). The
    draft's repository is asked only when the Hub answered for the target, so
    a commit revision holds its draft too. The installation's links name its
    files; only assembling it again reads the commits' snapshots in the
    current Hub cache."""
    if target.unreachable_reason:
        print(
            f"Could not reach the Hub ({target.unreachable_reason}); "
            f"using the installed {selection.repo_id}@{target.revision[:12]}.",
            flush=True,
        )
    recorded = installed["sources"]["target"]
    family = families.named(installed["family"])
    draft = family and _resolve_draft(family, selection, installed, target)
    if draft and draft.unreachable_reason:
        print(
            f"Could not reach the Hub ({draft.unreachable_reason}); "
            f"using the installed draft {_at(draft)}.",
            flush=True,
        )
    if changes := _changes(installed, family, draft):
        print(f"Updating {selection.model}: {'; '.join(changes)}.", flush=True)
        with _keeping_installation(
            selection, installed, f"update it ({'; '.join(changes)})"
        ):
            _install(selection, hub.Repository.recorded(recorded), installed, draft)
        return
    if (replaced := _retain_installed(selection)) is None:
        print(
            f"Splash model {selection.model} is already installed in {selection.link}",
            flush=True,
        )
        return
    # A concurrent installation replaced or damaged the assembly after it was
    # verified: no verified installation is left to keep.
    print(f"Reinstalling {selection.model}: {replaced}", flush=True)
    _install(selection, hub.Repository.recorded(recorded), installed, draft)


def _install_commit(selection, target, installed):
    """Install a commit of the target this selection has not installed; a
    verified installation of another commit is kept if that fails."""
    if target.unreachable_reason:
        print(
            f"Could not reach the Hub ({target.unreachable_reason}); installing "
            f"{selection.repo_id}@{target.revision[:12]} from the Hub cache.",
            flush=True,
        )
    elif installed is not None:
        print(
            f"{selection.repo_id} moved from "
            f"{installed['sources']['target']['revision'][:12]} "
            f"to {target.revision[:12]}.",
            flush=True,
        )
    if installed is None:
        _install(selection, target, None)
        return
    with _keeping_installation(
        selection, installed, f"install {selection.repo_id}@{target.revision}"
    ):
        _install(selection, target, installed)


@contextmanager
def _keeping_installation(selection, installed, attempt):
    """Run the block; if it fails, start the verified installation instead,
    with a warning naming the attempt that failed."""
    try:
        yield
    except (models.ModelError, OSError) as error:
        commit = installed["sources"]["target"]["revision"]
        models.warn(
            f"keeping the installed {selection.repo_id}@{commit[:12]}; "
            f"cannot {attempt}: {error}"
        )
        if _retain_installed(selection) is not None:
            raise


def _retain_installed(selection):
    """Verify the installation again under the installation lock, where a
    concurrent installation may have replaced it since it was read, and
    repair its pins. None when it verifies, else why it does not."""
    with models.installation_lock(selection.models_root):
        try:
            installed = assembly.verify(selection.link)
        except (models.ModelError, OSError) as error:
            return error
        hub.repair_pins(selection.link, assembly.pins(installed))
    return None


def _install(selection, repo, installed, draft=None):
    """Assemble repo's target and its draft, and publish the assembly at the
    selection link. draft is the draft's repository as this start resolved
    it, or None to resolve it here, asking the Hub only when it answered for
    repo."""
    # Every Hub request (header reads, downloads) happens here.
    with hub.as_model_errors(f"cannot install {selection.model}"):
        target = inspect_target(repo, selection.variant, selection.language_only)
        family = families.family_for(target.config)
        if draft is None:
            draft = _resolve_draft(family, selection, installed, repo)
        draft, files = _draft(family, installed, draft)
        print(
            f"Installing {selection.model} as {family.name} ({target.format}); "
            f"draft {draft.name}; "
            f"vision {'disabled' if selection.language_only else 'enabled'}.",
            flush=True,
        )
        downloaded = repo.download(set(target.files.values()))
        if target.base:
            fetched = target.base.download(set(target.base_files.values()))
            files |= {path: fetched[name] for path, name in target.base_files.items()}
    files |= {path: downloaded[name] for path, name in target.files.items()}
    record = {
        "version": 1,
        "model": selection.model,
        "family": family.name,
        "target_format": target.format,
        "vision_format": target.vision_format,
        "sources": {"target": repo.identity(), "draft": draft.identity()},
    }
    if target.base:
        record["sources"]["vision"] = target.base.identity()
    models_root = selection.models_root
    models_root.mkdir(parents=True, exist_ok=True)
    # Everything written under the models root is written under the lock.
    with models.installation_lock(models_root):
        if target.format == "gguf":
            record["metadata"], derived = assembly.derived_metadata(models_root, files)
            files |= derived
        # One record per linked file, however many assembly paths link it.
        records = {path: assembly.file_record(path) for path in set(files.values())}
        record["files"] = {name: records[path] for name, path in sorted(files.items())}
        # A new installation requires its pins before it is published; its
        # older pins are retired once it is, in the repositories the
        # assembly it replaces linked too.
        pins = [
            hub.pin(snapshot, repo_id, selection.link)
            for snapshot, repo_id in assembly.pins(record)
        ]
        replaced = [
            hub.pinned(snapshot, selection.link)
            for snapshot, _ in assembly.recorded_pins(selection.link)
        ]
        models.link_selection(
            selection.link, assembly.build(models_root, record, files)
        )
        hub.retire_other_pins(pins, replaced)
        assembly.collect_garbage(models_root)


def _changes(installed, family, draft):
    """What a start changes in a verified assembly of the target's installed
    commit: the draft, when draft, as this start resolved it, is another
    repository or commit, or the GGUF metadata, when this release derives it
    otherwise."""
    if family is None:
        return [f"no supported family is named {installed['family']}"]
    changes = []
    recorded = installed["sources"]["draft"]
    if draft.identity() != recorded:
        changes.append(
            f"{draft.name} moved from {recorded['revision'][:12]} to {draft.revision[:12]}"
            if draft.name == recorded["repo"]
            else f"its draft is now {_at(draft)}"
        )
    if installed["target_format"] == "gguf" and installed[
        "metadata"
    ] != assembly.metadata_key(installed["files"]):
        changes.append("the GGUF metadata adapter changed")
    base = families.BASE_VISION.get(family.name)
    if (vision := installed["sources"].get("vision")) and (
        base is None or base.identity() != vision
    ):
        changes.append(f"its base vision tower {vision['repo']} changed")
    return changes


def _resolve_draft(family, selection, installed, target):
    """The draft's repository, --draft-model or else the family's DFlash2
    release, at the commit its default branch names now: one Hub request, as
    for the target, made only when the Hub answered for the target. Else the
    installed draft stands in, unlisted, or without one a commit the Hub
    cache holds for this selection; the installed draft also stands in, with
    the reason, when the draft cannot be resolved."""
    name = selection.draft_model or family.draft.repo
    recorded = installed and installed["sources"]["draft"]
    asked = _answered(target)
    if recorded and not asked:
        return hub.Repository(recorded["repo"], recorded["revision"], frozenset())
    commit = recorded["revision"] if recorded and recorded["repo"] == name else None
    try:
        return hub.Repository.resolve(
            name,
            installation=selection.link,
            installed=commit,
            unreachable=None
            if asked
            else target.unreachable_reason or "the Hub did not list the target",
        )
    except models.ModelError as error:
        if not recorded:
            raise
        return hub.Repository(
            recorded["repo"],
            recorded["revision"],
            frozenset(),
            unreachable_reason=str(error),
        )


def _draft(family, installed, draft):
    """The draft repository and its downloaded files, by assembly path: draft,
    as this start resolved it, when it is not the installed draft, or else
    the installed one, which is kept too when draft cannot be fetched
    (downloaded and checked)."""
    recorded = installed and installed["sources"]["draft"]
    if draft.identity() != recorded:
        try:
            with hub.as_model_errors(f"cannot fetch the {family.name} draft"):
                return draft, _draft_files(draft, family)
        except models.ModelError as error:
            if not recorded:
                raise
            models.warn(
                f"cannot use the {family.name} draft {_at(draft)}; "
                f"keeping the installed one: {error}"
            )
    repo = hub.Repository.recorded(recorded)
    return repo, _draft_files(repo, family)


def _answered(repo):
    """Whether the Hub answered for repo this start: resolve listed it, and
    not from the Hub cache in the Hub's stead."""
    return bool(repo.files) and not repo.unreachable_reason


def _at(repo):
    """repo@commit, or a local directory's path."""
    return f"{repo.name}@{repo.revision[:12]}" if repo.revision else repo.name


def _draft_files(repo, family):
    """The family's DFlash2 checkpoint in repo, downloaded, by assembly path:
    config.json and the safetensors weights, model.safetensors or the shards
    its index names, at the root of the repository or --draft-model
    directory, as a DFlash2 release holds them. Its configuration must state
    the family's draft signature."""
    try:
        if "config.json" not in repo.files:
            raise models.ModelError("no config.json")
        weights = _weight_files(repo)
    except models.ModelError as error:
        raise models.ModelError(
            f"{repo.name} does not contain a DFlash2 checkpoint for {family.name}"
            f" ({error})"
        ) from error
    config = models.read_json(repo.file("config.json"))
    differences = [
        f"{key} {_config_value(config, key)!r}, not {expected!r}"
        for key, expected in family.draft.signature
        if not _same(_config_value(config, key), expected)
    ]
    if differences:
        raise models.ModelError(
            f"draft configuration is incompatible with {family.name}: "
            + "; ".join(differences)
        )
    downloaded = repo.download({"config.json", *weights})
    return {"draft/" + name: path for name, path in downloaded.items()}


def _config_value(config, key):
    """config's value at a key dotted into its objects, lists as tuples, or
    None."""
    value = config
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return tuple(value) if isinstance(value, list) else value


def _same(value, expected):
    """JSON equality that tells booleans from numbers."""
    return value == expected and isinstance(value, bool) == isinstance(expected, bool)
