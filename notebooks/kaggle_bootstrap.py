"""Logic behind the Kaggle bootstrap notebooks, kept out of the notebooks.

A notebook cell cannot be unit-tested and diffs unreadably, so everything in
here is the part of the Kaggle workflow that has to be *right*: which pip
install will not destroy the session's torch, where five separately-mounted
Kaggle Datasets get stitched into the one tree the manifest expects, which
rows this account's shard covers, whether a resume is a continuation or a
different bank wearing the same directory name, and what an error at 2am
actually means. The notebooks are drivers: parameters, prints, and prose.

Nothing here imports torch, loads a model, or touches the network. It is
importable (and tested) on a CPU box with no weights and no Kaggle.

Three things in this module encode a project rule rather than a convenience,
and changing them silently corrupts a bank:

* `shard_bounds` slices CONTIGUOUSLY and `shard_frame` never calls
  `reset_index`. See their docstrings -- both halves are load-bearing, and
  for different reasons.
* `require_gate` is a positional argument of `shard_frame` and
  `run_shard_argv`, so a teammate cannot build an extraction command without
  having verified the data first. That is deliberate ergonomics: the check
  costs minutes and skipping it costs GPU-hours.
* `check_resume` refuses a resume whose shard, splits, backbone or seed have
  moved, and it does so by reading `config.json` alone -- before the backbone
  is loaded, so the failure costs seconds rather than a 1.2 GB download.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass

# --- constants -------------------------------------------------------------

#: Kaggle's documented per-session wall clock for a GPU notebook. Treat it as
#: planning arithmetic, not a promise: the runtime's own countdown is the
#: authority, and an idle-timeout or a maintenance restart can end a session
#: earlier. `session_plan` deliberately budgets against USABLE_SESSION_HOURS.
SESSION_LIMIT_HOURS = 12.0

#: What to actually plan against. The gap absorbs the clone, the pip step, the
#: verify pass and the final checkpoint -- none of which are extraction but all
#: of which spend the same clock.
USABLE_SESSION_HOURS = 10.0

#: Kaggle's cap on notebook output (`/kaggle/working`). A shard bank has to fit
#: with room for the notebook's own files, or the session ends with nothing
#: saved -- the one failure mode that costs the whole run rather than the last
#: checkpoint interval.
WORKING_QUOTA_BYTES = 20 * 1024**3

#: Where a bank being written must live. `/kaggle/temp` is faster and does not
#: count against the quota, but it is DISCARDED when the session ends, which
#: is exactly the event `--resume` exists to survive.
WORKING_DIR = "/kaggle/working"
#: Scratch that does not persist. Correct for the symlink farm (free to
#: rebuild next session) and wrong for anything you would miss.
TEMP_DIR = "/kaggle/temp"

#: Packages the Kaggle image pins to its own CUDA build. Installing or
#: upgrading any of these with a normal dependency resolution is the classic
#: way to end a session with a torch that cannot see the GPU.
HOST_PINNED = ("torch", "torchvision", "torchaudio", "triton", "nvidia")

#: Distribution name -> the module you actually import to find out whether it
#: is already there. Checking the IMPORT name, not the distribution name, is
#: what stops us installing `opencv-python-headless` over Kaggle's
#: `opencv-python`: both provide `cv2`, the code only ever asks for `cv2`, and
#: installing the second one on top of the first is a well-known way to get a
#: broken cv2.
IMPORT_NAMES = {
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "pyarrow": "pyarrow",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "tqdm": "tqdm",
    "torch": "torch",
    "transformers": "transformers",
}

#: `transformers` floor that has DINOv3 and SigLIP2 (mirrors pyproject.toml).
#: Kaggle images move, and an image older than this cannot load the primary
#: backbone at all -- so this is checked before the GPU is paid for.
TRANSFORMERS_FLOOR = (4, 53)
#: pyproject's `requires-python`.
PYTHON_FLOOR = (3, 11)

#: Bytes per view per image in a bank, excluding `feats` (which scales with the
#: backbone's dim). presence (6, f32) + severity (6, f32) + proxies (3, f32).
_FIXED_BYTES_PER_VIEW = 6 * 4 + 6 * 4 + 3 * 4
#: recon.npy, when attached: (12, f32).
_RECON_BYTES_PER_VIEW = 12 * 4


# --- environment: install without losing the session's torch ----------------

def version_tuple(text: str) -> tuple[int, ...]:
    """The leading numeric components of a version string.

    Exists because the comparison it feeds is `installed >= 4.53`, and doing
    that on strings gets it backwards: `"4.9" > "4.53"` lexically, so a
    transformers too old to have DINOv3 would be waved through. Tolerates the
    suffixes real environments produce -- `2.10.0+cu126`, `4.53.0.dev0`,
    `1.26.4rc1` -- by stopping at the first component that is not an integer.
    """
    out: list[int] = []
    for part in str(text).strip().split("."):
        m = re.match(r"^(\d+)", part)
        if not m:
            break
        out.append(int(m.group(1)))
        if m.group(0) != part:          # e.g. "0+cu126", "0dev0", "4rc1"
            break
    return tuple(out)


def _at_least(got: tuple[int, ...], floor: tuple[int, ...]) -> bool:
    return got[: len(floor)] >= floor if got else False


def requirement_name(requirement: str) -> str:
    """The distribution name in a PEP 508 requirement string, lowercased."""
    return re.split(r"[<>=!~\[;\s]", str(requirement).strip(), maxsplit=1)[0].lower()


def read_dependencies(pyproject_path: str) -> list[str]:
    """`[project].dependencies` from a pyproject.toml, as written."""
    import tomllib

    with open(pyproject_path, "rb") as f:
        return list(tomllib.load(f)["project"]["dependencies"])


def missing_requirements(requirements, is_present, skip=HOST_PINNED) -> list[str]:
    """The subset of `requirements` that is neither host-pinned nor importable.

    `is_present(module_name) -> bool` is injected so this is decidable without
    importing anything (and so it is testable off Kaggle). Requirements whose
    distribution name starts with any entry in `skip` are dropped entirely --
    NOT reported as missing -- because the correct action for a host-pinned
    package is never "pip install it".

    Returns the requirement strings unchanged, version specifiers included, so
    the caller installs the bound the project actually declares.
    """
    skip_lower = tuple(s.lower() for s in skip)
    out = []
    for req in requirements:
        name = requirement_name(req)
        if any(name == s or name.startswith(s + "-") for s in skip_lower):
            continue
        module = IMPORT_NAMES.get(name, name.replace("-", "_"))
        if not is_present(module):
            out.append(req)
    return out


def module_is_present(module_name: str) -> bool:
    """True if `module_name` can be found without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def install_plan(pyproject_path: str, repo_dir: str, *, is_present=None,
                 transformers_version: str | None = None) -> list[list[str]]:
    """The pip invocations to run on Kaggle, in order, as argv lists.

    Two rules produce this plan, and both exist to keep the session's CUDA
    torch untouched:

    1. The project itself goes in with `--no-deps`. A plain `pip install -e .`
       hands pip the `torch>=2.0` line from pyproject and invites it to
       resolve torch -- which on Kaggle means downloading a wheel built for a
       different CUDA than the drivers, and losing the GPU for the rest of the
       session. `--no-deps` makes the editable install a pure path
       registration, which is all it is needed for.
    2. Everything else is installed only if it is genuinely absent, and
       host-pinned packages are never installed at all.

    A `transformers` older than the DINOv3 floor is upgraded, also with
    `--no-deps`: transformers depends on nothing that would drag torch, but
    passing `--no-deps` costs nothing and removes the question.

    argv lists, not shell strings: a repo path with a space in it must not
    become two arguments.
    """
    is_present = module_is_present if is_present is None else is_present
    plan = [[sys.executable, "-m", "pip", "install", "--no-deps", "-e", repo_dir]]
    missing = missing_requirements(read_dependencies(pyproject_path), is_present)
    if missing:
        plan.append([sys.executable, "-m", "pip", "install", *missing])
    if transformers_version is not None and not _at_least(
            version_tuple(transformers_version), TRANSFORMERS_FLOOR):
        floor = ".".join(str(p) for p in TRANSFORMERS_FLOOR)
        plan.append([sys.executable, "-m", "pip", "install", "--no-deps", "-U",
                     f"transformers>={floor}"])
    return plan


def environment_problems(python_version: str, torch_version: str | None,
                         transformers_version: str | None) -> list[str]:
    """Blocking environment faults, worst first; empty means go.

    Checked BEFORE the backbone is downloaded, because every entry here makes
    the download pointless: no torch means no extraction at all, and a
    transformers below the floor cannot construct DINOv3 whatever the weights
    say.
    """
    out = []
    if not _at_least(version_tuple(python_version), PYTHON_FLOOR):
        out.append(
            f"Python {python_version} is below the project's requires-python "
            f">={'.'.join(map(str, PYTHON_FLOOR))}. Kaggle images move; pick "
            "an image with a newer Python in Notebook options, or pin the "
            "environment to a previous version that had one.")
    if not torch_version:
        out.append(
            "torch is not importable. On Kaggle this means the notebook is on "
            "a CPU-only image or a pip install replaced torch -- do NOT try to "
            "pip install torch to fix it, factory-reset the session "
            "(Run > Factory reset) and re-run without any torch install.")
    if not transformers_version:
        out.append(
            "transformers is not importable; install_plan's transformers step "
            "did not run or failed.")
    elif not _at_least(version_tuple(transformers_version), TRANSFORMERS_FLOOR):
        out.append(
            f"transformers {transformers_version} is below "
            f"{'.'.join(map(str, TRANSFORMERS_FLOOR))}, which is the floor that "
            "has DINOv3. Upgrade it with --no-deps (install_plan does this) "
            "and restart the kernel.")
    return out


# --- HuggingFace auth -------------------------------------------------------

#: The Kaggle Secret this project expects the token under. One name, so five
#: teammates' notebooks are the same file.
HF_SECRET_NAME = "HF_TOKEN"


def hf_token(secrets_client=None, environ=None) -> str | None:
    """The HuggingFace token, from Kaggle Secrets first and `$HF_TOKEN` second.

    Never returns it via a print and never writes it to a file the notebook
    saves: this repo is public, and a token pasted into a cell is committed
    with the notebook's output. `secrets_client` is injected for testing and
    is `kaggle_secrets.UserSecretsClient()` in a real session.

    A missing secret is not an error here -- `hf_auth_advice` turns the absence
    into the instructions -- because raising inside an auth helper produces a
    traceback with the word "token" in it, which is exactly the cell a teammate
    then screenshots into a group chat.
    """
    environ = os.environ if environ is None else environ
    if secrets_client is not None:
        try:
            value = secrets_client.get_secret(HF_SECRET_NAME)
        except Exception:
            value = None
        if value and str(value).strip():
            return str(value).strip()
    value = environ.get("HF_TOKEN") or environ.get("HUGGING_FACE_HUB_TOKEN")
    return str(value).strip() if value and str(value).strip() else None


def requires_hf_token(backbone: str) -> bool:
    """Whether this backbone's weights need a HuggingFace token to download.

    Read off the registry rather than assumed, because the answer changed with
    the machine allocation: DINOv3 is gated behind Meta's licence and runs on
    the A4500, while the fleet runs SigLIP2 (Apache-2.0) and CLIP (MIT), which
    need nothing. A blanket requirement would stop four teammates to accept a
    licence their run never touches.
    """
    from aigcdet.features.backbones import BACKBONES

    return bool(BACKBONES[backbone].gated)


def hf_auth_advice(token: str | None, model_id: str,
                   gated: bool = True) -> list[str]:
    """What to do about HuggingFace auth, given whether a token was found.

    Says the gated-repo part even when a token IS present, because the two
    failures look identical from the notebook (both are a 401/403 on
    `from_pretrained`) and only one of them is fixed by adding a token: the
    licence is accepted per ACCOUNT, so the project owner's acceptance does
    nothing for a teammate's token.

    `gated=False` says so plainly instead. Telling someone a public repo is
    gated does not fail safe -- they go and do the useless work, and the real
    cause of whatever they were debugging stays unexamined.
    """
    if not gated:
        return [
            f"{model_id} is a public repo -- no token is required, and none "
            "was needed to reach this cell.",
            "If a download fails here it is the network or the mirror, not "
            "authentication. Re-run the cell.",
        ]
    if not token:
        return [
            f"No HuggingFace token found. {model_id} is a GATED repo and will "
            "fail to download without one.",
            "1. Log in to huggingface.co with YOUR OWN account and open "
            f"https://huggingface.co/{model_id} -- accept the licence there. "
            "Acceptance is per account: the project owner accepting it does "
            "nothing for yours.",
            "2. Create a READ token at "
            "https://huggingface.co/settings/tokens.",
            "3. In this notebook: Add-ons > Secrets > Add a secret, with "
            f"label {HF_SECRET_NAME} and the token as the value, then attach "
            "it to this notebook and re-run this cell.",
            "Do NOT paste the token into a cell. This repo is public and a "
            "notebook is committed with its cell source.",
        ]
    return [
        "HuggingFace token loaded from Kaggle Secrets (not printed).",
        f"If {model_id} still 401s or 403s, the token is fine and the LICENCE "
        "is not: open "
        f"https://huggingface.co/{model_id} while logged in as the account "
        "that issued this token and accept it. Acceptance is per account.",
    ]


# --- attaching the data -----------------------------------------------------

def top_level_names(manifest_df) -> set[str]:
    """The first path component of every `rel_path` -- the directory names the
    dataset root is expected to contain (`wildfake`, `coco`, ...).

    This is what turns "find the root" from a guess into a lookup: the
    manifest already records what the root looks like from the inside.
    """
    if "rel_path" not in manifest_df.columns:
        raise ValueError(
            "this manifest has no rel_path column, so it cannot say what its "
            "dataset root contains. Re-freeze it with a current write_manifest.")
    return {str(r).replace("\\", "/").split("/", 1)[0]
            for r in manifest_df["rel_path"]}


def content_root(path: str, expected_names=None, listdir=None, isdir=None) -> str:
    """Descend past the wrapper directories Kaggle adds around a Dataset.

    An uploaded Dataset arrives as `/kaggle/input/<slug>/<whatever-you-
    zipped>/...`, sometimes with two such levels. `rel_path` is relative to
    the root the manifest was FROZEN at, so aiming one level too high and one
    level too low produce the SAME symptom -- every row missing -- with
    opposite fixes. That is why this is computed rather than typed.

    `expected_names` (from `top_level_names`) is what makes it exact: descend
    until a directory contains one of the names the manifest says the root
    holds, and stop there. Without it this falls back to "descend while there
    is exactly one entry and it is a directory", which is right for the common
    wrapper and OVERSHOOTS a genuine root that happens to hold a single source
    directory -- so pass the manifest's names whenever they are available.
    """
    listdir = os.listdir if listdir is None else listdir
    isdir = os.path.isdir if isdir is None else isdir
    expected = set(expected_names or ())
    current = str(path)
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        try:
            entries = sorted(listdir(current))
        except OSError:
            return current
        if expected and any(e in expected for e in entries):
            return current
        if len(entries) != 1:
            return current
        child = os.path.join(current, entries[0])
        if not isdir(child):
            return current
        current = child
    return current


@dataclass(frozen=True)
class UnifiedRoot:
    """One tree stitched together from several Kaggle mounts."""

    root: str
    #: The per-mount content roots that were linked in, in the order given.
    sources: tuple[str, ...] = ()
    #: True when `root` is a symlink farm rather than a real directory tree.
    #: `verify_images`' extra-file scan uses `os.walk`, which does not follow
    #: symlinked directories, so an "extra files: 0" from a farm means "not
    #: checked", not "clean". The caller must pass `check_extra=False` and say
    #: so, rather than quoting a zero it did not earn.
    linked: bool = False


def unify_mounts(mounts, target: str, expected_names=None) -> UnifiedRoot:
    """Present several Kaggle Dataset mounts as the single root the manifest wants.

    The normalised dataset is published as ~5 Kaggle Datasets because of the
    per-Dataset size cap, and Kaggle mounts each at its own
    `/kaggle/input/<slug>/`. A manifest describes ONE tree
    (`aigcdet.data.manifest.dataset_root` refuses a frame implying two roots),
    so the five mounts have to look like one directory before anything can be
    rebased onto it.

    Symlinks, not copies: the data is tens of GB, `/kaggle/input` is read-only,
    and copying it would blow the working quota for no benefit. The farm goes
    in `/kaggle/temp` by convention -- it does not persist, and it costs
    nothing to rebuild next session.

    A name present in two mounts is an error, not a merge: the published
    Datasets are disjoint slices of one tree, so a collision means the wrong
    Datasets are attached (or one twice), and silently letting the first win
    would leave rows missing for a reason no error message would name.
    """
    os.makedirs(target, exist_ok=True)
    sources = []
    for mount in mounts:
        src = content_root(str(mount), expected_names)
        sources.append(src)
        for entry in sorted(os.listdir(src)):
            link = os.path.join(target, entry)
            dest = os.path.join(src, entry)
            if os.path.lexists(link):
                if os.path.realpath(link) == os.path.realpath(dest):
                    continue
                raise ValueError(
                    f"{entry!r} is present in more than one attached Dataset "
                    f"({os.path.realpath(link)} and {dest}). The published "
                    "Datasets are disjoint slices of one tree, so this means "
                    "the wrong set is attached, or one of them twice. Detach "
                    "the duplicate and re-run; do not let one silently win.")
            os.symlink(dest, link)
    return UnifiedRoot(root=target, sources=tuple(sources), linked=True)


# --- the verification gate --------------------------------------------------

@dataclass(frozen=True)
class VerifyGate:
    """Proof that the data under `root` is the data the manifest was frozen on.

    Carried as a value rather than left as a printed "OK" because it is a
    REQUIRED argument of `shard_frame` and `run_shard_argv`: a teammate who
    skips the verify cell has no `GATE` to pass and gets a NameError in
    seconds, instead of a bank built from a half-downloaded Dataset discovered
    ten GPU-hours later.
    """

    manifest_sha256: str
    root: str
    n_rows: int
    digest_kind: str | None
    n_digested: int
    sampled: bool
    extra_checked: bool
    warnings: tuple[str, ...] = ()


def open_verified_manifest(manifest_path: str, root: str, *, digest="auto",
                           sample: int | None = None, check_extra: bool = True,
                           workers: int = 8, verify=None, reader=None):
    """Read the frozen manifest, rebase it onto `root`, and prove the files match.

    Returns `(manifest_df, gate)` and raises `ValueError` -- carrying
    `VerifyReport.describe()`, which names WHICH of missing / unreadable /
    divergent went wrong and what each one's fix is -- if anything fatal is
    found. A pixel-identical re-encode and an unlisted extra file are not
    fatal and arrive as gate warnings.

    `sample` digests only that many evenly spaced rows. It is the right first
    look at a 100k-image Dataset and it is NOT proof; the gate records that it
    was a sample and `describe_gate` says so on every subsequent print, because
    the number a teammate remembers is "it said OK".
    """
    from aigcdet.data.manifest import read_manifest as _read
    from aigcdet.data.verify import verify_images as _verify
    from aigcdet.features.bank import manifest_fingerprint

    reader = _read if reader is None else reader
    verify = _verify if verify is None else verify

    df = reader(manifest_path, root=root)
    report = verify(df, root=root, digest=digest, sample=sample,
                    check_extra=check_extra, workers=workers)
    report.raise_for_status()

    warnings = []
    if report.sampled:
        warnings.append(
            f"only {report.n_digested} of {report.n_rows} rows were digested "
            "(sample). A clean sample is evidence, not proof -- re-run with "
            "sample=None before an extraction you intend to keep.")
    if not check_extra:
        warnings.append(
            "the extra-file scan was skipped, so 'extra files: 0' was not "
            "checked (os.walk does not follow the symlink farm's directories).")
    if report.n_reencoded:
        warnings.append(
            f"{report.n_reencoded} file(s) are re-encoded but pixel-identical; "
            "extraction is safe.")
    if report.digest_kind is None:
        warnings.append(
            "this manifest carries no content digests, so only presence was "
            "checked. Nothing verified the bytes.")
    gate = VerifyGate(
        manifest_sha256=manifest_fingerprint(df), root=root,
        n_rows=int(report.n_rows), digest_kind=report.digest_kind,
        n_digested=int(report.n_digested), sampled=bool(report.sampled),
        extra_checked=bool(check_extra), warnings=tuple(warnings))
    return df, gate


def require_gate_shape(gate) -> VerifyGate:
    """Raise unless `gate` is a `VerifyGate` at all.

    Split out from `require_gate` for the one caller that has no manifest
    frame in hand (`run_shard_argv`, which builds a command line): it can
    still refuse to produce a command for a teammate who never ran the
    verification cell, and `run_shard.py` re-checks the fingerprint in the
    child process where the frame does exist.
    """
    if not isinstance(gate, VerifyGate):
        raise TypeError(
            "the first argument must be the VerifyGate returned by "
            "open_verified_manifest. Run the verification cell -- it is the "
            "only thing standing between a broken Dataset mount and ten wasted "
            f"GPU-hours (got {type(gate).__name__}).")
    return gate


def require_gate(gate, manifest_df) -> VerifyGate:
    """Raise unless `gate` is a real gate for exactly `manifest_df`.

    Re-fingerprinting rather than trusting the object catches the case the
    gate exists for: the manifest was reloaded, re-filtered or re-split in a
    cell run after the verify cell, so the frame about to be extracted is no
    longer the frame that was checked.
    """
    from aigcdet.features.bank import manifest_fingerprint

    require_gate_shape(gate)
    actual = manifest_fingerprint(manifest_df)
    if actual != gate.manifest_sha256:
        raise ValueError(
            "this manifest is not the one that was verified: the gate holds "
            f"{gate.manifest_sha256[:16]}..., this frame fingerprints to "
            f"{actual[:16]}.... The manifest was reloaded, re-filtered or "
            "re-split after the verification cell ran. Re-run the "
            "verification cell against the frame you actually intend to "
            "extract.")
    return gate


def describe_gate(gate: VerifyGate) -> str:
    lines = [f"verified: {gate.n_rows} rows under {gate.root}",
             f"  digest {gate.digest_kind or 'none'} over {gate.n_digested} rows",
             f"  manifest fingerprint {gate.manifest_sha256[:16]}..."]
    lines += [f"  WARNING: {w}" for w in gate.warnings]
    return "\n".join(lines)


# --- sharding ---------------------------------------------------------------

def shard_bounds(n: int, n_shards: int) -> list[tuple[int, int]]:
    """`n_shards` contiguous half-open row ranges covering `range(n)` exactly.

    CONTIGUOUS, not strided (`iloc[k::5]`), and that is not a stylistic
    choice. `bank.merge_banks` concatenates shards in the order it is given
    and fingerprints the result over the concatenated `rel_path` list, and
    every downstream reader indexes the bank POSITIONALLY against the
    manifest. Only contiguous ascending slices, merged in shard order,
    reconstruct the frozen manifest's row order -- a strided split would
    produce a bank whose rows are 0,5,10,...,1,6,11,..., which merges without
    complaint (no row_id overlap) and then fails
    `FeatureBank.verify_against_manifest`, or worse, passes a check nobody ran
    and trains a head against permuted labels.

    The remainder goes to the FIRST `n % n_shards` shards, so the ranges are
    balanced to within one row and the partition is exhaustive: every row is
    extracted exactly once, which is the property `merge_banks`' overlap check
    would otherwise catch only after five people had each paid for a session.
    """
    n, n_shards = int(n), int(n_shards)
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    base, rem = divmod(n, n_shards)
    bounds, start = [], 0
    for k in range(n_shards):
        stop = start + base + (1 if k < rem else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def shard_frame(gate, manifest_df, shard: int, n_shards: int, *, splits=None):
    """This account's contiguous slice of the (split-filtered) manifest.

    `gate` is first and required: see `require_gate`.

    `splits` filters before slicing, so the five shards partition the rows
    that will actually be extracted rather than the whole manifest -- five
    equal slices of the whole manifest would give shard 4 nothing but
    `benchmark` rows and shard 0 nothing but `train`.

    The returned frame keeps the manifest's ORIGINAL index labels. There is no
    `reset_index` here and there must never be one: `extract_bank` derives
    every view's RNG from `(seed, row_id, view_idx)` where `row_id` is that
    index label, so a reset would restart every shard's keys at 0, five shards
    would collide in RNG-key space, and the same physical image would carry
    different pixels depending on who extracted it. `extract_bank` raises on a
    duplicated index, but it cannot see that two SEPARATE sessions produced
    overlapping keys -- that damage only surfaces as a bank that disagrees
    with itself.
    """
    require_gate(gate, manifest_df)
    if not 0 <= int(shard) < int(n_shards):
        raise ValueError(
            f"shard must be in range(0, {n_shards}), got {shard}. Shards are "
            "0-indexed; five teammates run 0, 1, 2, 3, 4.")
    df = manifest_df if splits is None else select_splits(manifest_df, splits)
    start, stop = shard_bounds(len(df), int(n_shards))[int(shard)]
    return df.iloc[start:stop]


def select_splits(df, splits):
    """`scripts/extract_features.py`'s own split filter, reused not copied.

    Imported from the script rather than reimplemented so the notebook and the
    CLI cannot drift on what an unknown split name does (it raises -- the
    alternative is an empty bank discovered after the extraction is paid for),
    and so neither one resets the index.

    `splits` may be a comma-separated string or a sequence of names.
    """
    if not isinstance(splits, str):
        splits = ",".join(str(s) for s in splits)
    return load_script_module("extract_features").select_splits(df, splits)


def carried_gate(manifest_sha256: str, root: str) -> VerifyGate:
    """A gate reconstructed in a CHILD process from the fingerprint alone.

    `run_shard.py` runs in its own interpreter (it must -- see
    `run_shard_argv`), so it cannot be handed the `VerifyGate` object the
    notebook holds. It is handed the fingerprint instead and rebuilds a gate
    from it, which `require_gate` then checks against the manifest the child
    reads for itself. That is the same guarantee: the child proves it is
    looking at the rows the parent verified, and refuses if the manifest moved
    between the two.

    The fields the parent measured (digest kind, sample size) are not carried,
    because the child makes no claim about them -- only about identity.
    """
    return VerifyGate(
        manifest_sha256=str(manifest_sha256), root=str(root), n_rows=-1,
        digest_kind=None, n_digested=0, sampled=False, extra_checked=False,
        warnings=("gate carried from the parent process by fingerprint only",))


def resolve_shard(manifest_path: str, root: str, splits: str, shard: int,
                  n_shards: int, expect_manifest_sha256: str):
    """`(full_manifest_df, this_shard_df)` for a child process.

    The shard is re-derived HERE from the manifest, rather than being passed
    as a row range on the command line. A row range typed into a notebook is a
    claim; recomputing it from `(shard, n_shards)` against the manifest the
    fingerprint just matched is a derivation, and the two only ever disagree
    when something is already wrong.
    """
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(manifest_path, root=root)
    gate = carried_gate(expect_manifest_sha256, root)
    return df, shard_frame(gate, df, shard, n_shards, splits=splits)


#: Cache for `load_script_module`. `scripts/extract_features.py` pulls in
#: `aigcdet.features.extract` (and so torch) on import, and `select_splits`
#: calls it on every shard; without this, `shard_plan` would re-execute that
#: import once per shard.
_SCRIPT_MODULES: dict[str, object] = {}


def load_script_module(name: str, repo_dir: str | None = None):
    """Import a module from `scripts/`, which is not an installed package.

    By file path rather than by putting `scripts/` on `sys.path`: the
    directory holds several single-word module names (`predict`, `audit`) that
    would shadow real packages for the rest of the session.
    """
    repo_dir = repo_root() if repo_dir is None else repo_dir
    path = os.path.join(repo_dir, "scripts", f"{name}.py")
    if path in _SCRIPT_MODULES:
        return _SCRIPT_MODULES[path]
    spec = importlib.util.spec_from_file_location(f"_aigcdet_scripts_{name}", path)
    if spec is None or spec.loader is None:                   # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SCRIPT_MODULES[path] = module
    return module


def repo_root() -> str:
    """The checkout this file lives in (`notebooks/` is a direct child)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def shard_plan(gate, manifest_df, n_shards: int, *, splits=None) -> list[dict]:
    """One row per shard: who extracts what, and how big it is.

    Printed before anybody starts, so five people can agree on the assignment
    from one table instead of five recollections of a chat message.
    """
    require_gate(gate, manifest_df)
    df = manifest_df if splits is None else select_splits(manifest_df, splits)
    rows = []
    for k, (start, stop) in enumerate(shard_bounds(len(df), int(n_shards))):
        part = df.iloc[start:stop]
        rows.append({
            "shard": k,
            "rows": int(stop - start),
            "iloc": f"[{start}:{stop})",
            "row_id_first": int(part.index[0]) if len(part) else None,
            "row_id_last": int(part.index[-1]) if len(part) else None,
            "n_fake": int((part["label"] == 1).sum()) if len(part) else 0,
            "splits": dict(sorted(part["split"].value_counts().items()))
                      if len(part) else {},
        })
    return rows


# --- sizing and scheduling --------------------------------------------------

def bank_bytes(n_images: int, dim: int, n_views: int = 11,
               with_recon: bool = False) -> int:
    """On-disk size of a bank of `n_images`.

    Worth computing rather than discovering: `BankWriter` preallocates every
    `.npy` at FULL size in `__init__`, before the first image is processed, so
    a shard that will not fit fails at the end of the session with the quota
    exceeded and nothing saved -- not gradually, and not with a warning at
    50%.
    """
    per_view = dim * 2 + _FIXED_BYTES_PER_VIEW      # feats are float16
    if with_recon:
        per_view += _RECON_BYTES_PER_VIEW
    return int(n_images) * int(n_views) * per_view


def fits_in_working(n_images: int, dim: int, n_views: int = 11,
                    quota: int = WORKING_QUOTA_BYTES,
                    reserve: int = 512 * 1024**2) -> tuple[bool, str]:
    """Whether a shard bank fits `/kaggle/working`, with a reserve for everything
    else the session writes (the clone, pip's cache, the notebook itself)."""
    need = bank_bytes(n_images, dim, n_views)
    ok = need + reserve <= quota
    return ok, (
        f"bank {need / 1024**3:.2f} GiB + {reserve / 1024**3:.2f} GiB reserve "
        f"vs {quota / 1024**3:.0f} GiB working quota: "
        + ("fits" if ok else
           f"DOES NOT FIT -- raise n_shards to at least "
           f"{_min_shards(n_images, dim, n_views, quota, reserve)}"))


def _min_shards(n_images, dim, n_views, quota, reserve) -> int:
    per_image = bank_bytes(1, dim, n_views)
    room = max(1, quota - reserve)
    return max(1, -(-int(n_images) * per_image // room))


@dataclass(frozen=True)
class SessionPlan:
    n_images: int
    seconds_per_image: float
    hours: float
    fits_session: bool
    sessions_needed: int
    checkpoint_every: int
    #: Work at risk when the session is killed: the images processed since the
    #: last metadata checkpoint.
    minutes_at_risk: float
    notes: tuple[str, ...] = ()


def measure_rate(n_images: int, seconds: float) -> float:
    """Seconds per image, from a timed smoke run.

    The point of taking this from a measurement instead of a constant: the
    project's own 8-13 h figure was measured on the A4500, Kaggle's T4 is a
    different machine, and the number that decides how many shards five people
    need is this one. A smoke run of 32 images costs a minute and replaces the
    guess.
    """
    if n_images <= 0:
        raise ValueError("cannot measure a rate from zero images")
    if seconds <= 0:
        raise ValueError("cannot measure a rate from a non-positive duration")
    return float(seconds) / float(n_images)


def marginal_rate(n1: int, seconds1: float, n2: int, seconds2: float) -> float:
    """Seconds per image with the fixed startup cost differenced out.

    A single timed smoke run measures the wrong thing. Most of a 40-image run
    is the ~1.2 GB backbone download, the CUDA context and the process start;
    dividing by 40 gives a per-image figure several times too large, and
    extrapolating it tells five teammates they need three times the shards
    they do. Two runs of different sizes and one subtraction removes every
    cost that does not scale with the image count.
    """
    n1, n2 = int(n1), int(n2)
    if n2 <= n1:
        raise ValueError(
            f"the second smoke run must be strictly larger ({n2} <= {n1}); the "
            "subtraction is what removes the startup cost")
    if seconds2 <= seconds1:
        raise ValueError(
            f"the larger run finished no slower ({seconds2} <= {seconds1}), so "
            "the timings are dominated by noise. Re-run with a bigger second "
            "sample.")
    return (float(seconds2) - float(seconds1)) / (n2 - n1)


def run_streaming(argv, on_line=print, env=None) -> int:
    """Run `argv`, echoing its output line by line, and return the exit code.

    Streamed rather than captured because the thing being run prints a tqdm
    bar for the next several hours, and a `subprocess.run(capture_output=True)`
    would show it only after the session that was meant to display it has
    ended.
    """
    import subprocess

    proc = subprocess.Popen([str(a) for a in argv], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env)
    assert proc.stdout is not None
    with proc.stdout:
        for line in proc.stdout:
            on_line(line.rstrip("\n"))
    return proc.wait()


def session_plan(n_images: int, seconds_per_image: float, *,
                 usable_hours: float = USABLE_SESSION_HOURS,
                 checkpoint_every: int = 500) -> SessionPlan:
    """How long this shard takes, and what a session timeout costs.

    `minutes_at_risk` is the honest answer to "what happens when Kaggle kills
    it": the `.npy` memmaps are preallocated and survive on their own, but a
    bank cannot be OPENED without `meta.parquet` / `views.parquet`, which
    `BankWriter` rewrites every `checkpoint_every` images. So a kill costs the
    images since the last checkpoint and nothing more -- provided the run was
    writing to `/kaggle/working`, which persists as notebook output. A run
    writing to `/kaggle/temp` loses everything, every time.
    """
    hours = float(n_images) * float(seconds_per_image) / 3600.0
    fits = hours <= usable_hours
    sessions = max(1, -(-int(round(hours * 60)) // max(1, int(usable_hours * 60))))
    notes = []
    if not fits:
        notes.append(
            f"{hours:.1f} h exceeds the {usable_hours:.0f} h budget: this shard "
            f"needs about {sessions} sessions. That is supported -- re-run the "
            "extraction cell with RESUME=True in a fresh session and it "
            "continues where it stopped -- but raising N_SHARDS is cheaper "
            "than restarting, because a restart re-reads the whole bank's "
            "metadata before it resumes.")
    at_risk = float(checkpoint_every) * float(seconds_per_image) / 60.0
    if at_risk > 30:
        notes.append(
            f"a kill would cost up to {at_risk:.0f} min of work; lower "
            "CHECKPOINT_EVERY (it only rewrites two small parquet files).")
    return SessionPlan(
        n_images=int(n_images), seconds_per_image=float(seconds_per_image),
        hours=hours, fits_session=fits, sessions_needed=sessions,
        checkpoint_every=int(checkpoint_every), minutes_at_risk=at_risk,
        notes=tuple(notes))


# --- resume -----------------------------------------------------------------

@dataclass(frozen=True)
class ResumeState:
    """What is already on disk at a bank directory."""

    exists: bool
    n_images: int = 0
    n_done: int = 0
    backbone: str | None = None
    seed: int | None = None
    n_views: int | None = None
    manifest_sha256: str | None = None

    @property
    def n_remaining(self) -> int:
        return max(0, self.n_images - self.n_done)

    @property
    def fraction_done(self) -> float:
        return 0.0 if not self.n_images else self.n_done / self.n_images


def read_resume_state(bank_dir: str) -> ResumeState:
    """Read a bank's progress from its metadata alone -- no arrays, no torch.

    Deliberately tolerant of a half-written bank: `config.json` is written
    before the first image, so a directory with a config and no `meta.parquet`
    is a session that died in its first checkpoint interval, which is
    `n_done=0` and not an error.
    """
    cfg_path = os.path.join(bank_dir, "config.json")
    if not os.path.exists(cfg_path):
        return ResumeState(exists=False)
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_done = 0
    meta_path = os.path.join(bank_dir, "meta.parquet")
    if os.path.exists(meta_path):
        # The row count from the parquet FOOTER: O(1), no column read, and it
        # works on the file a zero-image `BankWriter.close()` writes -- which
        # is an empty frame with no columns at all, so asking for `image_idx`
        # by name raises there. Progress reporting must never be the thing
        # that fails a resume.
        import pyarrow.parquet as pq

        n_done = int(pq.ParquetFile(meta_path).metadata.num_rows)
    return ResumeState(
        exists=True, n_images=int(cfg.get("n_images", 0)), n_done=n_done,
        backbone=cfg.get("backbone"), seed=cfg.get("seed"),
        n_views=cfg.get("n_views"), manifest_sha256=cfg.get("manifest_sha256"))


def check_resume(bank_dir: str, shard_df, *, backbone: str, seed: int,
                 n_views: int = 11) -> ResumeState:
    """Refuse a "resume" that is really a different bank in the same directory.

    `BankWriter` performs this check too, and correctly -- but it does so
    AFTER `extract_bank` has loaded the backbone, which on a cold Kaggle
    session is a 1.2 GB gated download. Doing it here from `config.json` turns
    "changed SHARD_INDEX and re-ran with RESUME=True" from a five-minute
    failure into an instant one, and names the parameter that moved instead of
    printing a config diff.

    A directory that does not exist is a fresh start, not an error: that is
    what the first session of a resumable run finds.
    """
    from aigcdet.features.bank import manifest_fingerprint

    state = read_resume_state(bank_dir)
    if not state.exists:
        return state
    want = manifest_fingerprint(shard_df)
    problems = []
    if state.manifest_sha256 != want:
        problems.append(
            f"rows: on disk {str(state.manifest_sha256)[:16]}..., requested "
            f"{want[:16]}... -- SHARD_INDEX, N_SHARDS, SPLITS or the manifest "
            "itself changed")
    if state.n_images != len(shard_df):
        problems.append(f"row count: on disk {state.n_images}, "
                        f"requested {len(shard_df)}")
    if state.backbone != backbone:
        problems.append(f"backbone: on disk {state.backbone!r}, "
                        f"requested {backbone!r}")
    if int(state.seed or -1) != int(seed):
        problems.append(f"seed: on disk {state.seed}, requested {seed}")
    if int(state.n_views or -1) != int(n_views):
        problems.append(f"n_views: on disk {state.n_views}, requested {n_views}")
    if problems:
        raise ValueError(
            f"the bank at {bank_dir} is not a continuation of this run:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nA resume must continue the SAME extraction. Either restore "
              "the parameters this directory was started with, or extract to a "
              "NEW directory. Do NOT delete and restart unless you accept "
              f"losing the {state.n_done} images already done.")
    return state


# --- building the extraction command ---------------------------------------

def run_shard_argv(gate, *, manifest_path: str, root: str, backbone: str,
                   out_dir: str, splits: str, shard: int, n_shards: int,
                   resume: bool = True, workers: int = 4, batch_size: int = 16,
                   checkpoint_every: int = 500, limit: int | None = None,
                   device: str = "cuda", runner: str | None = None) -> list[str]:
    """argv for `notebooks/run_shard.py`, the sharded Stage A entry point.

    A separate process, not an in-notebook `extract_bank` call, and that is
    forced by `workers > 0`: the CPU stage's process pool uses the "spawn"
    start method (this process holds a CUDA context and forking one is a
    documented deadlock), spawn re-imports the parent's `__main__`, and a
    notebook kernel's `__main__` is not re-importable. Inline (`workers=0`)
    would work in-notebook and would leave the ~200 ms/image CPU stage
    serialised behind the GPU for the whole run.

    `gate` is required and positional, so this command cannot be built without
    the verification having happened -- and `run_shard.py` re-derives the
    shard from the same frame rather than trusting a row range typed here.

    argv, never a shell string: `--split train,val_internal` is one argument
    and a Kaggle mount path can contain spaces.
    """
    require_gate_shape(gate)
    runner = runner or os.path.join(repo_root(), "notebooks", "run_shard.py")
    argv = [sys.executable, runner,
            "--manifest", str(manifest_path),
            "--root", str(root),
            "--backbone", str(backbone),
            "--out", str(out_dir),
            "--split", str(splits),
            "--shard", str(int(shard)),
            "--n-shards", str(int(n_shards)),
            "--device", str(device),
            "--batch-size", str(int(batch_size)),
            "--checkpoint-every", str(int(checkpoint_every)),
            "--workers", str(int(workers)),
            "--expect-manifest-sha256", str(gate.manifest_sha256)]
    if limit is not None:
        argv += ["--limit", str(int(limit))]
    if resume:
        argv.append("--resume")
    return argv


# --- merging ----------------------------------------------------------------

def merge_argv(out_dir: str, shard_dirs, repo_dir: str | None = None) -> list[str]:
    """argv for `scripts/merge_banks.py`, with the shards in ASCENDING order.

    Order is the whole point. `merge_banks` concatenates in the order given
    and re-fingerprints the result over that concatenation, so shards merged
    out of order produce a bank whose rows are a permutation of the manifest's
    -- which every positional reader downstream will misread.
    """
    repo_dir = repo_root() if repo_dir is None else repo_dir
    shard_dirs = list(shard_dirs)
    if not shard_dirs:
        raise ValueError("merge needs at least one shard directory")
    return [sys.executable, os.path.join(repo_dir, "scripts", "merge_banks.py"),
            "--out", str(out_dir), *[str(d) for d in shard_dirs]]


def sorted_shard_dirs(paths) -> list[str]:
    """Shard directories in ascending shard order, from a `*_shard<k>` naming.

    Sorting the paths as strings gives `shard0, shard1, shard10, shard2`,
    which merges a ten-shard bank into a permuted one. Anything without a
    trailing shard number sorts last, by name, and is reported rather than
    guessed at.
    """
    def key(p):
        m = re.search(r"(?:shard|_)(\d+)/?$", str(p).rstrip("/"))
        return (0, int(m.group(1)), "") if m else (1, 0, str(p))

    return sorted((str(p) for p in paths), key=key)


def verify_merged_bank(bank_dir: str, manifest_path: str,
                       root: str | None = None, splits: str = "train,val_internal") -> str:
    """Check a merged bank against the manifest rows it was actually built from.

    Not `train_rung --manifest`: that flag reads the WHOLE manifest, while a
    training bank is extracted from `--split train,val_internal`, so the
    fingerprints cannot match and the flag rejects a perfectly good bank. The
    equivalent check, done right, is against the same split-filtered frame the
    extraction used -- which is this.

    `root` is optional, and normally omitted: the comparison is over
    `rel_path`, the identity that means the same thing on every machine, so
    the merge session does not need the images attached at all. Pass a root
    only if `$AIGCDET_DATA_ROOT` is set to something misleading.
    """
    from aigcdet.data.manifest import read_manifest
    from aigcdet.features.bank import FeatureBank

    df = select_splits(read_manifest(manifest_path, root=root), splits)
    bank = FeatureBank.open(bank_dir)
    bank.check_invariants()
    bank.verify_against_manifest(df)
    n_val = int((bank.meta["split"] == "val_internal").sum())
    if n_val == 0:
        raise ValueError(
            "the merged bank has no val_internal rows, so Stage B cannot "
            "report a val AUC. Every shard must have been extracted with "
            "--split train,val_internal.")
    return (f"merged bank OK: {len(bank.meta)} rows, "
            f"{bank.config['n_views']} views, backbone "
            f"{bank.config['backbone']}, {n_val} val_internal rows")


# --- what to do when a cell fails ------------------------------------------

@dataclass(frozen=True)
class Diagnosis:
    kind: str
    fatal: bool
    action: str


#: (regex, kind, fatal, action). Order matters: the first match wins, so the
#: specific patterns precede the general ones. `fatal` means "re-running this
#: cell unchanged will fail again"; a non-fatal entry is worth one retry
#: before anything is changed.
_DIAGNOSES: tuple[tuple[str, str, bool, str], ...] = (
    (r"gated repo|awaiting a review|401 Client Error|403 Client Error|"
     r"is not authorized|Access to model .* is restricted",
     "hf-gated", True,
     "Your HuggingFace account has not accepted the DINOv3 licence, or the "
     "Kaggle Secret HF_TOKEN is missing/attached to a different notebook. "
     "Acceptance is PER ACCOUNT -- the project owner's does not cover you. "
     "Accept it at the model page, then Add-ons > Secrets."),
    (r"cannot resume the bank at|cannot resume: ",
     "resume-mismatch", True,
     "This directory holds a different bank: a parameter moved between "
     "sessions (usually SHARD_INDEX or SPLITS). Restore the parameters the "
     "directory was started with, or extract to a new --out. Do not delete "
     "it to make the error go away -- that discards completed images."),
    (r"is not the manifest the bank was built from|"
     r"manifest/bank row \d+ misaligned|"
     r"manifest has \d+ rows but bank has \d+ rows",
     "manifest-drift", True,
     "The bank and the manifest describe different rows. Either the manifest "
     "was re-split after the bank was built -- which is forbidden, the "
     "manifest is frozen -- or the bank is being checked against the whole "
     "manifest when it was extracted from a split subset. Never re-run "
     "build_dataset to 'fix' this."),
    (r"shards overlap",
     "shard-overlap", True,
     "Two shards cover the same rows: someone ran the wrong SHARD_INDEX. "
     "Compare each shard's config.json manifest_sha256 against the shard plan "
     "table, and re-extract the duplicate under its correct index."),
    (r"bank has no val_internal rows",
     "missing-split", True,
     "Extracted with --split train only. Stage B evaluates on the bank's own "
     "val_internal rows, so this bank is unusable and must be re-extracted "
     "with --split train,val_internal."),
    (r"--split names \[.*\], which the manifest does not contain",
     "bad-split-name", True,
     "A typo in SPLITS. The manifest's splits are train, val_internal, "
     "heldout_generator, benchmark."),
    (r"verify_images: FAILED|DECODE to different pixels|"
     r"EVERY row is missing",
     "data-mismatch", True,
     "The attached Datasets are not what the manifest was frozen against. The "
     "report names which of missing / unreadable / divergent it is -- read "
     "that, it has a different fix for each. Do not extract from this copy."),
    (r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED",
     "oom", False,
     "Lower BATCH_SIZE (try 8, then 4) and re-run with RESUME=True; the "
     "completed images are kept. If it OOMs at batch size 4, restart the "
     "session -- something else is holding VRAM."),
    (r"No space left on device|Disk quota exceeded|"
     r"OSError: \[Errno 28\]",
     "disk-full", True,
     "/kaggle/working is full. The bank's .npy files are preallocated at full "
     "size, so this means the shard was always too big: raise N_SHARDS. "
     "Delete nothing from the bank directory -- check the sizing cell's "
     "verdict instead."),
    (r"ConnectionError|ReadTimeout|Read timed out|IncompleteRead|"
     r"Connection reset by peer|Temporary failure in name resolution|"
     r"ProtocolError|Max retries exceeded",
     "network", False,
     "A transient network failure (the model download or the git clone). "
     "Re-run the cell. If it is the extraction cell, RESUME=True keeps "
     "everything already done."),
    (r"Your notebook tried to allocate more memory than is available|"
     r"MemoryError|Killed",
     "ram", False,
     "The kernel ran out of host RAM. Lower WORKERS (each subprocess holds an "
     "image's 11 views, ~8.6 MB each) and re-run with RESUME=True."),
    (r"could not start the worker pool|"
     r"An attempt has been made to start a new process before",
     "spawn", True,
     "extract_bank was called with workers > 0 from something that is not a "
     "re-importable __main__ -- almost always a direct call from a notebook "
     "cell. Use notebooks/run_shard.py (which the notebook does), or pass "
     "workers=0."),
    (r"No module named 'aigcdet'",
     "not-installed", True,
     "The install cell did not run, or the kernel was restarted after it. "
     "Re-run the install cell; it is idempotent and does not touch torch."),
    (r"No module named 'torch'|libcud|CUDA driver version is insufficient|"
     r"Torch not compiled with CUDA",
     "torch-broken", True,
     "The session's torch has been replaced or is CPU-only. Do NOT pip "
     "install torch. Factory-reset the session (Run > Factory reset), check "
     "the accelerator is set to GPU, and re-run -- the install cell uses "
     "--no-deps precisely to avoid this."),
)


def diagnose(error) -> Diagnosis:
    """Turn an exception or a traceback string into "fatal?" and "do what?".

    Exists because the expensive mistake at 2am is not misreading an error, it
    is re-running a fatal one -- burning another hour of a 30 h weekly budget
    on a cell that cannot succeed. Every fatal entry below names the parameter
    to change; every non-fatal one is genuinely worth one retry.
    """
    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    for pattern, kind, fatal, action in _DIAGNOSES:
        if re.search(pattern, text, re.IGNORECASE):
            return Diagnosis(kind=kind, fatal=fatal, action=action)
    return Diagnosis(
        kind="unknown", fatal=True,
        action="Not a failure this notebook has a playbook for. Read the LAST "
               "line of the traceback, not the first. Before re-running "
               "anything expensive, check the extraction cell's RESUME flag is "
               "True so nothing already done is repeated.")


def explain(error) -> str:
    d = diagnose(error)
    # Built in two statements rather than one nested f-string: PEP 701 (an
    # f-string expression spanning lines) is 3.12+, and Kaggle runs 3.11.
    verdict = ("FATAL -- re-running unchanged will fail again" if d.fatal
               else "RETRYABLE -- worth one re-run")
    return f"[{d.kind}] {verdict}\n{d.action}"


def as_dict(obj):
    """dataclass -> dict, for printing a plan as a DataFrame."""
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else dict(obj)
