"""Build the curation-plan train/val manifest.

The plan is a LINEAGE split, not a random one: a generator goes to `val` only
if its decoder lineage is absent from `train`, so a score on `val` is a
zero-shot number rather than an interpolation. Every assignment below is
therefore data, not a heuristic -- the tables are the specification and are
meant to be read against the plan document.

Four decisions the plan itself did not settle, resolved 2026-09-01:

  * SID_Set is EXCLUDED entirely. It is a pooled collection with no per-model
    labels, so it cannot be shown to be lineage-disjoint from train; putting
    it in val would let an already-seen lineage masquerade as zero-shot.
  * VQGAN -> val. Rule 4b: a tokenizer-decoder paradigm, and it sits with
    VQ-VAE. This is also how it is already flagged on disk.
  * Both OV7 builds are used. Their reals are disjoint (verified: 0 overlap),
    so the union is 54,623 pairs and carries no leak.
  * Unpaired reals are drawn to bring BOTH splits to ~1:1. Val's share is
    taken first and stratified across source, so `source` does not become a
    split signature -- the failure this whole corpus exists to avoid.

Paired reals (OV7, commercial) follow their own fake and are never drawn from,
which is what keeps a real out of both splits.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from PIL import Image

EXP = "/mnt/berstorage/techjam/experiments/data"
REPO = "/mnt/berstorage/techjam/aigcdet/data"

#: WildFake -> train. GANs by rule 3; ADM/DDPM/DDIM/VQDM/Imagen by rule 4a.
#: The SDwithAdaptor_* trio is the plan's "SD 1.5 360 [Wildfake]": all three
#: are SD-1.5 derivatives, so rule 1 places them with NTIRE-train's SD 1.5.
WF_TRAIN = {
    "BigGAN": "rule3_gan", "DF-GAN": "rule3_gan", "GALIP": "rule3_gan",
    "GigaGAN": "rule3_gan", "starGAN": "rule3_gan", "styleGAN": "rule3_gan",
    "adm": "rule4a_unet_diffusion", "ddim": "rule4a_unet_diffusion",
    "ddpm": "rule4a_unet_diffusion", "vqdm": "rule4a_unet_diffusion",
    "imagen": "rule4a_unet_diffusion",
    "SDwithAdaptor_lora": "rule1_ntire_train_overlap",
    "SDwithAdaptor_lycris": "rule1_ntire_train_overlap",
    "SDwithAdaptor_controlnet": "rule1_ntire_train_overlap",
}
#: WildFake -> val. Rule 4b: autoregressive / masked-encoder paradigms.
WF_VAL = {
    "VQVAE": "rule4b_autoregressive", "MAE": "rule4b_masked_encoder",
    "MAGE": "rule4b_masked_encoder", "VQGAN": "rule4b_autoregressive",
}
#: Open Images V7 -> train: every family whose lineage NTIRE-train already
#: covers (SD 1.5, SDXL 1.0 + inpainting, Kandinsky 2.2, Z-Image).
OV_TRAIN = {
    "sd15_t2i": "rule1_ntire_train_overlap",
    "sd15_img2img": "rule1_ntire_train_overlap",
    "sdxl_t2i": "rule1_ntire_train_overlap",
    "sdxl_self_cond": "rule1_ntire_train_overlap",
    "sdxl_img2img": "rule1_ntire_train_overlap",
    "kandinsky22_t2i": "rule1_ntire_train_overlap",
    "zimage_t2i": "rule1_ntire_train_overlap",
}
#: Open Images V7 -> val: FLUX.2 Klein by rule 2, the two unseen paradigms
#: (Stage-C compression, linear DiT) by rule 4b.
OV_VAL = {
    "klein4b_t2i": "rule2_ntire_val_overlap",
    "klein4b_ref_image": "rule2_ntire_val_overlap",
    "wuerstchen_t2i": "rule4b_stage_c_compression",
    "sana1600m_t2i": "rule4b_linear_dit",
}
#: Commercial APIs -> val. Rule 2 for the four with an NTIRE-val cousin,
#: rule 4b for the closed-source backbones.
COMMERCIAL_VAL = {
    "ideogram_40_turbo": "rule2_ntire_val_overlap",
    "bytedance_seedream_45": "rule2_ntire_val_overlap",
    "google_nano_banana_2_lite": "rule2_ntire_val_overlap",
    "openai_gpt_image_2": "rule4b_proprietary",
    "meta_muse_image": "rule4b_proprietary",
}
#: Families carved out of val into `test`: read once, never selected on.
#: Lineage-disjoint from what stays in val, which is what makes a test number
#: mean anything after selection has happened on val. VQVAE -> VQGAN -> MAGE
#: travel together: MAGE is masked generative modelling over VQGAN tokens and
#: VQGAN is VQVAE plus adversarial and perceptual losses, so splitting them
#: would put a decoder in test whose own ancestor was selected on.
TEST_FAMILIES = frozenset({
    # The two lineages with no cousin anywhere in train: paella_vq's 42x
    # compression and dc_ae's linear DiT. Plus the proprietary five, which are
    # the closest stand-ins we have for NTIRE-val's closed-source models.
    "wuerstchen_t2i", "sana1600m_t2i",
    "bytedance_seedream_45", "google_nano_banana_2_lite", "ideogram_40_turbo",
    "meta_muse_image", "openai_gpt_image_2",
})

#: The organisers' own labelled sets, and where each goes. `val` is large and
#: is what selection runs against; `val_hard` and `test_public` are the two
#: closest analogues of the private graded set, so they sit in `test` and are
#: read rarely. None carries per-generator labels -- `labels.csv` is
#: name -> binary -- so each enters as one pooled block, exactly as our NTIRE
#: train shards do.
NTIRE_SETS = (
    ("ntire_val", "raw_ntire_val/val_labels.csv",
     "raw_ntire_val/val_images/val_images", "val"),
    # val_hard is image-disjoint from val (verified: 0 of 2,500 overlap) but
    # it is the SAME 13 models, so it stops being a test number the moment
    # selection touches `ntire_val`. It is worth more in val as a hard-case
    # selection signal than as a compromised held-out one.
    ("ntire_val_hard", "raw_ntire_val/val_hard_labels.csv",
     "raw_ntire_val/val_images_hard/val_images_hard", "val"),
    ("ntire_test_public", "raw_ntire_test_public/test_labels.csv",
     "raw_ntire_test_public/test_images/test_images", "test_leaderboard"),
)

#: Splits that are reported SEPARATELY and never pooled, because they answer
#: different questions. `test_transfer` is lineage-disjoint from val, so it
#: estimates transfer to decoders nothing selected on. `test_leaderboard` is
#: the organisers' own draw and is deliberately distribution-MATCHED to val --
#: sharing lineage there is correct, not a defect, because the graded set
#: shares it too. Averaging the two mixes an unbiased estimate with a
#: knowingly optimistic one.
TEST_SPLITS = ("test_transfer", "test_leaderboard")

#: Splits handed over at their own counts and never topped up with our reals:
#: doing so would make them a different benchmark than the one being quoted.
FIXED_SPLITS = ("demo", "test_leaderboard")

#: Sources contributing reals that belong to no pair, and so are free to be
#: allocated to whichever split needs them.
REAL_POOL = ("coco_train2017", "open_images", "ntire", "wildfake")
EXCLUDE_SOURCES = ("sid_set",)

COLS = ["path", "label", "generator", "source", "licence", "width", "height",
        "split", "rel_path", "family", "assign_rule", "image_id", "variant",
        "content_sha256", "is_distorted"]


def _blank(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLS:
        if c not in df:
            df[c] = "" if c not in ("label", "width", "height", "is_distorted") else 0
    return df[COLS]


def from_union() -> pd.DataFrame:
    u = pd.read_parquet(f"{REPO}/manifest_union.parquet")
    u = u[~u.source.isin(EXCLUDE_SOURCES)].copy()
    u["family"] = u.generator.fillna("")
    u["image_id"] = ""
    u["variant"] = "normalized_union"
    u["split"] = ""
    u["assign_rule"] = ""

    fake = u.label == 1
    # NTIRE: we ingested the TRAIN split only (scripts/acquire_ntire_shard.sh
    # pulls ...-train), so rule 1 applies to every NTIRE row without needing
    # per-model labels the dataset does not carry.
    is_ntire = fake & (u.source == "ntire")
    u.loc[is_ntire, ["split", "assign_rule"]] = ["train", "rule1_ntire_train"]
    for table, split in ((WF_TRAIN, "train"), (WF_VAL, "val")):
        for gen, rule in table.items():
            m = fake & (u.source == "wildfake") & (u.generator == gen)
            u.loc[m, ["split", "assign_rule"]] = [split, rule]
    return _blank(u)


def _from_ov7_pairs(pairs: pd.DataFrame, root: str, variant: str) -> pd.DataFrame:
    """One row per image from an OV7 pairs table, real and fake both.

    Both builds are read from their image_id-keyed RAW trees rather than the
    normalised ones. The normalised trees renumber files sequentially per
    directory, which loses the real<->fake correspondence -- and without that
    correspondence a real can land opposite its own counterpart and put the
    same scene on both sides of the split.
    """
    lic = "Reals: CC BY 2.0 (Open Images V7). Generated: see LICENCES.json"
    rows = []
    for r in pairs.itertuples():
        split = "train" if r.family in OV_TRAIN else "val"
        rule = OV_TRAIN.get(r.family) or OV_VAL[r.family]
        rows.append((f"{root}/{r.fake_rel}", 1, r.family, variant, lic,
                     r.width, r.height, split, r.fake_rel, r.family, rule,
                     r.image_id, variant, "", 0))
        rows.append((f"{root}/{r.real_rel}", 0, "", variant, lic,
                     r.width, r.height, split, r.real_rel, "", "paired_real",
                     r.image_id, variant, "", 0))
    return pd.DataFrame(rows, columns=COLS)


def from_ov7_old() -> pd.DataFrame:
    p = pd.read_parquet(f"{REPO}/ov7_upload/pairs.parquet")
    return _from_ov7_pairs(p, f"{REPO}/ov7_upload", "open_images_v7_11978")


def from_ov7_42k() -> pd.DataFrame:
    p = pd.read_csv(f"{EXP}/raw_ov7_42k/pairs.csv")
    return _from_ov7_pairs(p, f"{EXP}/raw_ov7_42k", "open_images_v7_42k")


def from_commercial() -> pd.DataFrame:
    c = pd.read_csv(f"{EXP}/raw_commercial_api/manifest.csv")
    dims = {}
    for rel in c.path:
        with Image.open(f"{EXP}/raw_commercial_api/{rel}") as im:
            dims[rel] = im.size
    rows = []
    for r in c.itertuples():
        w, h = dims[r.path]
        fam = "" if r.kind == "real" else r.family
        rule = "paired_real" if r.kind == "real" else COMMERCIAL_VAL[r.family]
        rows.append((f"{EXP}/raw_commercial_api/{r.path}", int(r.label), fam,
                     "commercial_api" if r.kind != "real" else "open_images_v7_validation",
                     r.licence, w, h, "val", r.path, fam, rule, r.image_id,
                     "commercial_parity_resample", "", 0))
    return pd.DataFrame(rows, columns=COLS)


def from_ntire_sets() -> pd.DataFrame:
    """The organisers' val / val_hard / test_public, read with their labels.

    These arrive already class-balanced and already carrying the challenge's
    own distortion metadata, so they are NOT topped up by `balance_reals` --
    the split-level need it computes simply nets out against the reals they
    bring. `is_distorted` is kept because half of every set is degraded, and a
    number quoted over the mixture hides which half it came from.
    """
    lic = "NTIRE 2026 Robust AI-Generated Image Detection (deepfakesMSU)"
    out = []
    for name, labels, imgdir, split in NTIRE_SETS:
        L = pd.read_csv(f"{EXP}/{labels}")
        rows = []
        for r in L.itertuples():
            rel = f"{imgdir}/{r.image_name}"
            with Image.open(f"{EXP}/{rel}") as im:
                w, h = im.size
            rows.append((f"{EXP}/{rel}", int(r.label), name if r.label else "",
                         name, lic, w, h, split, rel, name if r.label else "",
                         f"ntire_{split}_pooled", "", name, "", int(r.is_distorted)))
        out.append(pd.DataFrame(rows, columns=COLS))
    return pd.concat(out, ignore_index=True)


def from_benchmark() -> pd.DataFrame:
    b = pd.read_parquet(f"{REPO}/benchmark_manifest.parquet").copy()
    # `path` in that file still carries a retired /track5/ root, so it is
    # rebuilt from `rel_path`. See the eval-manifest root mismatch: the train
    # and eval manifests were never rooted at the same place.
    b["path"] = f"{EXP}/demo/" + b.rel_path
    b["family"] = b.generator.fillna("")
    b["image_id"] = ""
    b["variant"] = "benchmark"
    # The organisers' DEMONSTRATION set: forbidden in training and explicitly
    # excluded from the final score. It is therefore neither a selection set
    # nor a test set -- it gets its own split so that nothing can quietly
    # early-stop or pick a threshold on a benchmark that does not count.
    b["split"] = "demo"
    b["assign_rule"] = b.label.map({1: "demo_dalle_advanced", 0: "demo_coco_val2017"})
    return _blank(b)


def balance_reals(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Draw the unpaired reals so train, val and test each land near 1:1.

    `demo` is never balanced: it is the organisers' set at their stated counts,
    and topping it up with our reals would make it a different benchmark.

    Stratified by source, because an unstratified draw would let whichever
    source happens to be largest dominate one split's real class -- and
    `source` is exactly the low-level signature this corpus exists to keep off
    the label.
    """
    pool = df[(df.label == 0) & (df.split == "") & df.source.isin(REAL_POOL)]
    need = {}
    for sp in ("val", "test_transfer", "train"):
        g = df[df.split == sp]
        need[sp] = max(int((g.label == 1).sum() - (g.label == 0).sum()), 0)

    remaining = pool
    for sp in ("val", "test_transfer"):  # the scarce splits are served first
        frac = min(need[sp] / len(remaining), 1.0) if len(remaining) else 0.0
        take = [g.sample(n=min(int(round(len(g) * frac)), len(g)), random_state=seed)
                for _, g in remaining.groupby("source")]
        drawn = pd.concat(take).head(need[sp]) if take else remaining.head(0)
        df.loc[drawn.index, ["split", "assign_rule"]] = [sp, f"balance_{sp}_real"]
        remaining = remaining.drop(drawn.index)

    df.loc[remaining.index, ["split", "assign_rule"]] = ["train", "balance_train_real"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=f"{REPO}/manifest_plan.parquet")
    ap.add_argument("--seed", type=int, default=20260901)
    a = ap.parse_args()

    df = pd.concat([from_union(), from_ov7_old(), from_ov7_42k(),
                    from_commercial(), from_ntire_sets(), from_benchmark()],
                   ignore_index=True)
    # Carve `test` out of val before balancing, so the real draw sees three
    # splits rather than two. A paired real follows its own fake, which is what
    # keeps a scene from landing opposite its counterpart.
    is_test = df.family.isin(TEST_FAMILIES) & (df.split == "val")
    df.loc[is_test, "split"] = "test_transfer"
    test_ids = set(df.loc[is_test, "image_id"]) - {""}
    moved = (df.split == "val") & (df.label == 0) & df.image_id.isin(test_ids)
    df.loc[moved, "split"] = "test_transfer"

    df = balance_reals(df, a.seed)

    unassigned = df[df.split == ""]
    if len(unassigned):
        raise SystemExit(f"{len(unassigned)} rows unassigned:\n"
                         f"{unassigned.groupby(['source', 'generator']).size()}")

    # A real that appears on both sides of a lineage split makes the val score
    # an interpolation, which is the one thing this manifest exists to prevent.
    for key in ("path",):
        dup = df[df.duplicated(key, keep=False)]
        both = dup.groupby(key).split.nunique()
        if (both > 1).any():
            raise SystemExit(f"{int((both > 1).sum())} paths in BOTH splits")

    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}  {len(df):,} rows")


if __name__ == "__main__":
    main()
