# save as: gt_create_kdtree.py
# for male groundtruth using other datas
import os, json, argparse
import numpy as np
import trimesh
from sklearn.neighbors import KDTree


# -------------------- Utility --------------------

def load_vertices(path):
    """Robust OBJ loader."""
    if not os.path.exists(path):
        print(f"❌ Missing OBJ file: {path}")
        return None

    try:
        m = trimesh.load(path, process=False)
    except Exception as e:
        print(f"❌ Failed to load mesh: {path}  ({e})")
        return None

    if hasattr(m, "vertices"):
        return np.asarray(m.vertices, dtype=np.float64)

    # Scene case
    try:
        m = m.dump().sum()
        return np.asarray(m.vertices, dtype=np.float64)
    except:
        print(f"❌ Invalid mesh format: {path}")
        return None


def assign_fdi_order(jaw, n):
    """FDI numbering."""
    if jaw == "lower":
        seq = list(range(31, 39)) + list(range(41, 49))
    else:
        seq = list(range(11, 19)) + list(range(21, 29))
    return seq[:n]


def principal_axis(points):
    """Compute PCA first axis for ordering."""
    c = points.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(points - c, full_matrices=False)
    return Vt[0]


# -------------------- Main GT Builder --------------------

def build_gt_for_jaw(case_name, jaw, base_obj, seg_dir, eps, save_json):
    print(f"\n[INFO] Processing {case_name} / {jaw}")

    if not os.path.exists(base_obj):
        print(f"❌ Base OBJ missing: {base_obj}")
        return False
    if not os.path.isdir(seg_dir):
        print(f"❌ Seg folder missing: {seg_dir}")
        return False

    # Load base mesh
    V_base = load_vertices(base_obj)
    if V_base is None:
        return False

    N = len(V_base)
    labels = np.zeros(N, dtype=np.int32)
    instances = np.zeros(N, dtype=np.int32)

    # Find segmentation OBJ files
    seg_files = [
        f for f in os.listdir(seg_dir)
        if f.startswith(f"{case_name}_{jaw}_seg_label_") and f.endswith(".obj")
    ]

    if not seg_files:
        print(f"⚠ No segmentation files found → creating empty JSON")
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        json.dump({
            "id_patient": case_name,
            "jaw": jaw,
            "labels": labels.tolist(),
            "instances": instances.tolist(),
        }, open(save_json, "w"))
        return True

    # Sort segmentation files by index
    pairs = []
    for fname in seg_files:
        try:
            idx = int(fname.split("_")[-1].split(".")[0])
            pairs.append((idx, os.path.join(seg_dir, fname)))
        except:
            print(f"⚠ Skip invalid seg filename: {fname}")
            continue

    pairs.sort(key=lambda x: x[0])
    tooth_files = [(idx, p) for idx, p in pairs if idx != 0]

    # If only gingiva exists
    if not tooth_files:
        print(f"⚠ Only gingiva found → labeling all as 0")
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        json.dump({
            "id_patient": case_name,
            "jaw": jaw,
            "labels": labels.tolist(),
            "instances": instances.tolist(),
        }, open(save_json, "w"))
        return True

    # KDTree assignment
    min_dist = np.full(N, np.inf)
    min_inst = np.zeros(N, dtype=np.int32)

    centroids = []
    inst_id = 0

    for tooth_idx, path in tooth_files:
        V_tooth = load_vertices(path)
        if V_tooth is None or len(V_tooth) == 0:
            print(f"⚠ Skip invalid/empty tooth: {path}")
            inst_id += 1
            continue

        try:
            tree = KDTree(V_tooth)
            d, _ = tree.query(V_base, k=1)
            d = d[:, 0]
        except Exception as e:
            print(f"⚠ KDTree failed for {path} ({e})")
            inst_id += 1
            continue

        centroids.append((tooth_idx, V_tooth.mean(axis=0)))

        mask = d < min_dist
        min_dist[mask] = d[mask]
        min_inst[(d <= eps) & mask] = inst_id + 1

        inst_id += 1

    instances = min_inst.copy()

    # FDI number assign
    if centroids:
        ids, cxyz = zip(*centroids)
        C = np.vstack(cxyz)
        axis = principal_axis(C)
        proj = C @ axis
        order = np.argsort(proj)
        ordered_ids = [ids[i] for i in order]

        fdi_seq = assign_fdi_order(jaw, len(ordered_ids))
        toothid_to_fdi = {tid: fdi_seq[i] for i, tid in enumerate(ordered_ids)}
        toothid_to_inst = {t[0]: i+1 for i, t in enumerate(tooth_files)}

        for tid in ordered_ids:
            inst = toothid_to_inst.get(tid, None)
            if inst is not None:
                labels[instances == inst] = toothid_to_fdi[tid]

    # Save JSON
    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    json.dump({
        "id_patient": case_name,
        "jaw": jaw,
        "labels": labels.tolist(),
        "instances": instances.tolist(),
    }, open(save_json, "w"))

    print(f"✔ Done {case_name}/{jaw} → verts={N}, unique_labels={np.unique(labels)}")
    return True


# -------------------- Main Runner --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_obj_data_path", required=True)
    ap.add_argument("--source_json_data_path", required=True)
    ap.add_argument("--save_data_path", required=True)
    ap.add_argument("--eps", type=float, default=0.2)
    args = ap.parse_args()

    for case in sorted(os.listdir(args.source_obj_data_path)):
        case_dir = os.path.join(args.source_obj_data_path, case)
        if not os.path.isdir(case_dir):
            continue

        for jaw in ["lower", "upper"]:
            base_obj = os.path.join(case_dir, f"{case}_{jaw}.obj")
            seg_dir = os.path.join(args.source_json_data_path, case, jaw)
            save_json = os.path.join(args.save_data_path, case, f"{case}_{jaw}.json")

            try:
                build_gt_for_jaw(case, jaw, base_obj, seg_dir, args.eps, save_json)
            except Exception as e:
                print(f"❌ Error processing {case}/{jaw}: {e}")


if __name__ == "__main__":
    main()
