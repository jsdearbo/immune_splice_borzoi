import torch
from torch.utils.data import IterableDataset, get_worker_info
import numpy as np
import glob
import random
import os
import gc


def extract_celltype_mapping_from_npz(shard_path):
    """
    Extract the cell type mapping from an NPZ shard file.
    Returns a dict mapping task_idx -> cell_type_name (if available).
    """
    try:
        with np.load(shard_path, mmap_mode="r") as data:
            mapping = {}
            for key in data.files:
                if key.startswith("celltype_idx_"):
                    idx_str = key.replace("celltype_idx_", "")
                    try:
                        idx = int(idx_str)
                        ct_name = str(data[key])
                        mapping[idx] = ct_name
                    except (ValueError, KeyError):
                        pass
            return mapping if mapping else None
    except Exception as e:
        print(f"Warning: Could not extract cell type mapping from {shard_path}: {e}")
        return None


class ShardedNPZDataset(IterableDataset):
    """
    Iterates over a directory of .npz shards.
    Exposes n_tasks attribute for grelu compatibility.
    Performs center-cropping on targets if crop_len_bins is provided.
    """
    def __init__(self, data_dir, split="train", shuffle_shards=True, seed=42, n_tasks=None, crop_len_bins=0, load_mask=True, load_confidence=True, unify_mask=False):
        super().__init__()
        self.data_dir = os.path.join(data_dir, split)
        self.shard_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.n_tasks = n_tasks
        self.crop_len_bins = crop_len_bins
        self.load_mask = load_mask
        self.load_confidence = load_confidence
        self.unify_mask = unify_mask
        self.shard_sizes = []
        self.total_examples = 0
        
        if len(self.shard_paths) == 0:
            raise FileNotFoundError(f"No .npz shards found in {self.data_dir}")
        
        # Auto-detect n_tasks if not provided
        if self.n_tasks is None:
            try:
                # Load only the header keys or a small slice if possible? 
                # npz doesn't support partial load well without loading specific file
                # But we can just load 'targets' to check shape
                with np.load(self.shard_paths[0]) as d:
                    shape = d['targets'].shape
                    # Heuristic: (N, Bins, Tasks) -> Tasks is smallest
                    self.n_tasks = min(shape[1], shape[2])
                    print(f"[{split}] Auto-detected n_tasks={self.n_tasks} from {os.path.basename(self.shard_paths[0])}")
            except Exception as e:
                print(f"[{split}] Warning: Could not auto-detect n_tasks: {e}")
        
        # Try to extract cell type mapping from first shard
        self.celltype_mapping = None
        try:
            self.celltype_mapping = extract_celltype_mapping_from_npz(self.shard_paths[0])
            if self.celltype_mapping:
                print(f"[{split}] Detected cell type mapping from NPZ:")
                for idx in sorted(self.celltype_mapping.keys()):
                    print(f"  Index {idx}: {self.celltype_mapping[idx]}")
            else:
                print(f"[{split}] No cell type mapping found in NPZ files (this is OK for non-conditional models)")
        except Exception as e:
            print(f"[{split}] Warning: Could not extract cell type mapping: {e}")
        
        # # DEBUG: Limit shards for speed
        # if len(self.shard_paths) > 2:
        #      print(f"DEBUG: Limiting shards to 2 of {len(self.shard_paths)} for speed")
        #      self.shard_paths = self.shard_paths[:2]

        for shard_path in self.shard_paths:
            try:
                with np.load(shard_path, mmap_mode="r") as shard:
                    n_records = int(shard["sequence"].shape[0])
            except Exception as exc:
                print(f"[{split}] Warning: failed to inspect {os.path.basename(shard_path)}: {exc}")
                n_records = 0
            self.shard_sizes.append(n_records)
            self.total_examples += n_records

        print(
            f"[{split}] Found {len(self.shard_paths)} shards. "
            f"n_tasks={self.n_tasks}, crop_bins={self.crop_len_bins}, examples={self.total_examples}\n"
            f"[{split}] Config: load_mask={self.load_mask}, load_confidence={self.load_confidence}, unify_mask={self.unify_mask}"
        )

    def __len__(self):
        return self.total_examples

    def _get_shard_list_for_worker(self):
        worker_info = get_worker_info()
        if worker_info is None:
            shards = list(self.shard_paths)
        else:
            per_worker = int(np.ceil(len(self.shard_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.shard_paths))
            shards = self.shard_paths[iter_start:iter_end]
        
        if self.shuffle_shards:
            random.seed(self.seed + (worker_info.id if worker_info else 0))
            random.shuffle(shards)
            
        return shards

    def __iter__(self):
        shard_list = self._get_shard_list_for_worker()
        
        for shard_path in shard_list:
            try:
                # Use mmap to avoid loading full shards into RAM
                with np.load(shard_path, mmap_mode="r") as data:
                    seqs = data["sequence"]
                    targs = data["targets"]
                    masks = data["mask"] if (self.load_mask and "mask" in data) else None
                    confs = data["confidence"] if (self.load_confidence and "confidence" in data) else None

                    n_examples = seqs.shape[0]
                    indices = list(range(n_examples))
                    if self.shuffle_shards:
                        random.shuffle(indices)

                    for i in indices:
                        seq = seqs[i]
                        if seq.ndim == 2 and seq.shape[-1] == 4:
                            seq = seq.T  # (4, L)

                        y = targs[i]
                        if y.ndim == 2 and y.shape[0] > y.shape[1]:
                            y = y.T  # (Tasks, Bins)

                        m = masks[i] if masks is not None else None
                        if m is not None and m.ndim == 2 and m.shape[0] > m.shape[1]:
                            m = m.T  # (Tasks, Bins)
                        
                        if self.unify_mask and m is not None:
                            # Union of masks across tasks: if any task is unmasked (1), all become unmasked.
                            # m is (Tasks, Bins). calculate max over tasks -> (1, Bins)
                            m_any = np.max(m, axis=0, keepdims=True)
                            # Broadcast back to (Tasks, Bins)
                            m = np.broadcast_to(m_any, m.shape).copy()

                        c = confs[i] if confs is not None else None
                        if c is not None and c.ndim == 2 and c.shape[0] > c.shape[1]:
                            c = c.T # (Tasks, Bins)

                        if self.crop_len_bins > 0:
                            n_bins = y.shape[1]
                            start = self.crop_len_bins
                            end = n_bins - self.crop_len_bins
                            if start >= end:
                                raise ValueError(
                                    f"Crop length {self.crop_len_bins} is too large for target length {n_bins}"
                                )
                            y = y[:, start:end]
                            if m is not None:
                                m = m[:, start:end]
                            if c is not None:
                                c = c[:, start:end]

                        x = torch.from_numpy(seq).float()
                        y = torch.from_numpy(y).float()
                        m = torch.from_numpy(m).float() if m is not None else None
                        c = torch.from_numpy(c).float() if c is not None else None

                        # yield tuple (x, y, mask?, confidence?)
                        # Logic: always yield mask if loaded, then confidence if loaded.
                        # Downstream training loop needs to be robust to tuple length.
                        item = [x, y]
                        if self.load_mask:
                            item.append(m)
                        if self.load_confidence:
                            item.append(c)
                        
                        yield tuple(item)
                
                # Explicit garbage collection helps if shards are massive
                # gc.collect() 
            
            except Exception as e:
                print(f"Error reading shard {shard_path}: {e}")
                continue


class TaskFilterWrapper(IterableDataset):
    """
    Wraps a dataset to filter targets to only specific task indices.
    Useful when training on a subset of cell types from a multitask dataset.
    
    Example: if dataset has 3 tasks but config specifies only task 0,
    this wrapper extracts y[:, :] -> y[0:1, :] (shape [1, L] instead of [3, L])
    """
    def __init__(self, base_ds, task_indices):
        super().__init__()
        self.base = base_ds
        self.task_indices = task_indices if isinstance(task_indices, (list, tuple)) else [task_indices]
        # Update n_tasks to reflect filtered output
        self.n_tasks = len(self.task_indices)

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        for batch in self.base:
            # batch is (x, y, mask?, conf?) where y is (T, L)
            x = batch[0]
            y = batch[1]     # (T, L)
            m = batch[2] if len(batch) >= 3 else None
            c = batch[3] if len(batch) >= 4 else None
            
            # Filter to only relevant tasks
            y_filtered = y[self.task_indices, :]  # (filtered_T, L)
            m_filtered = m[self.task_indices, :] if m is not None else None
            c_filtered = c[self.task_indices, :] if c is not None else None
            
            # Construct filtered item
            item = [x, y_filtered]
            if m is not None: item.append(m_filtered)
            if c is not None: item.append(c_filtered)
            
            yield tuple(item)


class ConditionalCellTypeWrapper(IterableDataset):
    """
    Wraps ShardedNPZDataset to emit conditional samples:
      (x, y_t, mask_t?, conf_t?, cell_type_id)
    where y_t is a single task slice (1, L).
    
    mode:
      - "random": choose one cell type per example
      - "expand": emit all cell types per example (3x samples)
    """
    def __init__(self, base_ds, mode="random", seed=123):
        super().__init__()
        self.base = base_ds
        self.n_tasks = base_ds.n_tasks
        self.mode = mode
        self.seed = seed

    def __len__(self):
        # IterableDataset length is approximate
        if self.mode == "expand":
            return len(self.base) * self.n_tasks
        return len(self.base)

    def __iter__(self):
        rng = random.Random(self.seed)
        for batch in self.base:
            # base yields: (x, y, mask?, conf?)
            x = batch[0]
            y = batch[1]     # (T, L)
            m = batch[2] if len(batch) >= 3 else None
            c = batch[3] if len(batch) >= 4 else None

            if self.mode == "random":
                t = rng.randrange(self.n_tasks)
                y_t = y[t:t+1, :]
                m_t = m[t:t+1, :] if m is not None else None
                c_t = c[t:t+1, :] if c is not None else None
                
                # Construct item
                item = [x, y_t]
                if m is not None: item.append(m_t)
                if c is not None: item.append(c_t)
                item.append(torch.tensor(t, dtype=torch.long))
                
                yield tuple(item)
            else:  # "expand"
                for t in range(self.n_tasks):
                    y_t = y[t:t+1, :]
                    m_t = m[t:t+1, :] if m is not None else None
                    c_t = c[t:t+1, :] if c is not None else None
                    
                    item = [x, y_t]
                    if m is not None: item.append(m_t)
                    if c is not None: item.append(c_t)
                    item.append(torch.tensor(t, dtype=torch.long))

                    yield tuple(item)
