"""MedSigLIP image/text encoding and FAISS storage."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from .image_utils import load_rgb_image
from .io_utils import l2_normalize, read_json, write_json
from .schemas import CaseRecord


class MedSigLIPEncoder:
    def __init__(self, model_path: str, *, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)
        local = Path(model_path).exists()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=local)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype if self.device.startswith("cuda") else torch.float32,
            local_files_only=local,
        ).to(self.device)
        self.model.eval()

    def encode_paths(
        self,
        image_paths: Sequence[str],
        batch_size: int = 16,
        *,
        show_progress: bool = True,
        progress_desc: str = "MedSigLIP image embeddings",
    ) -> np.ndarray:
        batches: list[np.ndarray] = []
        starts = range(0, len(image_paths), batch_size)
        iterator = (
            tqdm(starts, desc=progress_desc)
            if show_progress
            else starts
        )
        for start in iterator:
            paths = image_paths[start : start + batch_size]
            images: list[Image.Image] = []
            try:
                for path in paths:
                    images.append(load_rgb_image(path))
                inputs = self.processor(images=images, return_tensors="pt")
                image_inputs = {k: v.to(self.device) for k, v in inputs.items() if "pixel" in k}
                with self.torch.inference_mode():
                    if hasattr(self.model, "get_image_features"):
                        features = self.model.get_image_features(**image_inputs)
                        features = self._unwrap_features(features, "image")
                    else:
                        outputs = self.model(**image_inputs)
                        features = self._unwrap_features(outputs, "image")
                batches.append(l2_normalize(features.detach().float().cpu().numpy()).astype(np.float32))
            finally:
                for image in images:
                    image.close()
        if not batches:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def encode_one(self, image_path: str) -> np.ndarray:
        # batch-retrieve already owns the query-level progress bar. Suppressing
        # this nested one-item tqdm avoids printing a new 1/1 bar per query.
        return self.encode_paths(
            [image_path],
            batch_size=1,
            show_progress=False,
        )[0]

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Encode text with the MedSigLIP text tower.

        This is used only as an optional ontology-label ranker. It does not
        generate an answer or inspect the query image.
        """
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(self.device) for k, v in inputs.items() if "pixel" not in k}
            with self.torch.inference_mode():
                if hasattr(self.model, "get_text_features"):
                    features = self.model.get_text_features(**text_inputs)
                    features = self._unwrap_features(features, "text")
                else:
                    outputs = self.model(**text_inputs)
                    features = self._unwrap_features(outputs, "text")
            batches.append(l2_normalize(features.detach().float().cpu().numpy()).astype(np.float32))
        if not batches:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def _unwrap_features(self, features, modality: str):
        is_tensor = getattr(self.torch, "is_tensor", None)

        def tensor_like(value) -> bool:
            if callable(is_tensor):
                return bool(is_tensor(value))
            return all(
                callable(getattr(value, method, None))
                for method in ("detach", "float", "cpu", "numpy")
            )

        if tensor_like(features):
            return features
        direct = getattr(features, f"{modality}_embeds", None)
        if tensor_like(direct):
            return direct
        pooled = getattr(features, "pooler_output", None)
        if tensor_like(pooled):
            return pooled
        raise RuntimeError(
            f"MedSigLIP {modality} feature output has unsupported type: "
            f"{type(features).__name__}"
        )


class VisualStore:
    def __init__(self, visual_dir: str | Path, cfg: dict):
        self.dir = Path(visual_dir)
        self.cfg = cfg
        index_kind = str(cfg.get("faiss_index", "flat_ip")).lower()
        if index_kind != "flat_ip":
            raise ValueError(
                "This release materializes only exact FAISS IndexFlatIP; "
                f"received visual_index.faiss_index={index_kind!r}"
            )

        import faiss

        self.faiss = faiss
        self.case_ids: list[str] = read_json(self.dir / cfg["case_ids_file"])
        self.id_to_row = {case_id: i for i, case_id in enumerate(self.case_ids)}
        self.embeddings = np.load(self.dir / cfg["embedding_file"], mmap_mode="r")
        self.index = faiss.read_index(str(self.dir / cfg["index_file"]))

    @classmethod
    def build(
        cls,
        cases: Sequence[CaseRecord],
        encoder: MedSigLIPEncoder,
        visual_dir: str | Path,
        cfg: dict,
    ) -> "VisualStore":
        index_kind = str(cfg.get("faiss_index", "flat_ip")).lower()
        if index_kind != "flat_ip":
            raise ValueError(
                "This release materializes only exact FAISS IndexFlatIP; "
                f"received visual_index.faiss_index={index_kind!r}"
            )

        import faiss
        out_dir = Path(visual_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [case.image_path for case in cases]
        embeddings = encoder.encode_paths(image_paths, batch_size=int(cfg.get("batch_size", 16)))
        if embeddings.size == 0:
            raise ValueError("No embeddings were produced")
        embeddings = l2_normalize(embeddings.astype(np.float32))

        np.save(out_dir / cfg["embedding_file"], embeddings.astype(np.float16))
        write_json(out_dir / cfg["case_ids_file"], [case.case_id for case in cases])
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(out_dir / cfg["index_file"]))
        return cls(out_dir, cfg)

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        query = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        scores, rows = self.index.search(query, min(k, len(self.case_ids)))
        out: list[tuple[str, float]] = []
        for row, score in zip(rows[0].tolist(), scores[0].tolist()):
            if row >= 0:
                out.append((self.case_ids[row], float(score)))
        return out

    def neighbors_for_case(self, case_id: str, k: int) -> list[tuple[str, float]]:
        row = self.id_to_row[case_id]
        query = np.asarray(self.embeddings[row], dtype=np.float32)
        return self.search(query, k)

    def similarity_to_query(self, query_embedding: np.ndarray, case_id: str) -> float:
        row = self.id_to_row[case_id]
        candidate = np.asarray(self.embeddings[row], dtype=np.float32)
        query = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
        return float(np.dot(query, candidate))

    def similarity_between(self, case_a: str, case_b: str) -> float:
        a = np.asarray(self.embeddings[self.id_to_row[case_a]], dtype=np.float32)
        b = np.asarray(self.embeddings[self.id_to_row[case_b]], dtype=np.float32)
        return float(np.dot(a, b))
