"""
Agnostic Incremental Clustering Agent — apply agent-provided label/summary immediately
- When agent.run returns assign_to + optional label/summary/confidence, we use them safely:
  * immediate apply if confidence >= threshold OR cluster.auto_label True
  * otherwise keep current label but still run _review_cluster after append
"""

from __future__ import annotations
from typing import List, Dict, Callable, Optional, Any, Iterable, Protocol, Tuple
import json
import asyncio
import logging
import random
import re
import unicodedata

import numpy as np
from pydantic import BaseModel

# logging
logger = logging.getLogger("incremental_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)

SIM_EPSILON = 1e-12


# ---------------------------
# Protocols (duck-typing / agnostic)
# ---------------------------
class EmbeddingModelProtocol(Protocol):
    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        """Return an array shape (n, d). If normalize_embeddings=True the vectors should be unit norm."""


class AgentProtocol(Protocol):
    async def run(self, prompt: str) -> Any:
        """Return an object. The implementation may expose `.output` or return raw text."""


# ---------------------------
# Cluster data model (pydantic)
# Keep centroid serializable (List[float]) and confidences
# ---------------------------
class Cluster(BaseModel):
    members: List[str]
    label: str
    summary: str
    centroid: List[float] = []
    label_confidence: float = 0.0
    summary_confidence: float = 0.0
    auto_label: bool = False  # True if label was auto generated fallback

    def get_centroid_np(self) -> np.ndarray:
        return np.array(self.centroid, dtype=float)

    def set_centroid_np(self, vec: np.ndarray) -> None:
        self.centroid = vec.astype(float).tolist()


# ---------------------------
# Utilities
# ---------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v) + SIM_EPSILON
    return v / norm


def default_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a / (np.linalg.norm(a) + SIM_EPSILON)
    b = b / (np.linalg.norm(b) + SIM_EPSILON)
    return float(np.dot(a, b))


async def _run_with_retries(agent: AgentProtocol, prompt: str, retries: int = 2, backoff: float = 0.5):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            res = await agent.run(prompt)
            return res
        except Exception as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(backoff * (2 ** attempt) + random.random() * 0.1)
                continue
            raise last_exc


def safe_parse_json(maybe_obj: Any) -> Optional[Dict[str, Any]]:
    text = None
    if maybe_obj is None:
        return None
    if hasattr(maybe_obj, "output"):
        try:
            out = maybe_obj.output
            if isinstance(out, dict):
                return out
            text = str(out)
        except Exception:
            text = str(maybe_obj)
    else:
        text = str(maybe_obj)

    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def sanitize_label_summary(label: Optional[str], summary: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    def clean_text(t: Optional[str]) -> Optional[str]:
        if not t or not isinstance(t, str):
            return None
        t = unicodedata.normalize("NFKC", t).strip()
        t = re.sub(r"\s+", " ", t)
        t = t.strip(' "\'')
        return t if t else None

    label = clean_text(label)
    summary = clean_text(summary)

    # label rules
    if label:
        words = label.split()
        if len(words) > 4:
            label = " ".join(words[:4])
        if label.lower() in {"diversos", "outros", "plantas", "vários", "vários cultivos", "cultivos diversos", "diverso"}:
            label = None
        if len(label) > 60 or re.search(r"https?:\/\/|www\.", label):
            label = None

    # summary rules
    if summary:
        if len(summary) > 400:
            summary = summary[:400]
            if "." in summary:
                summary = summary.rsplit(".", 1)[0] + "."
    return label, summary


def auto_generate_label(doc_text: str) -> str:
    stop = {"de", "do", "da", "para", "e", "com", "em", "o", "a", "os", "as", "por", "no", "na"}
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", unicodedata.normalize("NFKC", doc_text))
    tokens = [t.capitalize() for t in tokens if t.lower() not in stop]
    if not tokens:
        return (doc_text[:40]).strip()
    label = tokens[0]
    if len(tokens) > 1:
        label = f"{label} {tokens[1]}"
    return label[:40]


# ---------------------------
# Default prompt template (Portuguese)
# ---------------------------
def DEFAULT_PROMPT_TEMPLATE(doc_text: str, clusters_info: List[Dict[str, Any]]) -> str:
    return f"""
Você é um especialista em classificação semântica de documentos.
Dado o documento a seguir (texto curto):
\"\"\"{doc_text}\"\"\"

E o estado atual dos clusters (id, label, summary, representative):
{json.dumps(clusters_info, ensure_ascii=False, indent=2)}

Decida se o documento deve:
1) Ser atribuído a um cluster existente (informe o id)
2) Ou criar um novo cluster

Retorne SOMENTE um JSON válido com as chaves:
- assign_to: <id do cluster (int) ou "new">
- label: <opcional, nova label para o cluster>
- summary: <opcional, resumo para o cluster>
- confidence: <opcional, número 0.0-1.0 indicando confiança na sugestão>

Exemplo de retorno:
{{"assign_to": "new", "label": "Oleaginosas", "summary": "Cultivos para produção de óleo.", "confidence": 0.8}}
"""


# ---------------------------
# IncrementalClusteringAgent
# ---------------------------
# ---------------------------
# IncrementalClusteringAgent (updated)
# ---------------------------
class IncrementalClusteringAgent:
    def __init__(
        self,
        agent: AgentProtocol,
        embedding_model: EmbeddingModelProtocol,
        *,
        sim_threshold: float = 0.75,
        llm_threshold: float = 0.45,
        similarity_fn: Callable[[np.ndarray, np.ndarray], float] = default_similarity,
        prompt_template: Callable[[str, List[Dict[str, Any]]], str] = DEFAULT_PROMPT_TEMPLATE,
        llm_retries: int = 2,
        llm_backoff: float = 0.5,
        update_labels_via_llm: bool = True,
    ):
        self.agent = agent
        self.embedding_model = embedding_model
        self.sim_threshold = sim_threshold
        self.llm_threshold = llm_threshold
        self.similarity_fn = similarity_fn
        self.prompt_template = prompt_template
        self.llm_retries = llm_retries
        self.llm_backoff = llm_backoff
        self.update_labels_via_llm = update_labels_via_llm

        self.clusters: Dict[int, Cluster] = {}
        self.next_cluster_id = 0

    # -------------------------
    # low-level helpers
    # -------------------------
    def _clusters_info_for_prompt(self, limit: int = 10) -> List[Dict[str, Any]]:
        info = []
        for cid, cluster in list(self.clusters.items())[-limit:]:
            rep = cluster.members[0] if cluster.members else ""
            info.append({"id": cid, "label": cluster.label, "summary": cluster.summary, "representative": rep})
        return info

    def _find_best_cluster(self, doc_emb: np.ndarray):
        best_cid, best_sim = None, -1.0
        for cid, cluster in self.clusters.items():
            centroid = cluster.get_centroid_np()
            sim = self.similarity_fn(doc_emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cid = cid
        return best_cid, float(best_sim)

    def _create_cluster(self, doc_text: str, doc_emb: np.ndarray, label: Optional[str] = None, summary: Optional[str] = None) -> int:
        cid = self.next_cluster_id
        self.next_cluster_id += 1
        cluster = Cluster(
            members=[doc_text],
            label=label or doc_text[:40],
            summary=summary or doc_text[:160],
            centroid=[]
        )
        cluster.set_centroid_np(normalize(doc_emb))
        self.clusters[cid] = cluster
        logger.info("Created cluster %d label=%s", cid, cluster.label)
        return cid

    def _append_to_cluster(self, cid: int, doc_text: str, doc_emb: np.ndarray):
        cluster = self.clusters[cid]
        cluster.members.append(doc_text)
        old_centroid = cluster.get_centroid_np()
        n = len(cluster.members)
        new_centroid = (old_centroid * (n - 1) + doc_emb) / n
        cluster.set_centroid_np(normalize(new_centroid))
        logger.info("Appended to cluster %d new_size=%d", cid, n)

    async def _update_cluster_label_summary(self, cid: int):
        """
        Regenerate cluster label and summary using the agent.
        """
        cluster = self.clusters[cid]
        if not self.update_labels_via_llm:
            return

        prompt = f"""
Você é um especialista em classificação semântica de documentos.
Dado o cluster a seguir com os documentos:
{json.dumps(cluster.members, ensure_ascii=False, indent=2)}

Forneça:
- Um rótulo curto representativo do cluster
- Um resumo conciso do cluster

Retorne JSON válido:
{{"label": "...", "summary": "..."}}
"""
        try:
            raw = await _run_with_retries(self.agent, prompt, retries=self.llm_retries, backoff=self.llm_backoff)
            parsed = safe_parse_json(raw)
            if parsed:
                cluster.label = parsed.get("label", cluster.label)
                cluster.summary = parsed.get("summary", cluster.summary)
                logger.info("Cluster %d label updated -> %s", cid, cluster.label)
                logger.info("Cluster %d summary updated -> %s", cid, cluster.summary)
        except Exception as e:
            logger.warning("Failed to update cluster %d label/summary: %s", cid, e)

    async def _ask_llm_assignment(self, doc_text: str) -> Dict[str, Any]:
        clusters_info = self._clusters_info_for_prompt()
        prompt = self.prompt_template(doc_text, clusters_info)
        raw = await _run_with_retries(self.agent, prompt, retries=self.llm_retries, backoff=self.llm_backoff)
        parsed = safe_parse_json(raw)
        if parsed is None:
            logger.warning("LLM returned unparsable result, falling back to new cluster")
            return {"assign_to": "new"}
        return parsed

    # -------------------------
    # Public API
    # -------------------------
    async def add_document(self, doc_text: str) -> int:
        if not doc_text:
            raise ValueError("doc_text empty")

        # 1️⃣ Encode document
        emb = self.embedding_model.encode([doc_text], normalize_embeddings=True)[0]

        # 2️⃣ Find best existing cluster by similarity
        best_cid, best_sim = self._find_best_cluster(emb)

        # 3️⃣ High similarity → append without asking LLM
        if best_cid is not None and best_sim >= self.sim_threshold:
            self._append_to_cluster(best_cid, doc_text, emb)
            # do NOT overwrite label/summary blindly
            return best_cid

        # 4️⃣ Otherwise, ask LLM for assignment
        parsed = await self._ask_llm_assignment(doc_text)
        assign_to = parsed.get("assign_to", "new")
        llm_label, llm_summary = sanitize_label_summary(parsed.get("label"), parsed.get("summary"))
        confidence = float(parsed.get("confidence", 0.0))

        # normalize assign_to
        if isinstance(assign_to, (int, float)) and not isinstance(assign_to, bool):
            assign_to = int(assign_to)

        # 5️⃣ Create new cluster if needed
        if assign_to == "new" or assign_to is None:
            cid = self._create_cluster(doc_text, emb, label=llm_label, summary=llm_summary)
            if confidence >= self.llm_threshold:
                self.clusters[cid].auto_label = True
        else:
            cid = int(assign_to)
            if cid not in self.clusters:
                # LLM suggested unknown cluster → fallback to new cluster
                cid = self._create_cluster(doc_text, emb, label=llm_label, summary=llm_summary)
                if confidence >= self.llm_threshold:
                    self.clusters[cid].auto_label = True
            else:
                # Append to existing cluster
                cluster = self.clusters[cid]
                self._append_to_cluster(cid, doc_text, emb)

                # Apply LLM label/summary if confidence high or cluster.auto_label
                if (confidence >= self.llm_threshold) or cluster.auto_label:
                    if llm_label:
                        cluster.label = llm_label
                    if llm_summary:
                        cluster.summary = llm_summary
                    cluster.auto_label = True
                    logger.info("Cluster %d label/summary updated via LLM (conf %.2f)", cid, confidence)

        # 6️⃣ Optionally, review cluster periodically (not after every append)
        # await self._update_cluster_label_summary(cid)

        return cid


    async def add_documents(self, docs: Iterable[str], concurrency: int = 4) -> List[int]:
        sem = asyncio.Semaphore(concurrency)
        docs_list = list(docs)
        results: List[Optional[int]] = [None] * len(docs_list)

        async def worker(i: int, text: str):
            async with sem:
                try:
                    cid = await self.add_document(text)
                    results[i] = cid
                except Exception as e:
                    logger.exception("Failed adding document: %s", e)
                    results[i] = None

        tasks = [asyncio.create_task(worker(i, t)) for i, t in enumerate(docs_list)]
        await asyncio.gather(*tasks)
        return [int(x) if x is not None else -1 for x in results]

    def export_state(self) -> Dict[str, Any]:
        return {
            "next_cluster_id": self.next_cluster_id,
            "clusters": {str(cid): c.model_dump() for cid, c in self.clusters.items()}
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        self.next_cluster_id = int(state.get("next_cluster_id", 0))
        clusters_raw = state.get("clusters", {})
        self.clusters = {}
        for cid_s, raw in clusters_raw.items():
            cid = int(cid_s)
            c = Cluster.model_validate(raw)
            self.clusters[cid] = c



# ---------------------------
# Minimal demo (pseudo-code)
# ---------------------------
if __name__ == "__main__":
    async def demo():
        from sentence_transformers import SentenceTransformer

        class DumbAgent:
            async def run(self, prompt: str):
                # naive heuristic: if 'milho' appears in prompt, propose label 'Milho' with high confidence
                if "milho" in prompt:
                    return '{"assign_to": "new", "label":"Milho","summary":"Cultivo de milho.","confidence":0.9}'
                # safe default low-confidence suggestion
                return '{"assign_to": "new", "label":"Agronegócio","summary":"Atividades agrícolas diversas.","confidence":0.4}'

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        agent = DumbAgent()
        inc = IncrementalClusteringAgent(agent, embedding_model)

        docs = ["Cultivo de milho", "Cultivo de soja", "Criação de bovinos para corte", "Cultivo de trigo"]
        for d in docs:
            cid = await inc.add_document(d)
            print("assigned", d, cid)

        print("state:", json.dumps(inc.export_state(), ensure_ascii=False, indent=2))

    asyncio.run(demo())
