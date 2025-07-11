# ======================== 1. INSTALLATION DES LIBRAIRIES ========================
#!pip install torch torchvision torchaudio --quiet
#!pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html --quiet
#!pip install py2neo pandas sentence-transformers fuzzywuzzy --quiet

# ======================== 2. IMPORTS ========================
import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from py2neo import Graph, NodeMatcher, Relationship
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import numpy as np

# ======================== 3. CONNEXION À NEO4J ========================
uri = "neo4j+s://1cb37128.databases.neo4j.io"
user = "neo4j"
password = "qUocbHeI6RTR3sqwFE6IhnAX5nk9N_KnQVFthB3E9S8"
graph = Graph(uri, auth=(user, password))
matcher = NodeMatcher(graph)

try:
    info = graph.run("RETURN 1").data()
    print("✅ Connexion Neo4j réussie :", info)
except Exception as e:
    print("❌ Erreur de connexion Neo4j :", e)
    exit(1)

# ======================== 4. ALIGNEMENT DES GRAPHES CSKG1 ET CSKG2 ========================
def align_and_fuse_graphs():
    print("\n🔗 Alignement des graphes CSKG1 (NVD) et CSKG2 (Nessus)...")

    # 1. Alignement des CVE
    cve_nvd = list(graph.nodes.match("CVE").where("_.source = 'NVD'"))
    cve_nessus = list(graph.nodes.match("CVE").where("_.source = 'Nessus'"))
    nvd_dict = {cve["name"]: cve for cve in cve_nvd}

    model = SentenceTransformer("all-MiniLM-L6-v2")

    count_exact, count_fuzzy, count_embed = 0, 0, 0

    for nessus_cve in cve_nessus:
        name = nessus_cve.get("name")
        if not name:
            continue

        # 1.1. Correspondance exacte
        if name in nvd_dict:
            graph.merge(Relationship(nessus_cve, "SAME_AS", nvd_dict[name]))
            count_exact += 1
            continue

        # 1.2. Correspondance floue
        best_match, best_score = None, 0
        for nvd_name in nvd_dict:
            score = fuzz.ratio(name, nvd_name)
            if score > best_score:
                best_score = score
                best_match = nvd_dict[nvd_name]

        if best_score > 90:
            graph.merge(Relationship(nessus_cve, "SAME_AS", best_match))
            count_fuzzy += 1
            continue

        # 1.3. Correspondance sémantique des descriptions
        desc1 = nessus_cve.get("description", "")
        desc2 = best_match.get("description", "") if best_match else ""

        if desc1 and desc2:
            emb1 = model.encode(desc1, convert_to_tensor=True)
            emb2 = model.encode(desc2, convert_to_tensor=True)
            sim = util.cos_sim(emb1, emb2).item()

            if sim > 0.85:
                graph.merge(Relationship(nessus_cve, "SAME_AS", best_match))
                count_embed += 1

    print(f"  Alignement CVE : {count_exact} exacts, {count_fuzzy} fuzzy, {count_embed} embeddings.")

    # 2. Fusion des noeuds CVE
    print("\n🔀 Fusion des noeuds CVE alignés...")
    graph.run("""
    MATCH (c1:CVE)-[:SAME_AS]->(c2:CVE)
    WHERE c1.source = 'Nessus' AND c2.source = 'NVD'
    SET c2.description = COALESCE(c2.description, c1.description),
        c2.cvssScore = COALESCE(c2.cvssScore, c1.cvssScore),
        c2.lastUpdated = COALESCE(c2.lastUpdated, c1.lastUpdated)
    """)

    # 3. Mise à jour des relations des plugins Nessus vers les CVE NVD
    print("\n🔄 Mise à jour des relations des plugins vers les CVE NVD...")
    graph.run("""
    MATCH (p:Plugin)-[r:detects]->(c1:CVE {source: 'Nessus'})-[:SAME_AS]->(c2:CVE {source: 'NVD'})
    MERGE (p)-[:detects]->(c2)
    DELETE r
    """)

    # 4. Alignement des produits et vendeurs
    print("\n🏷️ Alignement des produits et vendeurs...")
    for label in ["Product", "Vendor"]:
        graph.run(f"""
        MATCH (n1:{label} {{source: 'Nessus'}}), (n2:{label} {{source: 'NVD'}})
        WHERE n1.name = n2.name
        MERGE (n1)-[:SAME_AS]->(n2)
        """)

# ======================== 5. EXTRACTION DES TRIPLETS FUSIONNÉS ========================
def extract_fused_triplets():
    print("\n🔍 Extraction des triplets du graphe fusionné...")

    query = """
    MATCH (h)-[r]->(t)
    WHERE h.name IS NOT NULL AND t.name IS NOT NULL
    RETURN h.name AS head, type(r) AS relation, t.name AS tail,
           labels(h) AS head_labels, labels(t) AS tail_labels
    """

    results = graph.run(query).data()
    triplets_df = pd.DataFrame(results)

    if triplets_df.empty:
        raise ValueError("Aucun triplet récupéré depuis Neo4j.")

    print(f"✅ {len(triplets_df)} triplets extraits")
    return triplets_df

# ======================== 6. ENCODAGE DES ENTITÉS ========================
def encode_entities(triplets_df):
    print("\n🔢 Encodage des entités et relations...")

    entities = pd.Series(pd.concat([triplets_df["head"], triplets_df["tail"]]).unique()).reset_index()
    entity2id = dict(zip(entities[0], entities["index"]))
    relations = pd.Series(triplets_df["relation"].unique()).reset_index()
    rel2id = dict(zip(relations[0], relations["index"]))

    # Ajout de la relation à prédire si elle n'existe pas
    if "at_risk_of" not in rel2id:
        rel2id["at_risk_of"] = len(rel2id)

    # Construction des index de relation
    h_idx = torch.tensor([entity2id[h] for h in triplets_df["head"]])
    r_idx = torch.tensor([rel2id[r] for r in triplets_df["relation"]])
    t_idx = torch.tensor([entity2id[t] for t in triplets_df["tail"]])

    return entity2id, rel2id, h_idx, r_idx, t_idx

# ======================== 7. MODELE RotatE ========================
class RotatEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=64):
        super().__init__()
        self.ent = nn.Embedding(num_entities, embedding_dim)
        self.rel = nn.Embedding(num_relations, embedding_dim)

        # Initialisation de Xavier
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, h_idx, r_idx, t_idx):
        pi = 3.141592653589793
        h = self.ent(h_idx)
        r = self.rel(r_idx) * pi
        t = self.ent(t_idx)

        r_complex = torch.stack([torch.cos(r), torch.sin(r)], dim=-1)
        h_complex = torch.stack([h, torch.zeros_like(h)], dim=-1)

        h_r = torch.stack([
            h_complex[..., 0] * r_complex[..., 0] - h_complex[..., 1] * r_complex[..., 1],
            h_complex[..., 0] * r_complex[..., 1] + h_complex[..., 1] * r_complex[..., 0]
        ], dim=-1)

        t_complex = torch.stack([t, torch.zeros_like(t)], dim=-1)
        score = -torch.norm(h_r - t_complex, dim=-1).sum(dim=-1)
        return score


# ======================== 8. PRÉDICTION DES RELATIONS HOST-CVE ========================
def predict_host_cve_relations(entity2id, rel2id, rotate_model, threshold=-15, top_k=10):
    print("\n🔮 Prédiction des relations host-CVE...")

    # Identifier les hôtes et CVE
    hosts = [e for e in entity2id if isinstance(e, str) and e.startswith("192.168.")]
    cves = [e for e in entity2id if isinstance(e, str) and e.startswith("CVE-")]

    if not hosts or not cves:
        print("⚠️ Aucun hôte ou CVE trouvé pour la prédiction")
        return

    rel_name = "at_risk_of"
    rel_idx = rel2id[rel_name]

    print(f"  Prédiction pour {len(hosts)} hôtes et {len(cves)} CVEs...")

    results = []

    for h in hosts:
        for c in cves:
            h_id = torch.tensor([entity2id[h]])
            t_id = torch.tensor([entity2id[c]])
            r_id = torch.tensor([rel_idx])

            with torch.no_grad():
                score = rotate_model(h_id, r_id, t_id).item()

            results.append((h, rel_name, c, score))

    # Trier les résultats par score décroissant
    results.sort(key=lambda x: x[3], reverse=True)

    # Afficher les meilleures prédictions
    print(f"\n🏆 Top {top_k} prédictions :")
    for h, r, t, s in results[:top_k]:
        print(f"  ({h})-[:{r}]->({t})  Score: {s:.4f}")

    # Injecter les relations avec score élevé
    print("\n💉 Injection des relations...")
    injected = 0

    for h, r, t, s in results:
        if s < threshold:
            continue

        node_h = matcher.match(name=h).first()
        node_t = matcher.match(name=t).first()

        if node_h and node_t:
            # Vérifier si la relation existe déjà
            existing = graph.match((node_h, node_t), r_type=r).first()
            if not existing:
                rel = Relationship(node_h, r, node_t)
                rel["prediction_score"] = float(s)
                rel["source"] = "RotatE"
                graph.create(rel)
                print(f"  ✅ Injecté: ({h})-[:{r}]->({t}) (score: {s:.4f})")
                injected += 1

    print(f"\n🎉 {injected} nouvelles relations injectées avec score > {threshold}")

# ======================== 9. R-GCN POUR CLASSIFICATION DES HÔTES ========================
class RGCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, num_rels):
        super().__init__()
        self.conv1 = RGCNConv(in_feat, hidden_feat, num_rels)
        self.conv2 = RGCNConv(hidden_feat, out_feat, num_rels)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_type))
        return self.conv2(x, data.edge_index, data.edge_type)

from sklearn.utils.class_weight import compute_class_weight

def train_rgcn(entity2id, rel2id, triplets_df):
    print("\n🧠 Entraînement du R-GCN pour classification des hôtes...")

    x = torch.randn(len(entity2id), 64)
    edge_index = torch.tensor([
        [entity2id[h] for h in triplets_df["head"]],
        [entity2id[t] for t in triplets_df["tail"]]
    ], dtype=torch.long)
    edge_type = torch.tensor([rel2id[r] for r in triplets_df["relation"]], dtype=torch.long)

    # Création des labels (hôtes vulnérables = connectés à au moins un CVE)
    host_entities = [e for e in entity2id if isinstance(e, str) and e.startswith("192.168.")]
    hosts_with_cve = set(
        triplets_df[triplets_df["tail"].str.startswith("CVE-")]["head"]
    ).union(
        triplets_df[triplets_df["head"].str.startswith("CVE-")]["tail"]
    )
    y = torch.tensor([1 if e in hosts_with_cve else 0 for e in entity2id], dtype=torch.long)

    train_mask = torch.zeros(len(entity2id), dtype=torch.bool)
    train_indices = torch.randperm(len(entity2id))[:int(0.7 * len(entity2id))]
    train_mask[train_indices] = True

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type,
                y=y, train_mask=train_mask, num_nodes=len(entity2id))

    # Poids de classe pour gérer le déséquilibre
    class_weights = compute_class_weight("balanced", classes=np.unique(y.numpy()), y=y.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    rgcn = RGCN(64, 32, 2, len(rel2id))
    optimizer = torch.optim.Adam(rgcn.parameters(), lr=0.01)

    for epoch in range(3):
        rgcn.train()
        optimizer.zero_grad()
        out = rgcn(data)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = (out.argmax(dim=1)[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            print(f"  Epoch {epoch} - Loss: {loss.item():.4f} - Acc: {acc:.2%}")

    return rgcn, data


def classify_hosts(rgcn, data, entity2id):
    print("\n🏷️ Classification des hôtes vulnérables...")

    rgcn.eval()
    with torch.no_grad():
        out = rgcn(data)
        probs = torch.softmax(out, dim=1)[:, 1]  # Probabilité d'être vulnérable

    # Marquer les hôtes vulnérables dans la base de données
    threshold = 0.7
    vulnerable_hosts = 0

    for name, idx in entity2id.items():
        if not name.startswith("192.168."):
            continue

        prob = probs[idx].item()
        if prob > threshold:
            node = matcher.match(name=name).first()
            if node and "Host" in node.labels:
                node["vulnerability_score"] = float(prob)
                node["is_vulnerable"] = True
                graph.push(node)
                print(f"  🔴 {name} - Vulnérable (score: {prob:.2f})")
                vulnerable_hosts += 1

    print(f"\n🔍 {vulnerable_hosts} hôtes identifiés comme vulnérables (score > {threshold})")




from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
import pandas as pd

# === Évaluation avancée du modèle RotatE ===
def evaluate_rotate_with_ranking(model, entity2id, rel2id, test_triplets, k_values=[1, 3, 10]):
    print("\n📊 Évaluation de RotatE avec MRR et Hits@K...")

    model.eval()
    mrr_total = 0.0
    hits_at_k = {k: 0 for k in k_values}
    num_triplets = len(test_triplets)

    entity_ids = torch.arange(len(entity2id))

    for h_name, r_name, t_name in tqdm(test_triplets, desc="Evaluating RotatE"):
        h_id = torch.tensor([entity2id[h_name]])
        r_id = torch.tensor([rel2id[r_name]])
        t_id = entity2id[t_name]

        # Corruption de la tête
        scores_head = model(entity_ids, r_id.repeat(len(entity_ids)), torch.tensor([t_id]*len(entity_ids)))
        _, indices_head = torch.sort(scores_head, descending=True)
        rank_head = (indices_head == h_id).nonzero(as_tuple=True)[0].item() + 1

        # Corruption de la queue
        scores_tail = model(h_id.repeat(len(entity_ids)), r_id.repeat(len(entity_ids)), entity_ids)
        _, indices_tail = torch.sort(scores_tail, descending=True)
        rank_tail = (indices_tail == t_id).nonzero(as_tuple=True)[0].item() + 1

        for rank in [rank_head, rank_tail]:
            mrr_total += 1.0 / rank
            for k in k_values:
                if rank <= k:
                    hits_at_k[k] += 1

    mrr = mrr_total / (2 * num_triplets)
    hits = {f"Hits@{k}": hits_at_k[k] / (2 * num_triplets) for k in k_values}

    print(f"🔹 MRR: {mrr:.4f}")
    for k in k_values:
        print(f"🔹 Hits@{k}: {hits[f'Hits@{k}']:.4f}")

    return {
        "MRR": mrr,
        **hits
    }

# === Évaluation avancée du modèle R-GCN ===
def evaluate_rgcn_advanced(rgcn, data, entity2id, label_names=["non_vulnérable", "vulnérable"]):
    print("\n📊 Évaluation du R-GCN avec F1 et matrice de confusion...")

    rgcn.eval()
    with torch.no_grad():
        out = rgcn(data)
        preds = out.argmax(dim=1)
        labels = data.y

    micro_f1 = f1_score(labels.numpy(), preds.numpy(), average='micro')
    macro_f1 = f1_score(labels.numpy(), preds.numpy(), average='macro')

    print(f"🔹 Micro-F1 : {micro_f1:.4f}")
    print(f"🔹 Macro-F1 : {macro_f1:.4f}")

    cm = confusion_matrix(labels.numpy(), preds.numpy())
    print("\n🧾 Matrice de confusion :")
    print(pd.DataFrame(cm, index=label_names, columns=label_names))

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "confusion_matrix": cm
    }


# ======================== 11. PIPELINE PRINCIPAL ========================
# ======================== 11. PIPELINE PRINCIPAL ========================
def main():
    align_and_fuse_graphs()
    triplets_df = extract_fused_triplets()
    entity2id, rel2id, h_idx, r_idx, t_idx = encode_entities(triplets_df)

    print("\n🚀 Entraînement du modèle RotatE...")
    rotate_model = RotatEModel(len(entity2id), len(rel2id), embedding_dim=128)
    optimizer = torch.optim.Adam(rotate_model.parameters(), lr=0.01)

    # Entraînement du modèle RotatE
    for epoch in range(100):
        rotate_model.train()
        optimizer.zero_grad()
        loss = -torch.mean(rotate_model(h_idx, r_idx, t_idx))
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch} - Loss: {loss.item():.4f}")

    # Prédiction des relations entre hôtes et CVE
    predict_host_cve_relations(entity2id, rel2id, rotate_model, threshold=-0.4, top_k=20)

    # Entraînement du modèle R-GCN
    rgcn, data = train_rgcn(entity2id, rel2id, triplets_df)
    classify_hosts(rgcn, data, entity2id)

    # ✅ Évaluation avec un échantillon de 10% des triplets
    test_df = triplets_df.sample(frac=0.01, random_state=42)  # Prendre un échantillon de 10%
    test_triplets = list(zip(test_df["head"], test_df["relation"], test_df["tail"]))

    # Évaluation du modèle RotatE
    print("\n🚀 Évaluation du modèle RotatE sur un échantillon de 10% des triplets...")
    evaluate_rotate_with_ranking(rotate_model, entity2id, rel2id, test_triplets)

    # Évaluation du modèle R-GCN
    print("\n🚀 Évaluation du modèle R-GCN sur un échantillon de 10% des triplets...")
    evaluate_rgcn_advanced(rgcn, data, entity2id)

    print("\n🎉 Pipeline terminé avec succès!")

if __name__ == "__main__":
    main()
