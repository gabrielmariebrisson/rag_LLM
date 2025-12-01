"""Script d'évaluation Recall@5 : Comparaison avec/sans reranker."""
import asyncio
import os
import sys
import json
import pandas as pd
import time
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import Settings
from app.services.embeddings import EmbeddingService
from app.services.reranker import RerankerService
from app.vectorstores.qdrant_store import QdrantVectorStore


def calculate_recall_at_k(retrieved_contexts: List[str], ground_truth_context: str, k: int = 5) -> bool:
    """
    Calcule Recall@k : est-ce que le contexte ground truth est dans les k premiers résultats ?
    
    Args:
        retrieved_contexts: Liste des contextes récupérés
        ground_truth_context: Contexte attendu (ground truth)
        k: Nombre de résultats à considérer
        
    Returns:
        True si le ground truth est dans les k premiers, False sinon
    """
    # Normaliser les contextes (lowercase, strip)
    ground_truth_normalized = ground_truth_context.lower().strip()
    retrieved_normalized = [ctx.lower().strip() for ctx in retrieved_contexts[:k]]
    
    # Vérifier si le ground truth est dans les résultats
    return ground_truth_normalized in retrieved_normalized


async def evaluate_retrieval(
    csv_path: str,
    sample_size: int = 50,
    random_state: int = 42,
    k: int = 5
) -> Dict[str, Any]:
    """
    Évalue la performance de retrieval avec et sans reranker.
    
    Args:
        csv_path: Chemin vers le CSV SQuAD
        sample_size: Taille de l'échantillon de questions
        random_state: Seed pour la reproductibilité
        k: Nombre de documents à évaluer (Recall@k)
        
    Returns:
        Dictionnaire avec les métriques d'évaluation
    """
    # Charger la configuration
    load_dotenv()
    config = Settings()
    
    # Initialiser les services
    print("🔄 Initialisation des services...")
    embedding_service = EmbeddingService(config)
    reranker_service = RerankerService(config)
    vectorstore = QdrantVectorStore(config, embedding_service)
    await vectorstore.connect()
    
    # Charger le CSV
    print(f"📖 Chargement du CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df)} lignes chargées")
    
    # Échantillonner les questions
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    print(f"📊 Évaluation sur {len(sample_df)} questions")
    
    # Métriques
    recall_without_reranker = []
    recall_with_reranker = []
    times_without_reranker = []
    times_with_reranker = []
    details = []
    
    print("\n🔍 Évaluation en cours...")
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Questions"):
        question = row['question']
        ground_truth_context = row['context']
        
        # Test SANS reranker
        start_time = time.perf_counter()
        results_without = await vectorstore.hybrid_search(question, top_k=k)
        time_without = time.perf_counter() - start_time
        
        retrieved_contexts_without = [r['page_content'] for r in results_without]
        recall_without = calculate_recall_at_k(retrieved_contexts_without, ground_truth_context, k)
        recall_without_reranker.append(recall_without)
        times_without_reranker.append(time_without)
        
        # Test AVEC reranker
        start_time = time.perf_counter()
        results_with = await vectorstore.hybrid_search(question, top_k=config.RERANKER_TOP_K)
        
        # Reranker les résultats
        document_texts = [r['page_content'] for r in results_with]
        reranked = await reranker_service.rerank(question, document_texts, top_k=k)
        time_with = time.perf_counter() - start_time
        
        retrieved_contexts_with = [doc_text for doc_text, _ in reranked]
        recall_with = calculate_recall_at_k(retrieved_contexts_with, ground_truth_context, k)
        recall_with_reranker.append(recall_with)
        times_with_reranker.append(time_with)
        
        # Détails pour cette question
        details.append({
            "question": question,
            "ground_truth_context": ground_truth_context[:200] + "..." if len(ground_truth_context) > 200 else ground_truth_context,
            "recall_without_reranker": recall_without,
            "recall_with_reranker": recall_with,
            "time_without_reranker_ms": round(time_without * 1000, 2),
            "time_with_reranker_ms": round(time_with * 1000, 2)
        })
    
    # Calculer les métriques globales
    recall_without_avg = sum(recall_without_reranker) / len(recall_without_reranker) if recall_without_reranker else 0
    recall_with_avg = sum(recall_with_reranker) / len(recall_with_reranker) if recall_with_reranker else 0
    improvement = ((recall_with_avg - recall_without_avg) / recall_without_avg * 100) if recall_without_avg > 0 else 0
    
    avg_time_without = sum(times_without_reranker) / len(times_without_reranker) if times_without_reranker else 0
    avg_time_with = sum(times_with_reranker) / len(times_with_reranker) if times_with_reranker else 0
    
    results = {
        "sample_size": len(sample_df),
        "k": k,
        "recall_at_k": {
            "without_reranker": round(recall_without_avg, 4),
            "with_reranker": round(recall_with_avg, 4),
            "improvement_percent": round(improvement, 2)
        },
        "latency_ms": {
            "without_reranker_avg": round(avg_time_without * 1000, 2),
            "with_reranker_avg": round(avg_time_with * 1000, 2),
            "reranker_overhead_ms": round((avg_time_with - avg_time_without) * 1000, 2)
        },
        "details": details
    }
    
    await vectorstore.disconnect()
    return results


def print_results(results: Dict[str, Any]):
    """Affiche les résultats de l'évaluation de manière lisible."""
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DE L'ÉVALUATION")
    print("=" * 60)
    
    print(f"\n📈 Sample Size: {results['sample_size']} questions")
    print(f"🎯 Recall@{results['k']}:")
    print(f"   - Sans Reranker: {results['recall_at_k']['without_reranker']:.4f} ({results['recall_at_k']['without_reranker']*100:.2f}%)")
    print(f"   - Avec Reranker: {results['recall_at_k']['with_reranker']:.4f} ({results['recall_at_k']['with_reranker']*100:.2f}%)")
    print(f"   - Amélioration: +{results['recall_at_k']['improvement_percent']:.2f}%")
    
    print(f"\n⏱️  Latence moyenne:")
    print(f"   - Sans Reranker: {results['latency_ms']['without_reranker_avg']:.2f} ms")
    print(f"   - Avec Reranker: {results['latency_ms']['with_reranker_avg']:.2f} ms")
    print(f"   - Overhead Reranker: +{results['latency_ms']['reranker_overhead_ms']:.2f} ms")
    
    print("\n" + "=" * 60)


def save_results(results: Dict[str, Any], output_dir: str = "evaluation_results"):
    """Sauvegarde les résultats en JSON et CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder JSON complet
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Résultats JSON sauvegardés: {json_path}")
    
    # Sauvegarder CSV avec détails
    csv_path = os.path.join(output_dir, "evaluation_details.csv")
    df_details = pd.DataFrame(results['details'])
    df_details.to_csv(csv_path, index=False)
    print(f"💾 Détails CSV sauvegardés: {csv_path}")
    
    # Sauvegarder résumé
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    summary = {
        "metric": [
            "Recall@5 (Sans Reranker)",
            "Recall@5 (Avec Reranker)",
            "Amélioration (%)",
            "Latence moyenne sans reranker (ms)",
            "Latence moyenne avec reranker (ms)",
            "Overhead reranker (ms)"
        ],
        "value": [
            results['recall_at_k']['without_reranker'],
            results['recall_at_k']['with_reranker'],
            results['recall_at_k']['improvement_percent'],
            results['latency_ms']['without_reranker_avg'],
            results['latency_ms']['with_reranker_avg'],
            results['latency_ms']['reranker_overhead_ms']
        ]
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(summary_path, index=False)
    print(f"💾 Résumé CSV sauvegardé: {summary_path}")


if __name__ == "__main__":
    csv_path = "squad_2.0/train.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Fichier CSV introuvable: {csv_path}")
        print("💡 Assurez-vous que le fichier existe dans le répertoire racine du projet")
        sys.exit(1)
    
    print("=" * 60)
    print("Évaluation Recall@5: Sans Reranker vs Avec Reranker")
    print("=" * 60)
    
    # Exécuter l'évaluation
    results = asyncio.run(evaluate_retrieval(
        csv_path=csv_path,
        sample_size=50,
        random_state=42,
        k=5
    ))
    
    # Afficher les résultats
    print_results(results)
    
    # Sauvegarder les résultats
    save_results(results)
    
    print("\n✅ Évaluation terminée!")

