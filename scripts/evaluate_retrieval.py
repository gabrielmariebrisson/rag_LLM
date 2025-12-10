"""Script d'évaluation Recall@5, Recall@10 et TTFT : Comparaison avec/sans reranker."""
import asyncio
import os
import sys
import json
import pandas as pd
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import Settings
from app.core.paths import SQUAD_CSV, EVALUATION_RESULTS
from app.core.prompts import SYSTEM_PROMPT, format_user_prompt
from app.services.embeddings import EmbeddingService
from app.services.reranker import RerankerService
from app.services.llm_client import LLMClient
from app.vectorstores.qdrant_store import QdrantVectorStore

_executor = ThreadPoolExecutor(max_workers=4)


def calculate_recall_at_k(retrieved_results: List[Dict[str, Any]], ground_truth_context: str, k: int = 5) -> bool:
    """
    Calcule Recall@k : est-ce que le contexte ground truth est dans les k premiers résultats ?
    
    Args:
        retrieved_results: Liste des résultats récupérés (dicts avec 'page_content' et 'metadata')
        ground_truth_context: Contexte attendu (ground truth)
        k: Nombre de résultats à considérer
        
    Returns:
        True si le ground truth est dans les k premiers, False sinon
    """
    # Normaliser le ground truth
    ground_truth_normalized = ground_truth_context.lower().strip()
    
    # Extraire les contextes depuis les métadonnées ou page_content
    retrieved_contexts_normalized = []
    for result in retrieved_results[:k]:
        # Priorité 1: Utiliser metadata['context'] si disponible (format exact du CSV)
        if result.get('metadata') and 'context' in result['metadata']:
            ctx = str(result['metadata']['context']).lower().strip()
            retrieved_contexts_normalized.append(ctx)
        # Priorité 2: Extraire depuis page_content (format "Title: X\nContext: Y\nAnswer: Z")
        elif result.get('page_content'):
            page_content = result['page_content'].lower()
            # Essayer d'extraire le contexte depuis le format "context: ..."
            if 'context:' in page_content:
                parts = page_content.split('context:')
                if len(parts) > 1:
                    context_part = parts[1].split('answer:')[0].strip() if 'answer:' in parts[1] else parts[1].strip()
                    retrieved_contexts_normalized.append(context_part)
            else:
                # Si pas de format structuré, utiliser tout le page_content
                retrieved_contexts_normalized.append(page_content.strip())
    
    # Vérifier si le ground truth est dans les résultats (comparaison exacte)
    if ground_truth_normalized in retrieved_contexts_normalized:
        return True
    
    # Comparaison partielle : vérifier si une portion significative du ground truth est présente
    # (au moins 80% du ground truth doit être trouvé dans un des résultats)
    if ground_truth_normalized:
        gt_words = set(ground_truth_normalized.split())
        if len(gt_words) > 0:
            for retrieved_ctx in retrieved_contexts_normalized:
                if retrieved_ctx:
                    retrieved_words = set(retrieved_ctx.split())
                    # Calculer l'overlap (mots en commun)
                    overlap = len(gt_words.intersection(retrieved_words))
                    overlap_ratio = overlap / len(gt_words)
                    # Si au moins 80% des mots du ground truth sont présents
                    if overlap_ratio >= 0.8:
                        return True
    
    return False


async def measure_ttft(
    query: str,
    context: str,
    config: Settings,
    llm_client: Optional[LLMClient] = None
) -> Dict[str, float]:
    """
    Mesure le TTFT (Time To First Token) et la latence totale de génération LLM.
    
    Args:
        query: Question de l'utilisateur
        context: Contexte récupéré depuis la base vectorielle
        config: Configuration de l'application
        llm_client: Client LLM (créé automatiquement si None)
        
    Returns:
        Dictionnaire avec 'ttft_ms' et 'total_latency_ms'
    """
    if llm_client is None:
        llm_client = LLMClient(config)
    
    # Formater le prompt
    user_prompt = format_user_prompt(context, query)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    model_name = config.LLM_MODEL_NAME
    
    loop = asyncio.get_event_loop()
    
    def measure_with_streaming():
        """Mesure le TTFT avec streaming."""
        ttft_start = None
        total_start = time.perf_counter()
        
        try:
            # Essayer le streaming avec le client LLM
            stream_generator = llm_client.generate(
                messages=messages,
                model=model_name,
                stream=True
            )
            
            # Si c'est un générateur (streaming activé)
            if hasattr(stream_generator, '__iter__') and not isinstance(stream_generator, str):
                first_token_received = False
                full_response = ""
                
                for chunk_content in stream_generator:
                    if not first_token_received:
                        ttft_start = time.perf_counter()
                        first_token_received = True
                    if chunk_content:
                        full_response += chunk_content
                
                total_end = time.perf_counter()
                ttft_ms = (ttft_start - total_start) * 1000 if ttft_start else None
                total_ms = (total_end - total_start) * 1000
                
                return {
                    "ttft_ms": round(ttft_ms, 2) if ttft_ms else None,
                    "total_latency_ms": round(total_ms, 2),
                    "response_length": len(full_response),
                    "estimated": False  # Mesure réelle avec streaming
                }
        except (NotImplementedError, AttributeError, TypeError) as e:
            # Streaming non supporté ou erreur, utiliser fallback
            pass
        except Exception as e:
            # Autre erreur, utiliser fallback
            pass
        
        # Fallback: appel normal (approximation du TTFT)
        total_start = time.perf_counter()
        response = llm_client.generate(messages=messages, model=model_name, stream=False)
        total_end = time.perf_counter()
        total_ms = (total_end - total_start) * 1000
        
        # Approximation: TTFT ≈ 30% de la latence totale (estimation conservatrice)
        # Note: Cette estimation peut être inexacte, le streaming est préférable
        estimated_ttft_ms = total_ms * 0.3
        
        return {
            "ttft_ms": round(estimated_ttft_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "response_length": len(response) if response else 0,
            "estimated": True  # Indicateur que c'est une estimation
        }
    
    return await loop.run_in_executor(_executor, measure_with_streaming)


async def evaluate_retrieval(
    csv_path: str,
    sample_size: int = 50,
    random_state: int = 42,
    k: int = 5,
    enable_ttft_measurement: bool = True
) -> Dict[str, Any]:
    """
    Évalue la performance de retrieval avec et sans reranker.
    
    Args:
        csv_path: Chemin vers le CSV SQuAD
        sample_size: Taille de l'échantillon de questions
        random_state: Seed pour la reproductibilité
        k: Nombre de documents à évaluer (Recall@k)
        enable_ttft_measurement: Si True, mesure également le TTFT
        
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
    
    # Initialiser le client LLM si on mesure le TTFT
    llm_client = None
    if enable_ttft_measurement:
        try:
            llm_client = LLMClient(config)
            print("✅ Client LLM initialisé pour mesure TTFT")
        except Exception as e:
            print(f"⚠️  Impossible d'initialiser le client LLM: {e}")
            print("   La mesure du TTFT sera ignorée")
            enable_ttft_measurement = False
    
    # Charger le CSV
    print(f"📖 Chargement du CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df)} lignes chargées")
    
    # Échantillonner les questions
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    print(f"📊 Évaluation sur {len(sample_df)} questions")
    
    # Métriques
    recall_5_without_reranker = []
    recall_5_with_reranker = []
    recall_10_without_reranker = []
    recall_10_with_reranker = []
    times_without_reranker = []
    times_with_reranker = []
    ttft_measurements = []
    details = []
    
    print("\n🔍 Évaluation en cours...")
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Questions"):
        question = row['question']
        ground_truth_context = row['context']
        
        # Test SANS reranker - Recherche avec top_k=10 pour évaluer Recall@10
        start_time = time.perf_counter()
        results_without = await vectorstore.hybrid_search(question, top_k=10)
        time_without = time.perf_counter() - start_time
        
        # Calculer Recall@5 et Recall@10
        recall_5_without = calculate_recall_at_k(results_without, ground_truth_context, k=5)
        recall_10_without = calculate_recall_at_k(results_without, ground_truth_context, k=10)
        recall_5_without_reranker.append(recall_5_without)
        recall_10_without_reranker.append(recall_10_without)
        times_without_reranker.append(time_without)
        
        # Test AVEC reranker
        start_time = time.perf_counter()
        results_with = await vectorstore.hybrid_search(question, top_k=config.RERANKER_TOP_K)
        
        # Reranker les résultats (utiliser page_content pour le reranker)
        document_texts = [r['page_content'] for r in results_with]
        reranked = await reranker_service.rerank(question, document_texts, top_k=10)  # Prendre top 10 pour évaluer Recall@10
        time_with = time.perf_counter() - start_time
        
        # Reconstruire les résultats rerankés avec métadonnées pour la comparaison
        text_to_result = {r['page_content']: r for r in results_with}
        reranked_results_with_metadata = []
        for doc_text, score in reranked:
            if doc_text in text_to_result:
                reranked_results_with_metadata.append(text_to_result[doc_text])
        
        # Calculer Recall@5 et Recall@10 avec reranker
        recall_5_with = calculate_recall_at_k(reranked_results_with_metadata, ground_truth_context, k=5)
        recall_10_with = calculate_recall_at_k(reranked_results_with_metadata, ground_truth_context, k=10)
        recall_5_with_reranker.append(recall_5_with)
        recall_10_with_reranker.append(recall_10_with)
        times_with_reranker.append(time_with)
        
        # Mesurer TTFT si activé (seulement sur un échantillon pour éviter de ralentir l'évaluation)
        ttft_info = None
        if enable_ttft_measurement and llm_client and idx % 5 == 0:  # Mesurer TTFT toutes les 5 questions
            try:
                # Construire le contexte à partir des résultats
                context = "\n\n".join([r['page_content'] for r in results_without[:3]])
                ttft_result = await measure_ttft(question, context, config, llm_client)
                ttft_measurements.append(ttft_result)
                ttft_info = ttft_result
            except Exception as e:
                tqdm.write(f"⚠️  Erreur mesure TTFT: {e}")
        
        # Détails pour cette question
        top_results_without = [r.get('metadata', {}).get('context', r.get('page_content', ''))[:100] for r in results_without[:3]]
        top_results_with = [r.get('metadata', {}).get('context', r.get('page_content', ''))[:100] for r in reranked_results_with_metadata[:3]]
        
        detail_entry = {
            "question": question,
            "ground_truth_context": ground_truth_context[:200] + "..." if len(ground_truth_context) > 200 else ground_truth_context,
            "recall_5_without_reranker": recall_5_without,
            "recall_5_with_reranker": recall_5_with,
            "recall_10_without_reranker": recall_10_without,
            "recall_10_with_reranker": recall_10_with,
            "time_without_reranker_ms": round(time_without * 1000, 2),
            "time_with_reranker_ms": round(time_with * 1000, 2),
            "top_3_results_without": "; ".join(top_results_without),
            "top_3_results_with": "; ".join(top_results_with)
        }
        
        if ttft_info:
            detail_entry["ttft_ms"] = ttft_info.get("ttft_ms")
            detail_entry["total_latency_ms"] = ttft_info.get("total_latency_ms")
            if ttft_info.get("estimated"):
                detail_entry["ttft_estimated"] = True
        
        details.append(detail_entry)
    
    # Calculer les métriques globales
    recall_5_without_avg = sum(recall_5_without_reranker) / len(recall_5_without_reranker) if recall_5_without_reranker else 0
    recall_5_with_avg = sum(recall_5_with_reranker) / len(recall_5_with_reranker) if recall_5_with_reranker else 0
    improvement_5 = ((recall_5_with_avg - recall_5_without_avg) / recall_5_without_avg * 100) if recall_5_without_avg > 0 else 0
    
    recall_10_without_avg = sum(recall_10_without_reranker) / len(recall_10_without_reranker) if recall_10_without_reranker else 0
    recall_10_with_avg = sum(recall_10_with_reranker) / len(recall_10_with_reranker) if recall_10_with_reranker else 0
    improvement_10 = ((recall_10_with_avg - recall_10_without_avg) / recall_10_without_avg * 100) if recall_10_without_avg > 0 else 0
    
    avg_time_without = sum(times_without_reranker) / len(times_without_reranker) if times_without_reranker else 0
    avg_time_with = sum(times_with_reranker) / len(times_with_reranker) if times_with_reranker else 0
    
    # Calculer statistiques TTFT
    ttft_stats = None
    if ttft_measurements:
        valid_ttfts = [m["ttft_ms"] for m in ttft_measurements if m.get("ttft_ms") is not None]
        if valid_ttfts:
            ttft_stats = {
                "avg_ttft_ms": round(sum(valid_ttfts) / len(valid_ttfts), 2),
                "min_ttft_ms": round(min(valid_ttfts), 2),
                "max_ttft_ms": round(max(valid_ttfts), 2),
                "p50_ttft_ms": round(sorted(valid_ttfts)[len(valid_ttfts) // 2], 2),
                "p95_ttft_ms": round(sorted(valid_ttfts)[int(len(valid_ttfts) * 0.95)], 2) if len(valid_ttfts) > 1 else round(valid_ttfts[0], 2),
                "sample_size": len(valid_ttfts),
                "estimated_count": sum(1 for m in ttft_measurements if m.get("estimated", False))
            }
    
    results = {
        "sample_size": len(sample_df),
        "recall_at_5": {
            "without_reranker": round(recall_5_without_avg, 4),
            "with_reranker": round(recall_5_with_avg, 4),
            "improvement_percent": round(improvement_5, 2)
        },
        "recall_at_10": {
            "without_reranker": round(recall_10_without_avg, 4),
            "with_reranker": round(recall_10_with_avg, 4),
            "improvement_percent": round(improvement_10, 2)
        },
        "latency_ms": {
            "without_reranker_avg": round(avg_time_without * 1000, 2),
            "with_reranker_avg": round(avg_time_with * 1000, 2),
            "reranker_overhead_ms": round((avg_time_with - avg_time_without) * 1000, 2)
        },
        "ttft_stats": ttft_stats,
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
    
    print(f"\n🎯 Recall@5:")
    print(f"   - Sans Reranker: {results['recall_at_5']['without_reranker']:.4f} ({results['recall_at_5']['without_reranker']*100:.2f}%)")
    print(f"   - Avec Reranker: {results['recall_at_5']['with_reranker']:.4f} ({results['recall_at_5']['with_reranker']*100:.2f}%)")
    print(f"   - Amélioration: +{results['recall_at_5']['improvement_percent']:.2f}%")
    
    print(f"\n🎯 Recall@10:")
    print(f"   - Sans Reranker: {results['recall_at_10']['without_reranker']:.4f} ({results['recall_at_10']['without_reranker']*100:.2f}%)")
    print(f"   - Avec Reranker: {results['recall_at_10']['with_reranker']:.4f} ({results['recall_at_10']['with_reranker']*100:.2f}%)")
    print(f"   - Amélioration: +{results['recall_at_10']['improvement_percent']:.2f}%")
    
    print(f"\n⏱️  Latence moyenne de retrieval:")
    print(f"   - Sans Reranker: {results['latency_ms']['without_reranker_avg']:.2f} ms")
    print(f"   - Avec Reranker: {results['latency_ms']['with_reranker_avg']:.2f} ms")
    print(f"   - Overhead Reranker: +{results['latency_ms']['reranker_overhead_ms']:.2f} ms")
    
    if results.get('ttft_stats'):
        ttft = results['ttft_stats']
        print(f"\n⚡ TTFT (Time To First Token) - LLM:")
        print(f"   - Moyenne: {ttft['avg_ttft_ms']:.2f} ms")
        print(f"   - P50 (médiane): {ttft['p50_ttft_ms']:.2f} ms")
        print(f"   - P95: {ttft['p95_ttft_ms']:.2f} ms")
        print(f"   - Min: {ttft['min_ttft_ms']:.2f} ms")
        print(f"   - Max: {ttft['max_ttft_ms']:.2f} ms")
        print(f"   - Échantillon: {ttft['sample_size']} mesures")
        if ttft.get('estimated_count', 0) > 0:
            print(f"   ⚠️  {ttft['estimated_count']} mesures sont des estimations (streaming non disponible)")
    else:
        print(f"\n⚡ TTFT: Non mesuré (LLM non configuré ou erreur)")
    
    print("\n" + "=" * 60)


def save_results(results: Dict[str, Any], output_dir: Optional[str] = None):
    """Sauvegarde les résultats en JSON et CSV."""
    if output_dir is None:
        output_dir = str(EVALUATION_RESULTS)
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
    summary_data = {
        "metric": [
            "Recall@5 (Sans Reranker)",
            "Recall@5 (Avec Reranker)",
            "Amélioration Recall@5 (%)",
            "Recall@10 (Sans Reranker)",
            "Recall@10 (Avec Reranker)",
            "Amélioration Recall@10 (%)",
            "Latence moyenne sans reranker (ms)",
            "Latence moyenne avec reranker (ms)",
            "Overhead reranker (ms)"
        ],
        "value": [
            results['recall_at_5']['without_reranker'],
            results['recall_at_5']['with_reranker'],
            results['recall_at_5']['improvement_percent'],
            results['recall_at_10']['without_reranker'],
            results['recall_at_10']['with_reranker'],
            results['recall_at_10']['improvement_percent'],
            results['latency_ms']['without_reranker_avg'],
            results['latency_ms']['with_reranker_avg'],
            results['latency_ms']['reranker_overhead_ms']
        ]
    }
    
    # Ajouter les métriques TTFT si disponibles
    if results.get('ttft_stats'):
        ttft = results['ttft_stats']
        summary_data["metric"].extend([
            "TTFT Moyenne (ms)",
            "TTFT P50 (ms)",
            "TTFT P95 (ms)",
            "TTFT Min (ms)",
            "TTFT Max (ms)"
        ])
        summary_data["value"].extend([
            ttft['avg_ttft_ms'],
            ttft['p50_ttft_ms'],
            ttft['p95_ttft_ms'],
            ttft['min_ttft_ms'],
            ttft['max_ttft_ms']
        ])
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_path, index=False)
    print(f"💾 Résumé CSV sauvegardé: {summary_path}")


if __name__ == "__main__":
    csv_path = str(SQUAD_CSV)
    
    if not os.path.exists(csv_path):
        print(f"❌ Fichier CSV introuvable: {csv_path}")
        print("💡 Assurez-vous que le fichier existe dans data/raw/squad_2.0/")
        sys.exit(1)
    
    print("=" * 60)
    print("Évaluation Recall@5, Recall@10 et TTFT")
    print("Comparaison: Sans Reranker vs Avec Reranker")
    print("=" * 60)
    
    # Exécuter l'évaluation
    results = asyncio.run(evaluate_retrieval(
        csv_path=csv_path,
        sample_size=50,
        random_state=42,
        k=5,  # Utilisé pour la recherche initiale, mais on évalue aussi Recall@10
        enable_ttft_measurement=True
    ))
    
    # Afficher les résultats
    print_results(results)
    
    # Sauvegarder les résultats
    save_results(results)
    
    print("\n✅ Évaluation terminée!")

