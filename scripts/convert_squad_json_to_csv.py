"""Script pour convertir SQuAD 2.0 JSON en CSV."""
import json
import pandas as pd
import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.paths import DATA_RAW

def convert_squad_json_to_csv(json_path: str, csv_path: str):
    """
    Convertit le fichier JSON SQuAD 2.0 en CSV.
    
    Args:
        json_path: Chemin vers le fichier JSON SQuAD
        csv_path: Chemin de sortie pour le CSV
    """
    print(f"📖 Lecture du fichier JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Version SQuAD: {data.get('version', 'unknown')}")
    
    # Extraire les données
    rows = []
    doc_id = 0
    
    for article in data['data']:
        title = article.get('title', '')
        
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            
            for qa in paragraph['qas']:
                question = qa['question']
                qa_id = qa['id']
                
                # SQuAD 2.0 peut avoir des questions sans réponse
                if qa.get('is_impossible', False):
                    answers = []
                else:
                    answers = [ans['text'] for ans in qa.get('answers', [])]
                
                rows.append({
                    'id': qa_id,
                    'title': title,
                    'question': question,
                    'context': context,
                    'answers': str(answers) if answers else ''
                })
                doc_id += 1
    
    # Créer le DataFrame
    df = pd.DataFrame(rows)
    
    # Sauvegarder en CSV
    print(f"💾 Sauvegarde du CSV: {csv_path}")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"✅ Conversion terminée: {len(df)} lignes créées")
    print(f"📊 Aperçu:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    json_path = str(DATA_RAW / "squad_2.0" / "train-v2.0.json")
    csv_path = str(DATA_RAW / "squad_2.0" / "train.csv")
    
    if not os.path.exists(json_path):
        print(f"❌ Fichier JSON introuvable: {json_path}")
        print("💡 Téléchargez d'abord le fichier depuis https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")
        print(f"💡 Placez-le dans: {DATA_RAW / 'squad_2.0'}")
        sys.exit(1)
    
    print("=" * 60)
    print("Conversion SQuAD 2.0 JSON -> CSV")
    print("=" * 60)
    
    convert_squad_json_to_csv(json_path, csv_path)

