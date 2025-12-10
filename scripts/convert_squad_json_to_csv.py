"""Script pour convertir SQuAD 2.0 JSON en CSV ou utiliser les CSV existants."""
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


def convert_existing_csv_to_format(csv_input_path: str, csv_output_path: str):
    """
    Convertit les fichiers CSV existants (train-squad.csv, validation-squad.csv) 
    au format attendu par le système.
    
    Args:
        csv_input_path: Chemin vers le CSV existant
        csv_output_path: Chemin de sortie pour le CSV formaté
    """
    print(f"📖 Lecture du fichier CSV existant: {csv_input_path}")
    df = pd.read_csv(csv_input_path)
    
    print(f"✅ Fichier CSV chargé: {len(df)} lignes")
    print(f"📊 Colonnes: {list(df.columns)}")
    
    # Vérifier les colonnes disponibles
    required_cols = ['context', 'question', 'id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Colonnes manquantes: {missing_cols}")
        return None
    
    # Créer le DataFrame au format attendu
    # Les CSV existants ont: context, question, id, answer_start, text
    # Le format attendu est: id, title, question, context, answers
    
    result_df = pd.DataFrame({
        'id': df['id'],
        'title': '',  # Pas de titre dans les CSV existants
        'question': df['question'],
        'context': df['context'],
        'answers': df.get('text', '').apply(lambda x: f"['{x}']" if pd.notna(x) and x else '')
    })
    
    # Sauvegarder en CSV
    print(f"💾 Sauvegarde du CSV formaté: {csv_output_path}")
    result_df.to_csv(csv_output_path, index=False, encoding='utf-8')
    
    print(f"✅ Conversion terminée: {len(result_df)} lignes créées")
    print(f"📊 Aperçu:")
    print(result_df.head())
    
    return result_df


if __name__ == "__main__":
    squad_dir = DATA_RAW / "squad_2.0"
    json_path = str(squad_dir / "train-v2.0.json")
    csv_path = str(squad_dir / "train.csv")
    
    # Vérifier d'abord si les CSV existants sont disponibles
    existing_csv_train = str(squad_dir / "train-squad.csv")
    existing_csv_val = str(squad_dir / "validation-squad.csv")
    
    if os.path.exists(existing_csv_train):
        print("=" * 60)
        print("Conversion SQuAD 2.0 CSV existant -> Format attendu")
        print("=" * 60)
        convert_existing_csv_to_format(existing_csv_train, csv_path)
        
        # Convertir aussi le fichier de validation si disponible
        if os.path.exists(existing_csv_val):
            val_csv_path = str(squad_dir / "validation.csv")
            print("\n" + "=" * 60)
            print("Conversion du fichier de validation")
            print("=" * 60)
            convert_existing_csv_to_format(existing_csv_val, val_csv_path)
    elif os.path.exists(json_path):
        print("=" * 60)
        print("Conversion SQuAD 2.0 JSON -> CSV")
        print("=" * 60)
        convert_squad_json_to_csv(json_path, csv_path)
    else:
        print(f"❌ Aucun fichier source trouvé!")
        print(f"💡 Options disponibles:")
        print(f"   1. Télécharger les CSV depuis Kaggle et les placer dans: {squad_dir}")
        print(f"      - train-squad.csv")
        print(f"      - validation-squad.csv")
        print(f"   2. Télécharger le JSON depuis https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")
        print(f"      et le placer dans: {squad_dir}")
        sys.exit(1)

