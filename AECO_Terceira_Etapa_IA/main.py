"""AECO - Aula 10: preparação dos dados Iris para a rede neural.

Etapa 1 de 3. Execute: python main.py
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def preparar_dados(semente=42):
    """Separa os dados e padroniza usando apenas estatísticas do treinamento."""
    iris = load_iris()
    entradas = np.asarray(iris.data, dtype=np.float64)
    classes = np.asarray(iris.target, dtype=np.int64)

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        entradas, classes, test_size=0.20, random_state=semente, stratify=classes
    )

    media = np.mean(x_treino, axis=0)
    desvio = np.std(x_treino, axis=0)
    desvio = np.where(desvio == 0.0, 1.0, desvio)

    x_treino = (x_treino - media) / desvio
    x_teste = (x_teste - media) / desvio
    y_treino_one_hot = np.eye(3, dtype=np.float64)[y_treino]
    y_teste_one_hot = np.eye(3, dtype=np.float64)[y_teste]

    return {
        "x_treino": x_treino,
        "x_teste": x_teste,
        "y_treino": y_treino,
        "y_teste": y_teste,
        "y_treino_one_hot": y_treino_one_hot,
        "y_teste_one_hot": y_teste_one_hot,
        "nomes_classes": tuple(iris.target_names),
    }


def main():
    dados = preparar_dados()
    print("AECO - Etapa 1/3: pré-processamento Iris")
    print(f"Treinamento: {dados['x_treino'].shape[0]} amostras")
    print(f"Teste: {dados['x_teste'].shape[0]} amostras")
    print(f"Características: {dados['x_treino'].shape[1]}")
    print(f"Classes: {len(dados['nomes_classes'])}")
    print(f"One-hot (treinamento): {dados['y_treino_one_hot'].shape}")
    print(f"Média do treino após padronização: {np.round(dados['x_treino'].mean(axis=0), 6)}")
    print(f"Desvio-padrão do treino após padronização: {np.round(dados['x_treino'].std(axis=0), 6)}")
    print("Status: pré-processamento concluído; rede neural nas próximas etapas.")


if __name__ == "__main__":
    main()
