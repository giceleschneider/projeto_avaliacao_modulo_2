"""Projeto de inspeção de peças fundidas com OpenCV e TensorFlow.

Para executar:
python main.py --dataset "/caminho/para/dados"
"""

import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# Valores pequenos para facilitar o estudo e reduzir o tempo de treinamento.
TAMANHO_IMAGEM = (128, 128)
TAMANHO_LOTE = 32
EPOCAS = 10
SEMENTE = 42


def criar_pastas_saida(pasta_saida):
    """Cria a pasta onde os gráficos e o modelo serão guardados."""
    os.makedirs(pasta_saida, exist_ok=True)


def mostrar_processamento_classico(pasta_dataset, pasta_saida):
    """Aplica OpenCV em uma imagem OK e uma imagem defeituosa."""
    classes = ["def_front", "ok_front"]
    imagens = []

    for classe in classes:
        pasta_classe = os.path.join(pasta_dataset, classe)
        arquivos = [nome for nome in os.listdir(pasta_classe)
                    if nome.lower().endswith((".jpg", ".jpeg", ".png"))]
        arquivo_escolhido = random.choice(arquivos)
        caminho = os.path.join(pasta_classe, arquivo_escolhido)
        imagem = cv2.imread(caminho)
        imagens.append((classe, imagem))

    figura, eixos = plt.subplots(2, 6, figsize=(18, 6))
    titulos = ["Original", "Cinza", "Gaussian Blur", "Limiar", "Canny", "Morfologia"]

    for linha, (classe, imagem) in enumerate(imagens):
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(cinza, (5, 5), 0)
        _, limiar = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY_INV)
        bordas = cv2.Canny(blur, 40, 120)
        kernel = np.ones((3, 3), np.uint8)
        morfologia = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, kernel, iterations=2)

        etapas = [cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB), cinza, blur,
                  limiar, bordas, morfologia]
        for coluna, etapa in enumerate(etapas):
            eixo = eixos[linha, coluna]
            if coluna == 0:
                eixo.imshow(etapa)
            else:
                eixo.imshow(etapa, cmap="gray")
            eixo.set_title(f"{classe}: {titulos[coluna]}")
            eixo.axis("off")

    figura.suptitle("Análise exploratória: destaque de possíveis trincas e ranhuras")
    figura.tight_layout()
    caminho_grafico = os.path.join(pasta_saida, "processamento_opencv.png")
    figura.savefig(caminho_grafico, dpi=150)
    plt.close(figura)
    print(f"Gráfico OpenCV salvo em: {caminho_grafico}")


def carregar_dados(pasta_dataset):
    """Carrega imagens e separa automaticamente treino e validação."""
    treino = tf.keras.utils.image_dataset_from_directory(
        pasta_dataset,
        validation_split=0.2,
        subset="training",
        seed=SEMENTE,
        image_size=TAMANHO_IMAGEM,
        batch_size=TAMANHO_LOTE,
        label_mode="binary"
    )
    validacao = tf.keras.utils.image_dataset_from_directory(
        pasta_dataset,
        validation_split=0.2,
        subset="validation",
        seed=SEMENTE,
        image_size=TAMANHO_IMAGEM,
        batch_size=TAMANHO_LOTE,
        label_mode="binary"
    )

    print("Classes encontradas:", treino.class_names)
    # Mantém os dados em memória para deixar o treinamento mais rápido.
    autotune = tf.data.AUTOTUNE
    return treino.cache().prefetch(autotune), validacao.cache().prefetch(autotune)


def criar_modelo():
    """Monta uma CNN básica com data augmentation dinâmico."""
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.15),
    ], name="data_augmentation")

    modelo = models.Sequential([
        layers.Input(shape=(TAMANHO_IMAGEM[0], TAMANHO_IMAGEM[1], 3)),
        augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(1, activation="sigmoid")
    ])
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return modelo


def salvar_graficos(historico, pasta_saida):
    """Gera um gráfico com loss e acurácia de treino e validação."""
    figura, eixos = plt.subplots(1, 2, figsize=(12, 4))
    epocas = range(1, len(historico.history["loss"]) + 1)

    eixos[0].plot(epocas, historico.history["loss"], label="Loss de treino")
    eixos[0].plot(epocas, historico.history["val_loss"], label="Loss de validação")
    eixos[0].set_title("Curva de Loss")
    eixos[0].set_xlabel("Épocas")
    eixos[0].legend()

    eixos[1].plot(epocas, historico.history["accuracy"], label="Acurácia de treino")
    eixos[1].plot(epocas, historico.history["val_accuracy"], label="Acurácia de validação")
    eixos[1].set_title("Curva de Acurácia")
    eixos[1].set_xlabel("Épocas")
    eixos[1].legend()

    figura.tight_layout()
    caminho_grafico = os.path.join(pasta_saida, "treinamento.png")
    figura.savefig(caminho_grafico, dpi=150)
    plt.close(figura)
    print(f"Gráfico do treinamento salvo em: {caminho_grafico}")


def main():
    parser = argparse.ArgumentParser(description="Inspeção visual de peças fundidas")
    parser.add_argument("--dataset", required=True, help="Pasta que contém def_front e ok_front")
    parser.add_argument("--saida", default="resultados", help="Pasta para salvar gráficos e modelo")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEMENTE)
    criar_pastas_saida(args.saida)
    mostrar_processamento_classico(args.dataset, args.saida)
    treino, validacao = carregar_dados(args.dataset)
    modelo = criar_modelo()
    modelo.summary()

    historico = modelo.fit(treino, validation_data=validacao, epochs=EPOCAS)
    salvar_graficos(historico, args.saida)
    caminho_modelo = os.path.join(args.saida, "modelo_pecas.keras")
    modelo.save(caminho_modelo)
    print(f"Modelo salvo em: {caminho_modelo}")


if __name__ == "__main__":
    main()
