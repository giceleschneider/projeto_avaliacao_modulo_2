# Inspeção de peças fundidas com OpenCV e TensorFlow

Projeto para classificar imagens de peças fundidas como `def_front` ou `ok_front` usando processamento de imagem com OpenCV e uma rede neural convolucional (CNN) em TensorFlow/Keras.

## Objetivo

1. Aplicar filtros e etapas de processamento clássico com OpenCV para destacar possíveis trincas, ranhuras ou imperfeições.
2. Carregar o dataset em lotes com `tf.keras.utils.image_dataset_from_directory`.
3. Treinar uma CNN simples para distinguir peças defeituosas de peças aprovadas.
4. Salvar gráficos de treinamento e o modelo treinado em uma pasta local de saída.

## Estrutura do projeto

```text
projeto_avaliacao_modulo_2/
├── main.py                  # fluxo principal do projeto
├── executar_projeto.py      # script com execução direta
├── requirements.txt         # dependências do projeto
├── README.md                # documentação do projeto
├── dados/                   # dataset local do projeto
├── resultados/              # gráficos e modelo gerados em execução
├── .gitignore               # arquivos locais que não devem ir para o repositório
└── .venv/                   # ambiente virtual local (opcional)
```

## Requisitos

- Python 3.10+
- pip
- bibliotecas de [requirements.txt](requirements.txt)

### Dependências principais

- `opencv-python`
- `tensorflow`
- `matplotlib`
- `numpy`

## Como configurar

Crie e ative um ambiente virtual, se preferir:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
python3 -m pip install -r requirements.txt
```

## Como executar

### Opção 1: usando o script principal

```bash
python3 main.py --dataset /caminho/para/dados
```

### Opção 2: usando o script pré-configurado

```bash
python3 executar_projeto.py
```

> O script `executar_projeto.py` já usa o caminho local do dataset dentro do workspace e chama o fluxo principal do projeto.

## Estrutura esperada do dataset

O dataset deve ficar em uma pasta com esta organização:

```text
dados/
├── def_front/
│   ├── imagem_001.jpg
│   └── ...
└── ok_front/
    ├── imagem_001.jpg
    └── ...
```

## Saídas geradas

Ao executar o projeto, a pasta `resultados/` é criada e recebe:

- `processamento_opencv.png`: comparação das etapas do OpenCV (original, cinza, blur, limiar, Canny e morfologia)
- `treinamento.png`: curvas de loss e accuracy
- `modelo_pecas.keras`: modelo treinado em Keras

## Observações importantes

- A ordem das pastas define os rótulos: `def_front` recebe label 0 e `ok_front` recebe label 1.
- O augmentation é aplicado somente durante o treinamento para aumentar a diversidade dos exemplos.
- O projeto usa imagens com tamanho reduzido para acelerar o treinamento e facilitar testes locais.
- Arquivos gerados localmente, dados pesados e o dataset não precisam ficar no repositório remoto; por isso estão no `.gitignore`.

## Desenvolvimento

A rede implementada é uma CNN simples com:

- `Conv2D`
- `MaxPooling2D`
- `Flatten`
- `Dense`
- `Dropout`
- otimizador `Adam`

A configuração atual usa 10 épocas e lote de 32 imagens, mantendo o processo leve e adequado para estudo e avaliação inicial.

## Roteiro de apresentação

“Primeiro apliquei processamento de imagem com OpenCV para realçar regiões de interesse nas peças. Em seguida, carreguei o dataset com Keras, apliquei aumento de dados e treinei uma CNN para distinguir peças OK de defeituosas. Por fim, gerei gráficos de aprendizado e salvei o modelo treinado na pasta de resultados.”
