"""Executa o projeto com o caminho das imagens já configurado.

"""

import main


PASTA_DATASET = "/Users/giceleschneider/Documents/Projetos/projeto_avaliacao_modulo_2/dados"
PASTA_SAIDA = "resultados"


if __name__ == "__main__":
    main.tf.keras.utils.set_random_seed(main.SEMENTE)
    main.criar_pastas_saida(PASTA_SAIDA)
    main.mostrar_processamento_classico(PASTA_DATASET, PASTA_SAIDA)
    treino, validacao = main.carregar_dados(PASTA_DATASET)
    modelo = main.criar_modelo()
    modelo.summary()

    historico = modelo.fit(treino, validation_data=validacao, epochs=main.EPOCAS)
    main.salvar_graficos(historico, PASTA_SAIDA)
    modelo.save(f"{PASTA_SAIDA}/modelo_pecas.keras")
    print("Projeto concluído. Veja os arquivos na pasta resultados.")
