# Verificação dos artefatos finais

Auditoria executada em 22 out. 2026. A primeira etapa conferiu os arquivos e recalculou métricas a partir das previsões salvas. Na revisão complementar, o ZIP foi extraído em uma pasta isolada, os dados foram preparados novamente e os seis checkpoints foram avaliados sem novo treinamento. Os modelos, hiperparâmetros e resultados experimentais originais foram preservados.

## Verificações executadas

- Os hashes SHA-256 dos seis `best.pt` coincidem com os registros em `metrics.json` e com os candidatos de `selection.json`.
- Os hashes dos manifests de treino, validação e teste coincidem com o protocolo congelado. Todos os modelos usam a mesma divisão. Na inspeção dos manifests, são 1.768/379/380 exemplos; os três pares de pixels idênticos estão somente no treino.
- Os seis checkpoints foram carregados em CPU com `weights_only=True`; configuração, ordem de classes, época e F1 de validação coincidem com os arquivos de métricas.
- Nos dois pares com/sem pesos, as configurações diferem somente em `id` e `weighted`. Nas três CNNs sem pesos, diferem somente em `id`, largura e altura. A taxa final da MLP é 0,0001; a da CNN é 0,001, diferença declarada no artigo.
- Cada arquivo de previsões contém exatamente os mesmos 380 caminhos de teste, na ordem do manifesto, com rótulos reais iguais aos do manifesto. As probabilidades somam 1 dentro da tolerância de 0,00001, o argmax coincide com a previsão e a confiança coincide com a probabilidade prevista.
- Acurácia, F1 macro, precisão/recall/F1/suporte das seis classes e matrizes de confusão foram recalculados com scikit-learn a partir dessas previsões; coincidem com os artefatos. As porcentagens das matrizes usam o suporte da linha.
- A melhor época de cada run coincide com o máximo F1 de validação e o desempate por menor perda. Quantidade de épocas e soma dos tempos coincidem com os históricos.
- Os registros de conclusão dos seis treinos antecedem o congelamento; os seis registros de avaliação são posteriores. O hash da seleção registrado nas métricas coincide com o arquivo final.

## Resultados conferidos

| Experimento | F1 macro de validação | Acurácia de teste | F1 macro de teste |
|---|---:|---:|---:|
| MLP 128 | 0,382769 | 0,428947 | 0,354322 |
| MLP 128 + pesos | 0,333535 | 0,363158 | 0,302775 |
| CNN 128 | 0,633256 | 0,626316 | 0,594358 |
| CNN 128 + pesos | 0,619206 | 0,510526 | 0,499688 |
| CNN 256 | 0,573630 | 0,615789 | 0,560937 |
| CNN 512 | 0,471916 | 0,568421 | 0,449550 |

A seleção pela validação confirma `cnn_128`. Esse modelo acertou 238 e errou 142 das 380 imagens de teste.

Os deltas manuais por classe presentes no artigo foram conferidos: na MLP, os pesos alteram F1 de glass em −34,1 pontos percentuais e cardboard em −18,9; o recall de trash vai de 0% a 60%. Na CNN, os deltas de F1 são metal +1,4, cardboard +0,7, paper −23,5 e plastic −15,9 pontos percentuais; o recall de trash cai de 40% para 30%.

A consistência dos registros não comprova externamente todo o histórico de execução nem a participação humana. As verificações executáveis e a inspeção visual complementares estão descritas abaixo.

## Verificações complementares da execução

- O comando `python -m pytest -q` passou em todos os **26 testes**. A suíte verifica classes, estratificação, proteção contra duplicatas, perda ponderada, invariância da CNN, seleção sem teste, retomada determinística e avaliação sem novo treinamento em dados sintéticos.
- Os dois notebooks foram executados integralmente, em ordem: seis células de código na análise exploratória e doze em resultados. As saídas incluem tabelas e figuras reais, sem células de erro. Usou-se IPython no mesmo processo para dispensar sockets de kernel bloqueados neste ambiente.
- O PDF foi gerado em uma coluna e contém dez páginas. A revisão visual verificou títulos, tabelas, legendas, gráficos, matrizes e referências; nomes e matrículas permaneceram provisórios por solicitação do usuário. Naquela revisão, a publicação no GitHub e seu link ainda estavam pendentes; a atualização de identificação está registrada ao final deste documento.
- O ZIP de distribuição contém os seis melhores modelos, os registros e os códigos, mas exclui dataset bruto, caches, ambiente Python e checkpoints de retomada. O download verificado reconstrói os dados.

## Revisão complementar e reprodução em cópia isolada

- O ZIP anterior à revisão foi extraído fora da pasta de trabalho do projeto. Sua integridade e os 162 arquivos listados no manifesto SHA-256 foram conferidos. O mesmo ZIP de dados foi fornecido localmente e validado pelo script de download: não se repetiu o download pela rede.
- Executaram-se `src.prepare` e `src.evaluate` nessa cópia. Os quatro manifestos (geral, treino, validação e teste) foram reconstruídos de forma idêntica. Os seis modelos produziram **2.280 previsões**: os seis CSVs de previsões, seis relatórios por classe, doze matrizes (contagens/percentuais) e o resumo global ficaram **idênticos byte a byte** aos registros originais.
- Após corrigir as proteções do código, repetiram-se preparação e avaliação na cópia isolada, novamente com igualdade exata. O JSON `docs/REPRODUCIBILIDADE.json` registra o resultado por modelo. Os arquivos de avaliação originais não foram sobrescritos por essa conferência.
- Os dois notebooks foram novamente executados integralmente na cópia isolada com o código revisado: **18 células de código**, sem erro. A execução usou IPython no mesmo processo, conforme metadados dos notebooks, pois este ambiente restringe sockets de kernel.
- A suíte completa passou: **26 testes em 5,83 segundos**. Inclui regressões para bloqueio de busca/pilotos após teste, dropout efetivo, recusa de opções não implementadas, preservação dos manifestos congelados e detecção de divergência de configuração antes da inferência.
- `pip check` não encontrou dependências quebradas. A reprodução usou o ambiente Python instalado que gerou o estudo; **não** foi feita uma instalação nova das dependências em outra máquina ou sistema operacional.
- Todas as dez páginas finais foram inspecionadas visualmente. Incluíram-se a tabela da busca preliminar, sua separação dos seis treinos finais e a numeração atualizada. Também foram corrigidas a pontuação duplicada e a descrição bibliográfica do evento de Adam.

## Correções que preservam os experimentos

1. A busca interna `--pilot` agora respeita a mesma proibição de iniciar novos ajustes após o teste, antes de criar diretórios ou treinar.
2. O construtor aplica o dropout declarado no JSON. O valor dos seis modelos existentes já era 0,3; seus pesos e comportamento foram preservados.
3. Configurações que o código não implementa (como SGD, CUDA ou aumento de dados) são recusadas, em vez de registradas silenciosamente sem execução.
4. `prepare` recusa uma nova divisão incompatível com a seleção congelada antes de alterar os manifestos publicados.
5. `evaluate` verifica a configuração atual, o registro do treino e os metadados do checkpoint antes de abrir imagens de teste.

O atendimento item a item e as pendências de entrega constam de `docs/REVISAO_RUBRICA.md`. Nenhuma destas conferências garante nota nem substitui a revisão exigida da equipe.

## Atualização de identificação — 30 out. 2026

O usuário informou a mudança para o **Grupo 1** e a renomeação da organização e do repositório já existentes para [Redes-Neurais-Unifor-Grupo-1](https://github.com/Redes-Neurais-Unifor-Grupo-1/Redes-Neurais-Unifor-Grupo-1). A identificação e o link foram atualizados na documentação local. Esta atualização ainda precisa ser publicada; o acesso externo e o conteúdo remoto não foram reconferidos.

Este registro não representa novo treinamento, nova avaliação científica ou revisão humana dos resultados. As verificações anteriores e seus resultados permanecem como registro histórico. Nomes e matrículas continuam provisórios por solicitação do usuário; a revisão pela equipe e a entrega no AVA continuam pendentes.
