# Revisão de atendimento à atividade

**Data:** 22 out. 2026. **Escopo:** comparação entre o enunciado da atividade, o artigo PDF, os dois notebooks, README e registros tabulares do projeto. Nomes e matrículas não integram esta revisão. A revisão documental não alterou experimentos nem realizou treinamento, inferência ou testes de integração. As verificações técnicas complementares estão em `docs/VERIFICACAO.md`.

**Documento examinado:** `docs/artigo.pdf`, 10 páginas A4, SHA-256 `dee035d9b3f546c6dc04147393fe54004d06d9c959b91290bad24abef5edfe26`. Foram examinados texto extraído e imagens renderizadas das dez páginas; as páginas alteradas durante o acabamento foram novamente inspecionadas. A revisão identifica esta versão: qualquer regeneração exige conferir novamente paginação e trechos alterados.

**Resultado:** todos os elementos científicos e gráficos obrigatórios foram encontrados no artigo e no notebook de resultados. As tabelas e afirmações numéricas examinadas são consistentes com os registros. Não foi encontrado erro científico ou numérico material no PDF inspecionado. A busca opcional está registrada em tabela no artigo, no notebook e em CSV. Permanecem pendentes o GitHub e seu link na primeira página. A submissão no AVA e a participação/revisão efetiva da equipe dependem de ações externas e não são comprovadas pelos arquivos; por isso, esta revisão não declara a entrega integralmente concluída nem garante uma nota.

## Critérios usados nesta auditoria

- **Atendido documentalmente:** há evidência legível no lugar exigido e não foi encontrada contradição na conferência realizada. Não equivale a refazer experimentalmente o projeto.
- **Pendente:** exigência objetiva que os arquivos ainda mostram incompleta.
- **Não verificável nesta revisão:** depende de ação externa ou participação humana que documentos não comprovam.
- **Opcional / não realizado:** não é falta dos seis experimentos obrigatórios.
- **Melhoria recomendada:** refinamento real, sem alegação de falta obrigatória não sustentada pelo enunciado.

Os caminhos abaixo são relativos à raiz do projeto `trashnet/`. As páginas mencionadas são do artigo final, não do enunciado. Os números de células contam todas as células do notebook, incluindo Markdown.

## Matriz: dados, preparação e treinamento

| Requisito do enunciado | Evidência concreta | Status / observação |
|---|---|---|
| Usar apenas `dataset-resized.zip`, 2.527 imagens 512 × 384 | Artigo p. 1–2; README “Dados”; `data/audit.json` com total, dimensão e hash do ZIP | Atendido documentalmente. |
| Preservar seis classes, incluindo trash, com contagens de referência | Artigo p. 1; Figura 1 p. 3; tabela de divisão no README | Atendido; 501/594/403/482/410/137 na ordem exigida. |
| Análise exploratória de quantidade, dimensão, canais e integridade | Artigo p. 1–2; notebook exploratório células 3–5 | Atendido; RGB, 512 × 384, nenhum arquivo corrompido e duplicatas explicitadas. |
| Examinar exemplos das seis classes e dificuldades visuais | Artigo p. 3, Figura 2 e comentário; notebook exploratório células 8–9 | Atendido; dois exemplos por classe e hipóteses visuais contextualizadas. |
| Documentar problemas encontrados e tratamento | Artigo p. 2–3; `data/audit.json`; notebook exploratório células 4–5 | Atendido; três pares conflitantes preservados no treino; uma troca determinística documentada. |
| Divisão estratificada 70/15/15, semente fixa e registrada | Artigo p. 2; `data/{train,val,test}.csv`; README com tabela por classe | Atendido documentalmente; 1.768/379/380, semente 42, arredondamento inteiro explicado. |
| Mesma divisão nos seis experimentos | Artigo p. 2; manifesto/protocolo; configuração e seleção publicadas localmente | Atendido documentalmente; consistência de hashes foi objeto da auditoria técnica separada, não repetida aqui. |
| RGB, resolução derivada da referência e normalização documentada | Artigo p. 2; notebook exploratório células 10–11 | Atendido; Lanczos, divisão por 255, 512 × 384 preservada, proporção 4:3. |
| Aumento/balanceamento somente no treino, quando usados | Artigo p. 2; README; notebook exploratório célula 10 | Atendido; sem aumento/reamostragem; pesos da perda calculados só do treino. |
| Versões/duplicatas da mesma imagem na mesma partição | Artigo p. 2; auditoria e notebook exploratório célula 5 | Atendido para duplicatas exatas identificadas; a limitação para quase duplicatas está declarada. |
| MLP sem pesos, 128 × 96 | Artigo Tabelas 2–3 E1; Figura 6; `artifacts/runs/mlp_128/` | Atendido documentalmente. |
| MLP com pesos, 128 × 96 | Artigo Tabelas 2–3 E2; Figura 6; `artifacts/runs/mlp_128_weighted/` | Atendido documentalmente. |
| CNN sem pesos, 128 × 96 | Artigo Tabelas 2–3 E3; Figura 6; `artifacts/runs/cnn_128/` | Atendido documentalmente. |
| CNN com pesos, 128 × 96 | Artigo Tabelas 2–3 E4; Figura 7; `artifacts/runs/cnn_128_weighted/` | Atendido documentalmente. |
| CNN sem pesos, 256 × 192 | Artigo Tabelas 2–3 E5; Figura 7; `artifacts/runs/cnn_256/` | Atendido documentalmente. |
| CNN sem pesos, 512 × 384 | Artigo Tabelas 2–3 E6; Figura 7; `artifacts/runs/cnn_512/` | Atendido documentalmente; retomada após interrupção explicitada. |
| Descrever e justificar arquiteturas, perda, otimizador, taxa, lote e parada | Artigo p. 2: arquitetura completa, entropia cruzada/logits, Adam, lote 32, orçamento 30/paciência 7 e busca da taxa | Atendido. As justificativas são concisas, mas estão presentes e não prometem otimalidade. |
| Registrar semente e época do checkpoint | Artigo p. 2 e Tabela 3; comentários das Figuras 6–7; registros | Atendido; distingue época escolhida de total executado. |
| Ponderação com maior contribuição das classes raras e pesos só do treino | Artigo p. 2: N/(6 × n(c)); notebook exploratório célula 7 exibe pesos | Atendido. |
| Repetir cada par com/sem pesos com mesma configuração e reinício | Artigo p. 2; notebook resultados célula 5 exibe configs finais | Atendido documentalmente; taxa MLP 0,0001 em ambos, CNN 0,001 em ambos. |
| Mesma CNN, filtros, profundidade, semente, otimizador, taxa, parada, hardware e lote entre resoluções | Artigo p. 2 e p. 4; 24.198 parâmetros em todas; configs | Atendido documentalmente; lote 32 não foi alterado. |
| Seleção por F1 macro de validação; teste só após escolhas | Artigo p. 2, Tabela 2 e p. 9; `artifacts/selection.json`; notebook resultados células 1–3 | Atendido documentalmente; vencedor CNN 128. A busca prévia é separada, e a diferença de taxas entre arquiteturas é declarada. |

## Matriz: métricas, tabelas, gráficos e análise

O enunciado exige os elementos desta seção **tanto no artigo quanto no notebook de resultados**. Ambos foram localizados.

| Requisito | Artigo | Notebook `02_resultados.ipynb` / arquivos | Status |
|---|---|---|---|
| Acurácia e F1 macro no teste dos seis modelos | Tabela 2, p. 4 | Célula 9; `artifacts/tables/performance.csv` | Atendido. |
| Precisão, recall, F1 e suporte de todas as classes, para os seis | Tabela 4, seis painéis, p. 5 | Célula 11; `per_class_comparison.csv` | Atendido; 36 combinações modelo/classe. |
| Parâmetros e tamanho salvo | Tabela 3, p. 4 | Célula 9; `computational_cost.csv` | Atendido; bytes convertidos para MiB, não confundidos com RAM. |
| Épocas, tempo total e tempo médio por época | Tabela 3, p. 4 | Célula 9; históricos e métricas | Atendido; época salva/total separados. |
| Hardware, lote e método de medição | Texto p. 2 e p. 4 | Célula 9 com hardware e método de cada execução; README | Atendido; CPU i3-1125G4, 14,8 GiB, quatro threads, lote 32. |
| Excluir download e preparação do custo de treinamento | Texto p. 2 | Células 8–9; README | Atendido documentalmente; também exclui gravação dos checkpoints do tempo das épocas. |
| Pico RAM/GPU, quando possível, ou motivo da ausência | Tabela 3 e texto p. 2/4 | Células 8–9 | Atendido; RSS amostrado, limites explicados; CPU sem GPU. |
| Barras das contagens por classe em treino/val/teste | Figura 1, p. 3 | Célula 7 | Atendido; 18 barras com contagens e legenda. |
| Pelo menos duas imagens por classe | Figura 2, p. 3 | Célula 7 | Atendido; 12 imagens, classes e identificadores visíveis. |
| Mesma imagem nas três resoluções | Figura 3, p. 3 | Célula 7 | Atendido; mesma referência e três títulos de resolução. |
| Perda e acurácia de treino/validação para seis experimentos | Figuras 6–7, p. 6–7 | Célula 13 | Atendido; 12 eixos, linhas de treino/validação e checkpoint. |
| Uma matriz de confusão por modelo | Figura 8, p. 8 | Célula 15 | Atendido; seis matrizes. |
| Contagens e percentuais por linha | Figura 8, p. 8 | Célula 15; CSVs de cada run | Atendido; ambas as informações na célula. |
| Real nas linhas, prevista nas colunas, ordem glass/paper/cardboard/plastic/metal/trash | Figura 8, p. 8 | Células 10/14–15 | Atendido; eixos e ordem visíveis. |
| Gráfico de acurácia e F1 de teste da CNN sem pesos por resolução | Figura 4, p. 4 | Célula 19 | Atendido; três resoluções. |
| Gráfico de tempo médio e total por resolução | Figura 5, p. 4 | Célula 19 | Atendido; dois painéis com unidades próprias (s/época e minutos totais). |
| Escolher modelo de erros pelo maior F1 macro de validação | P. 9 e Tabela 2 | Células 3 e 21 | Atendido; CNN 128, F1 val. 0,633256. |
| Até seis erros, rótulo real/previsto e probabilidade prevista | Figura 9, p. 9 | Célula 21; `selected_model_errors.csv` | Atendido; seis erros e probabilidades, dos 142 erros totais. |
| `zero_division=0` e explicação do comportamento observado | P. 2 e p. 5 | Células 8 e 11 | Atendido; explicita classes sem previsão, sem confundir com suporte zero. |

## Matriz: discussão, artigo e entrega

| Requisito | Evidência | Status / observação |
|---|---|---|
| Comparar MLP/CNN em 128 × 96 com e sem pesos | P. 5: deltas de F1 +24,0/+19,7 p.p.; notebook célula 17 | Atendido. P. 2/5 e notebook reconhecem taxas distintas; não atribuem causalidade só à arquitetura. |
| Interpretar curvas, aprendizagem e sobreajuste | P. 6–7; notebook células 12 e 22 | Atendido; descreve tendências e instabilidade, com ressalva sobre perdas ponderadas e modos de treino/avaliação. |
| Efeito dos pesos sobre trash e demais classes, ganhos/perdas | P. 5, Tabela 4 e três parágrafos interpretativos; notebook célula 17/22 | Atendido; mostra benefício em trash da MLP, prejuízo global e fracasso da ponderação para trash da CNN. |
| Resolução versus desempenho, tempo por época, total e memória | P. 4, p. 10; notebook célula 19/22 | Atendido; medidas sustentam que imagens maiores não compensaram neste protocolo. |
| Classes confundidas e hipóteses visuais dos erros | P. 8–9; notebook célula 21/22 | Atendido; confusões direcionais e seis imagens concretas. |
| Limitações para fotografias de situações reais | P. 9–10; notebook célula 22 | Atendido; objetos isolados, fundos, oclusão, sujeira, iluminação, única semente e dependência entre imagens. |
| Comparar com Yang/Thung: 63% SVM seis classes, 22% CNN cinco | P. 9; notebook célula 22 | Atendido; números coincidem com os citados no próprio enunciado. |
| Explicar que classes/divisão/resolução/treino impedem comparação direta | P. 9 | Atendido; explicita 70/30, 70/13/17, 256 × 256 e diferenças de protocolo. |
| Distinguir ~75% posterior do resultado publicado original, se mencionado | P. 9; README; notebook célula 22 | Atendido; confirmado no README original do TrashNet. |
| Introdução, métodos, resultados, discussão, conclusão e referências | P. 1–10 | Atendido; discussão também acompanha as tabelas e curvas, além da seção 4. |
| Até dez páginas, uma coluna, sem capa separada | PDF e revisão visual de todas as páginas | Atendido; dez páginas, texto em uma coluna. Painéis de tabelas/figuras são permitidos pelo enunciado. |
| Título e resumo na primeira página | P. 1 | Atendido. Nomes/matrículas fora do escopo desta auditoria. |
| Link do GitHub na primeira página | P. 1 diz “pendente de publicação e inserção do link”; `metadata.json` tem URL nula | **Pendente.** É requisito explícito ainda não atendido. |
| Citações e referências de fontes externas | Chamadas autor-data p. 1–2/9 e referências p. 10 | Atendido nos elementos auditados. Não foi emitida certificação de conformidade integral com normas ABNT. As fontes e o alcance da conferência estão descritos abaixo. |
| Figuras/tabelas numeradas, interpretadas e legíveis | Figuras 1–9 e Tabelas 1–4, p. 3–9 | Atendido; sem cortes ou sobreposições percebidos. Matrizes têm texto menor, mas legível na renderização inspecionada. |
| README com dependências/versões/dados/instalação/pastas/execução | README; requisitos exatos, portáveis e lock | Atendido documentalmente; instalação não foi repetida nesta auditoria. |
| Avaliação dos modelos sem novo treinamento | Seção própria do README, sequência download/prepare/evaluate/reporting/article | Atendido documentalmente; execução técnica em revisão separada. |
| Notebook exploratório com inspeção, distribuição, exemplos e tratamento | `01_analise_exploratoria.ipynb`, 12 células, seis de código | Atendido; seis células executadas e nenhuma saída de erro salva. |
| Scripts/notebooks para os seis treinos com configs | `src/train.py`, `src/models.py`, `configs/experiments.json`, README | Atendido documentalmente. |
| Notebook de métricas, figuras e análise | `02_resultados.ipynb`, 24 células, 12 de código | Atendido; 12 células executadas e nenhuma saída de erro salva. |
| Seis modelos salvos, divisão, configs, históricos, métricas e previsões | `artifacts/runs/<id>/`, `data/`, `artifacts/selection.json` | Atendido no pacote local; acesso público depende da publicação. |
| Código e modelos acessíveis no GitHub | README informa repositório ainda não criado | **Pendente.** Preparar arquivos localmente não equivale à publicação pedida. |
| Cada integrante enviar o mesmo PDF no AVA até 27/11/2026 23h59 | README/guia instruem; não há comprovante de submissão | Não verificável; ação de entrega ainda externa ao pacote. |
| Revisão/compreensão humana e condução pela equipe, com IA como apoio | `docs/USO_DE_IA.md` e guia declaram revisão pendente | Não verificável; documentação não comprova participação. Não atestar atendimento integral dessa condição. |
| Projeto com atribuição das fontes externas, sem declaração falsa de revisão | Artigo, README e `USO_DE_IA.md` | Atribuições presentes; não houve investigação externa de plágio ou confirmação de autoria humana. |
| YOLO de classificação com comparação adicional | Artigo p. 2 e README declaram não realizado | Opcional / não realizado; não é uma falha da rubrica obrigatória e não gera bônus. |

## Busca opcional de hiperparâmetros

O enunciado recomenda registrar configurações e F1 macro de validação em tabela, separando a busca dos seis experimentos controlados. Esse registro está presente nos três formatos:

- **Artigo, Tabela 1, p. 2:** taxas, F1 macro de validação, época salva/total e tempo da investigação preliminar da MLP sem pesos.
- **Notebook de resultados, célula 5:** tabela completa, decisão e configurações finais.
- **Registros:** `artifacts/search/search_summary.csv`, `decision.json` e resultados dos pilotos, separados dos seis modelos finais.

| Taxa Adam | F1 macro de validação | Época salva / total | Tempo (s) |
|---:|---:|---:|---:|
| 0,0001 | 0,382769 | 23/30 | 79,4 |
| 0,0003 | 0,127707 | 3/10 | 24,6 |
| 0,0010 | 0,063390 | 1/8 | 16,5 |

A taxa 0,0001 foi selecionada antes do teste e aplicada aos dois treinos finais da MLP. As CNNs mantiveram 0,001. O artigo esclarece que a busca não prova otimalidade e que a diferença de taxas impede atribuir toda diferença entre MLP e CNN exclusivamente à arquitetura. O tempo do piloto vencedor difere do novo treino final; os dois valores pertencem a execuções distintas e estão corretamente separados.

**Status:** atendido. A numeração final é Tabela 1 (busca), Tabela 2 (desempenho global), Tabela 3 (custo) e Tabela 4 (métricas por classe). As referências cruzadas do texto foram conferidas.

## Conferência científica e numérica

As seguintes verificações foram feitas por confronto dos valores exibidos com os CSVs/JSONs e por aritmética simples, sem nova inferência ou recomputação integral das métricas:

| Afirmação do artigo | Valor nos registros / conferência | Resultado |
|---|---|---|
| CNN 128 escolhida por validação | F1 val. 0,6332564852; demais candidatos têm menor F1 | Correto. |
| CNN escolhida: 62,6% e F1 0,594 no teste | 0,6263157895 e 0,5943579320 | Arredondamento correto. |
| 142 erros em 380 imagens | Acurácia corresponde a 238 acertos; metadados de erros registram 142 | Correto. |
| Melhor época/total E1–E6 | 23/30, 8/15, 10/17, 8/15, 10/17, 11/18 | Tabela 3 e comentários coerentes com registros. |
| F1 CNN−MLP sem/com pesos | +24,0035827 / +19,6913131 p.p. | Texto +24,0/+19,7 correto. |
| ΔF1 macro ponderação MLP/CNN | −5,1546933 / −9,4669629 p.p. | Texto −5,2/−9,5 correto. |
| F1 de trash MLP 0 → 0,353; recall 0 → 0,600 | CSV: 0 → 0,3529411765; 0 → 0,6 | Correto. |
| F1 de trash CNN 0,432 → 0,261; recall 0,400 → 0,300 | CSV: 0,4324324324 → 0,2608695652; 0,4 → 0,3 | Correto. |
| Deltas F1 de glass/cardboard na MLP | −34,1463 / −18,8633 p.p. | Texto −34,1/−18,9 correto. |
| Deltas F1 metal/cardboard/paper/plastic na CNN | +1,3836 / +0,6896 / −23,5467 / −15,8824 p.p. | Texto +1,4/+0,7/−23,5/−15,9 correto. |
| Queda F1 nas resoluções 256/512 | −3,3421204 / −14,4807693 p.p. | Texto −3,3/−14,5 correto. |
| Razões 512/128 de tempo médio/total/RSS | 14,786557 / 15,656355 / 4,024330 | Texto 14,8×/15,7×/4,0× correto. |
| Razão de parâmetros MLP/CNN | 4.727.366 / 24.198 = 195,361848 | “Cerca de 195 vezes” correto. |
| Classe maior/menor | 594/137 = aproximadamente 4,34 | Correto. |
| Seis arquivos conflitantes equivalem a 0,24% | 6/2527 × 100 = aproximadamente 0,2374% | Correto. |
| Confusões mais frequentes do modelo escolhido | glass→plastic 18; glass→metal 17; metal→glass 13 | Correspondem à matriz de CNN 128. |
| Probabilidades dos seis erros ilustrados | 93,0727%; 90,2334%; 86,0508%; 85,0493%; 84,6071%; 84,0567% | Figura mostra 93,1/90,2/86,1/85,0/84,6/84,1%, corretamente. |
| Tabelas por classe | 36 linhas; suportes 76/89/60/73/62/20 em cada modelo | Valores de P/R/F1 conferidos contra o CSV, com duas casas; sem divergência encontrada. |

As afirmações metodológicas também são adequadamente limitadas: global average pooling mantém os parâmetros, não o custo; maior peso não garante melhora da minoria; softmax não foi calibrada; o agrupamento de hashes não elimina quase duplicatas; a observação das curvas é descrita como compatível com sobreajuste/instabilidade, não como demonstração causal. A comparação MLP/CNN explicita a diferença de taxas. A conclusão é restrita a esta divisão/semente e propõe avaliação independente para novas escolhas.

## Citações e apresentação

As nove fontes das referências aparecem em chamadas no texto ou identificam o dataset/estudo comparado: Harris, Hunter, Kingma/Ba, McKinney, Paszke, Pedregosa, Pillow, Thung/Yang e Yang/Thung. Não foram encontrados tokens de ferramenta, citações sem identificação ou referências claramente inventadas. As imagens exploratórias são atribuídas ao TrashNet; gráficos e tabelas são atribuídos aos registros deste estudo.

A conferência de fontes primárias incluiu:

- [TrashNet](https://github.com/garythung/trashnet): confirma contagens, coleta, dimensões e caráter posterior dos aproximadamente 75%.
- [Adam](https://arxiv.org/abs/1412.6980): confirma autores e publicação na terceira ICLR, San Diego, 2015. A referência final inclui esses dados; o depósito inicial no arXiv em 2014 não torna incorreto o ano de conferência 2015.
- [Scikit-learn](https://jmlr.org/papers/v12/pedregosa11a.html): confirma autores, ano, volume e páginas.
- [PyTorch](https://papers.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html): confirma título, autores, NeurIPS 32 e 2019.
- [Pillow](https://pillow.readthedocs.io/en/stable/): confirma documentação da versão 13.0.0.
- [Orientação oficial de citação do Matplotlib](https://matplotlib.org/3.6.1/users/project/citing.html): confirma Hunter, 2007, volume 9, número 3, páginas 90–95 e DOI.
- Os registros primários indexados de [NumPy/Nature](https://doi.org/10.1038/s41586-020-2649-2) e [pandas/SciPy](https://doi.org/10.25080/Majora-92bf1922-00a) são compatíveis com os dados bibliográficos utilizados. Algumas tentativas de acesso direto aos DOIs/PDFs encontraram restrições de acesso ou timeout; isso não demonstra que os links sejam inválidos.

O [relatório original de Yang e Thung](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf) apresentou timeout nas novas tentativas desta rodada. Os números 63%/22% foram confrontados com o próprio enunciado, e as informações adicionais de protocolo, com a pesquisa preservada em `docs/REFERENCIAS_E_PROTOCOLO.md`. Não se declara nova leitura integral remota desse relatório nesta rodada.

A revisão visual verificou uma coluna de texto, ausência de capa separada, limite de dez páginas, numeração, legibilidade, títulos, fontes e legendas. Não foram percebidos cortes ou sobreposições. Os seis painéis por classe e as seis matrizes são apresentados como painéis, permitidos pelo enunciado. A pontuação duplicada em “p.p.” foi corrigida. Essas conferências não constituem certificação integral de todas as normas ABNT aplicáveis.

## Pendências para entrega e alcance da revisão

1. **GitHub e link na primeira página:** criar/publicar o repositório, conferir acesso público aos seis `best.pt`, inserir a URL real em `docs/metadata.json` e regenerar o artigo. Não preencher com URL fictícia.
2. **AVA:** cada integrante deve enviar o mesmo PDF, até 27/11/2026 às 23h59. A existência do pacote local ou do repositório não comprova submissão; o enunciado torna o PDF no AVA condição de avaliação.
3. **Participação da equipe:** os integrantes devem conduzir a revisão, compreender o código e assumir análise e decisões, conforme o enunciado. `docs/USO_DE_IA.md` e `docs/GUIA_DA_EQUIPE.md` organizam essa etapa, mas não comprovam que ela ocorreu. Não há declaração de revisão humana concluída.

A tabela da busca e os refinamentos de referência/pontuação identificados durante a revisão foram incorporados ao PDF inspecionado. Não foram identificadas outras lacunas obrigatórias de conteúdo nesta auditoria documental. A avaliação do professor inclui interpretação, participação e entrega efetiva; portanto, não há garantia de pontuação.
