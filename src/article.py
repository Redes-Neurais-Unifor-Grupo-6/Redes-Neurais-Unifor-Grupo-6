"""Gera artigo de dez páginas somente a partir de experimentos concluídos.

Identificação e URL são editáveis em docs/metadata.json. Nunca preenche resultados
ausentes com valores simulados. O Markdown exportado permite revisão do texto.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether

IDS = ['mlp_128', 'mlp_128_weighted', 'cnn_128', 'cnn_128_weighted', 'cnn_256', 'cnn_512']
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
NAMES = ['MLP 128', 'MLP 128 + pesos', 'CNN 128', 'CNN 128 + pesos', 'CNN 256', 'CNN 512']
W = A4[0] - 108

def dec(value, digits=3):
    return f'{float(value):.{digits}f}'.replace('.', ',')

def pct(value, digits=1):
    return dec(float(value) * 100, digits) + '%'

def intfmt(value):
    return f'{int(value):,}'.replace(',', '.')

def pp(value):
    return ('+' if value >= 0 else '') + dec(value * 100, 1) + ' p.p.'

class Article:
    def __init__(self, root):
        self.root = root
        self.meta = json.loads((root / 'docs/metadata.json').read_text())
        self.audit = json.loads((root / 'data/audit.json').read_text())
        self.runs = {id_: json.loads((root / f'artifacts/runs/{id_}/metrics.json').read_text()) for id_ in IDS}
        self.histories = {id_: pd.read_csv(root / f'artifacts/runs/{id_}/history.csv') for id_ in IDS}
        self.predictions = {id_: pd.read_csv(root / f'artifacts/runs/{id_}/test_predictions.csv') for id_ in IDS}
        self.best_id = min(IDS, key=lambda i: (-self.runs[i]['val_macro_f1'], self.runs[i]['val_loss'], i))
        self.best = self.runs[self.best_id]
        self.best_name = NAMES[IDS.index(self.best_id)]
        self.story, self.markdown = [], []
        self.styles = {
            'body': ParagraphStyle('body', fontName='Times-Roman', fontSize=12, leading=14, alignment=TA_JUSTIFY, spaceAfter=7),
            'heading': ParagraphStyle('heading', fontName='Times-Bold', fontSize=12, leading=15, spaceAfter=9, spaceBefore=4),
            'title': ParagraphStyle('title', fontName='Times-Bold', fontSize=16, leading=18, alignment=TA_CENTER, spaceAfter=13),
            'small': ParagraphStyle('small', fontName='Times-Roman', fontSize=10, leading=12, spaceAfter=6),
            'caption': ParagraphStyle('caption', fontName='Times-Roman', fontSize=10, leading=12, spaceAfter=4),
            'reference': ParagraphStyle('reference', fontName='Times-Roman', fontSize=11, leading=13, spaceAfter=8, wordWrap='LTR'),
        }

    def p(self, text, style='body'):
        self.story.append(Paragraph(text, self.styles[style]))
        self.markdown.append(text + '\n')

    def heading(self, text):
        self.p(text, 'heading')

    def page(self):
        self.story.append(PageBreak())
        self.markdown.append('\n---\n')

    def tab(self, rows, widths=None, size=9.5, record=True):
        style = ParagraphStyle('cell', fontName='Times-Roman', fontSize=size, leading=size+2)
        cells = [[Paragraph(str(v), style) for v in row] for row in rows]
        table = Table(cells, colWidths=widths, hAlign='LEFT', repeatRows=1)
        table.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, 0), .7, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), .5, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), .7, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eef1f3')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ]))
        if record:
            self.story.extend([table, Spacer(1, 5)])
            self.markdown.append('\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in [rows[0], ['---']*len(rows[0]), *rows[1:]])+'\n')
        return table

    def figure(self, filename, caption, max_height=500, width=W):
        path = self.root / 'artifacts/figures' / filename
        if not path.exists():
            raise FileNotFoundError(path)
        iw, ih = PILImage.open(path).size
        scale = min(width / iw, max_height / ih)
        self.p(caption, 'caption')
        self.story.append(Image(str(path), width=iw*scale, height=ih*scale))
        self.p('Fonte: TrashNet (Thung; Yang, [2016]); processamento descrito neste estudo.' if filename in {'class_distribution.png', 'class_examples.png', 'resolution_examples.png'} else 'Fonte: registros dos experimentos descritos neste estudo.', 'small')
        self.markdown.append(f'![{caption}](../artifacts/figures/{filename})\n')

    def build(self):
        b = self.best
        train_n = sum(self.audit['split_counts']['train'].values())
        val_n = sum(self.audit['split_counts']['val'].values())
        test_n = sum(self.audit['split_counts']['test'].values())
        mlp, mlpw, cnn, cnnw, cnn256, cnn512 = [self.runs[i] for i in IDS]
        search = pd.read_csv(self.root / 'artifacts/search/search_summary.csv')
        final_config = json.loads((self.root / 'configs/experiments.json').read_text())
        mlp_lr = next(c for c in final_config['experiments'] if c['id'] == 'mlp_128').get('learning_rate', final_config['protocol']['learning_rate'])
        # 1 - identidade, resumo e problema científico.
        self.p(html.escape(self.meta['title']), 'title')
        authors = '; '.join(f"{html.escape(a['name'])} (matrícula {html.escape(a['registration'])})" for a in self.meta['authors'])
        self.p(authors, 'small')
        url = self.meta.get('github_url')
        self.p(f'GitHub: <link href="{html.escape(url)}">{html.escape(url)}</link>' if url else 'GitHub: <b>pendente de publicação e inserção do link.</b>', 'small')
        self.heading('Resumo')
        self.p(f'Este estudo compara redes perceptron multicamadas (MLP) e convolucionais (CNN) na classificação de seis categorias do TrashNet. Foram executados seis experimentos controlados: ambas as arquiteturas em 128 × 96 pixels, com e sem ponderação de classes, e a mesma CNN sem pesos em 256 × 192 e 512 × 384 pixels. A partição estratificada contém {train_n} imagens de treino, {val_n} de validação e {test_n} de teste. Três pares de imagens idênticas com rótulos conflitantes foram mantidos dentro de uma única partição por par, evitando vazamento. A seleção utilizou exclusivamente F1 macro de validação. O modelo escolhido, {self.best_name}, obteve {pct(b["test_accuracy"])} de acurácia e {dec(b["test_macro_f1"])} de F1 macro no teste. As diferenças de desempenho são discutidas em conjunto com tempo, memória, desequilíbrio e limitações do conjunto. Os achados descrevem uma única semente e partição, sem inferência de superioridade estatística.')
        self.p('<b>Palavras-chave:</b> redes neurais; classificação de imagens; TrashNet; desequilíbrio de classes; custo computacional.', 'small')
        self.heading('1 Introdução')
        self.p('A identificação visual de materiais pode apoiar sistemas de triagem de resíduos, mas diferenças de aparência dentro de uma categoria e semelhanças entre materiais tornam a classificação desafiadora. O TrashNet reúne objetos fotografados em condições relativamente controladas e permite estudar esse problema em escala computacional acessível (Thung; Yang, [2016]).')
        self.p('O objetivo é investigar três questões: como MLP e CNN se comportam sob condições equivalentes; como pesos inversos à frequência afetam as classes, especialmente <i>trash</i>; e se aumentar a resolução da CNN produz ganhos compatíveis com seu custo. A comparação é controlada, sem pré-treinamento, e prioriza F1 macro para dar igual importância às seis categorias.')
        self.heading('2 Materiais e métodos')
        self.p('Utilizou-se exclusivamente o arquivo <i>dataset-resized.zip</i> indicado na atividade, com 2.527 imagens. A ordem das classes em toda a análise é glass, paper, cardboard, plastic, metal e trash, com 501, 594, 403, 482, 410 e 137 arquivos, respectivamente. A razão entre a maior e a menor classe é 4,34:1. A auditoria confirmou imagens RGB de 512 × 384 pixels, sem arquivos corrompidos.')
        # 2 - desenho experimental fechado antes do teste.
        self.page()
        self.heading('2.1 Partição, integridade e pré-processamento')
        self.p(f'A divisão estratificada usou <i>train_test_split</i> e semente 42: 70% para treino e duas metades do restante para validação e teste ({train_n}/{val_n}/{test_n}). Os manifestos registram caminho, rótulo, partição e hashes SHA-256 do arquivo e dos pixels. Encontraram-se três pares idênticos com rótulos divergentes: glass115/metal91, glass176/plastic152 e glass389/plastic332. Preservaram-se os rótulos e as contagens por classe de cada conjunto. Uma troca determinística entre arquivos da mesma classe (plastic332 por plastic1) reuniu os três pares no treino. A troca não dependeu de métricas dos modelos.')
        self.p('Cada resolução foi gerada diretamente da referência, com conversão RGB e interpolação Lanczos; 512 × 384 preserva o tamanho original. Os pixels foram divididos por 255, sem estimar estatísticas no teste. Não houve aumento, reamostragem ou exclusão da classe trash. Os caches foram preparados antes da medição de tempo. Em PyTorch, a entrada segue lote × canais × altura × largura.')
        self.heading('2.2 Arquiteturas e treinamento')
        self.p(f'A MLP achata os 36.864 valores RGB e utiliza camadas densas de 128 e 64 unidades, ReLU e dropout de 0,3, seguidas por seis logits ({intfmt(mlp["parameters"])} parâmetros). A CNN usa convoluções 3 × 3 com 16, 32 e 64 filtros, padding 1, BatchNorm, ReLU e max-pooling 2 × 2 em cada bloco; a primeira convolução tem passo 2. Global average pooling, dropout de 0,3 e uma camada linear de seis saídas completam a rede ({intfmt(cnn["parameters"])} parâmetros em todas as resoluções).')
        self.p('A MLP fornece uma referência sem estrutura espacial explícita. Na CNN, compartilhamento de filtros e agregação global reduzem parâmetros e permitem variar apenas a resolução. PyTorch (Paszke et al., 2019) implementa as redes; NumPy (Harris et al., 2020), pandas (McKinney, 2010), Matplotlib (Hunter, 2007) e Pillow (2026) apoiam dados e figuras. Usou-se Adam (Kingma; Ba, 2015) por sua adaptação das atualizações, lote 32 para caber na RAM, semente 42 e sem decaimento de pesos. A entropia cruzada recebe logits e expressa o erro multiclasse. O orçamento fixo de 30 épocas e a paciência 7 limitam custo e conservam a melhor validação.')
        self.p('O melhor checkpoint maximiza F1 macro de validação; empate usa menor perda. Só a melhoria estrita de F1 reinicia a paciência. A ponderação usa peso(c) = N / (6 × n(c)), com N como total de treino e n(c) como contagem da classe c no treino. Arquitetura, inicialização, ordem dos lotes e demais configurações são preservadas entre os pares com e sem pesos. A extensão YOLO não foi realizada.')
        self.p('Tabela 1 - Busca preliminar na MLP sem pesos, antes do teste. Demais configurações iguais às descritas acima; custos separados dos seis treinos finais.', 'caption')
        rows = [['Taxa Adam', 'F1 macro val.', 'Época salva / total', 'Tempo total (s)']]
        for row in search.sort_values('learning_rate').itertuples():
            rows.append([dec(row.learning_rate, 4), dec(row.val_macro_f1), f'{row.best_epoch}/{row.epochs_trained}', dec(row.train_seconds, 1)])
        self.tab(rows, [100, 110, 145, W-355], 10)
        self.p(f'Fonte: artifacts/search. A taxa {dec(mlp_lr, 4)} venceu pela validação e foi aplicada a dois novos treinos MLP. As CNNs mantiveram 0,001, taxa inicial fixa. As taxas distintas limitam atribuir diferenças entre MLP e CNN somente à arquitetura; a busca não demonstra otimalidade.', 'small')
        self.heading('2.3 Avaliação e custo')
        self.p('A avaliação dos seis checkpoints no mesmo teste ocorreu após concluir treinos e seleção por validação. Usou-se scikit-learn (Pedregosa et al., 2011), seis rótulos e <i>zero_division=0</i>: razões indefinidas recebem zero. F1 macro é a média dos seis F1. O tempo inclui treino e validação por época, excluindo download, preparação e gravação de checkpoints. Cada experimento roda em processo próprio. RSS é amostrado a cada 100 ms no laço, incluindo salvamentos e páginas mapeadas, e mede memória do processo.')
        # 3 - EDA com todas as figuras exigidas.
        self.page()
        self.heading('3 Resultados: análise exploratória')
        self.p('A estratificação mantém a predominância de paper e a menor representação de trash (Figura 1). Os exemplos da Figura 2 são do treinamento. Variações de cor, formato, transparência, amassamento e iluminação podem aproximar materiais diferentes. Os três tamanhos da Figura 3 preservam a proporção 4:3; detalhes finos podem desaparecer na redução.', 'body')
        self.figure('class_distribution.png', 'Figura 1 - Quantidade de imagens por classe e conjunto.', 150)
        self.figure('class_examples.png', 'Figura 2 - Dois exemplos de treino de cada classe do TrashNet.', 188)
        self.figure('resolution_examples.png', 'Figura 3 - A mesma imagem nas três resoluções de entrada (largura × altura).', 120)
        self.p('Os seis arquivos de três pares conflitantes constituem 0,24% do conjunto. Agrupar duplicatas evita vazamento exato, mas não corrige ruído de rótulo nem comprova ausência de objetos repetidos ou imagens quase idênticas.', 'small')
        # 4 - tabelas agregadas e resolução.
        self.page()
        self.heading('3.1 Desempenho global e custo computacional')
        self.p('Tabela 2 - Validação e teste dos seis experimentos; pesos referem-se à função de perda.', 'caption')
        rows = [['ID', 'Modelo', 'Entrada', 'Pesos', 'F1 val.', 'Acur. teste', 'F1 teste']]
        for k, id_ in enumerate(IDS):
            m = self.runs[id_]
            rows.append([f'E{k+1}', m['architecture'].upper(), f'{m["image_width"]} × {m["image_height"]}', 'Sim' if m['weighted'] else 'Não', dec(m['val_macro_f1']), pct(m['test_accuracy']), dec(m['test_macro_f1'])])
        self.tab(rows, [27, 47, 73, 39, 67, 74, W-327], 10)
        self.p('Fonte: registros dos experimentos. Seleção: maior F1 de validação, antes do teste.', 'small')
        self.p('Tabela 3 - Custo medido; época escolhida / épocas executadas e tamanho do checkpoint.', 'caption')
        rows = [['ID', 'Parâmetros', 'Modelo MiB', 'Épocas', 'Total s', 's/época', 'RSS MiB']]
        for k, id_ in enumerate(IDS):
            m = self.runs[id_]
            rows.append([f'E{k+1}', f'{int(m["parameters"]):,}'.replace(',', '.'), dec(m['model_size_bytes']/1024**2, 2), f'{m["best_epoch"]}/{m["epochs_trained"]}', dec(m['train_seconds'],1), dec(m['mean_epoch_seconds'],1), dec(m['peak_rss_mb'],1)])
        self.tab(rows, [27, 74, 67, 57, 82, 74, W-381], 9.6)
        self.p('Fonte: perf_counter e RSS do processo. MiB = 2<super>20</super> bytes; tamanho do arquivo com pesos e metadados.', 'small')
        hardware = cnn['hardware']
        self.p(f'Hardware: Intel Core i3-1125G4; {dec(hardware.get("ram_total_gib", 0), 1)} GiB de RAM; quatro threads CPU, sem GPU. Python {hardware.get("python", "")}; PyTorch {hardware.get("torch", "")}; lote 32 em todos os treinos. Tempos de parede dependem também da carga do sistema, que não é dedicado.', 'small')
        self.figure('resolution_performance.png', 'Figura 4 - Acurácia e F1 macro no teste da CNN sem pesos, por resolução.', 130)
        self.figure('resolution_cost.png', 'Figura 5 - Tempo médio por época e total da CNN sem pesos, por resolução.', 130)
        self.p(f'As resoluções 256 e 512 reduziram F1 macro em {dec((cnn["test_macro_f1"]-cnn256["test_macro_f1"])*100,1)} e {dec((cnn["test_macro_f1"]-cnn512["test_macro_f1"])*100,1)} p.p. frente a 128. Em 512, o tempo por época aumentou {dec(cnn512["mean_epoch_seconds"]/cnn["mean_epoch_seconds"],1)} vezes, o total {dec(cnn512["train_seconds"]/cnn["train_seconds"],1)} vezes e o RSS {dec(cnn512["peak_rss_mb"]/cnn["peak_rss_mb"],1)} vezes. Neste protocolo, o aumento de resolução não compensou o custo.', 'small')
        # 5 - 36 métricas por classe, todas explícitas.
        self.page()
        self.heading('3.2 Desempenho por classe e desequilíbrio')
        self.p('Tabela 4 - Precisão (P), recall (R), F1 e suporte (N) no teste. Cada painel corresponde ao ID da Tabela 2.', 'caption')
        from sklearn.metrics import classification_report
        reports = {}
        tables = []
        for k, id_ in enumerate(IDS):
            pred = self.predictions[id_]
            report = classification_report(pred.y_true, pred.y_pred, labels=list(range(6)), target_names=CLASSES, output_dict=True, zero_division=0)
            reports[id_] = report
            rows = [[f'E{k+1} - {NAMES[k]}', 'P', 'R', 'F1', 'N']]
            rows += [[c, dec(report[c]['precision'],2), dec(report[c]['recall'],2), dec(report[c]['f1-score'],2), str(int(report[c]['support']))] for c in CLASSES]
            tables.append(self.tab(rows, [98, 33, 33, 33, 32], 9.5, False))
            self.markdown.append('\n'+'\n'.join('| '+' | '.join(row)+' |' for row in [rows[0], ['---']*5, *rows[1:]])+'\n')
        panel = Table([[tables[0], tables[1]], [tables[2], tables[3]], [tables[4], tables[5]]], colWidths=[W/2, W/2])
        panel.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),14)]))
        self.story.append(panel)
        self.p('Fonte: previsões dos checkpoints no teste. Valores arredondados apenas na apresentação; CSVs preservam a precisão completa.', 'small')
        for base, weighted, name in [('mlp_128','mlp_128_weighted','MLP'), ('cnn_128','cnn_128_weighted','CNN')]:
            r0, r1 = reports[base]['trash'], reports[weighted]['trash']
            self.p(f'Na {name}, os pesos alteraram o recall de trash de {dec(r0["recall"])} para {dec(r1["recall"])} e seu F1 de {dec(r0["f1-score"])} para {dec(r1["f1-score"])}. A variação do F1 macro foi {pp(self.runs[weighted]["test_macro_f1"]-self.runs[base]["test_macro_f1"])} O aumento da contribuição de classes raras na perda não garante melhoria de todas as categorias; os ganhos e perdas específicos estão na Tabela 4.')
        self.p('Na MLP, o ganho de recall de trash (0% para 60%) custou reconhecimento de outras classes: o F1 de glass caiu 34,1 p.p. e o de cardboard, 18,9 p.p.; não houve nenhuma previsão glass com pesos. Na CNN, pesos elevaram ligeiramente F1 de metal (+1,4 p.p.) e cardboard (+0,7 p.p.), mas reduziram paper (-23,5 p.p.) e plastic (-15,9 p.p.). Logo, o maior peso da minoria não produziu melhoria global e sequer beneficiou trash na CNN.', 'small')
        zeros = [(NAMES[k], c) for k, id_ in enumerate(IDS) for c in CLASSES if (self.predictions[id_].y_pred == CLASSES.index(c)).sum() == 0]
        if zeros:
            self.p('Não houve previsões para as seguintes combinações modelo/classe: '+ '; '.join(f'{n}/{c}' for n,c in zeros) + '. Nesses casos, precisão indefinida foi registrada como zero; isso evidencia falha de reconhecimento, não ausência de exemplos reais.', 'small')
        else:
            self.p('Todos os modelos produziram ao menos uma previsão para cada classe. A convenção zero_division=0 foi mantida de forma uniforme, embora não tenha sido necessária para precisão neste teste.', 'small')
        self.p(f'Em 128 × 96, a CNN alterou o F1 macro de teste em {pp(cnn["test_macro_f1"]-mlp["test_macro_f1"])} frente à MLP sem pesos e em {pp(cnnw["test_macro_f1"]-mlpw["test_macro_f1"])} com pesos. A MLP possui cerca de {int(mlp["parameters"]/cnn["parameters"])} vezes o número de parâmetros da CNN; filtros locais e compartilhados oferecem uma representação espacial diferente do vetor de pixels. A evidência descreve estas configurações, incluindo as taxas distintas, e não toda a família de modelos.', 'small')
        # 6 e 7 - curvas com escalas legíveis.
        for part, ids in [(1, IDS[:3]), (2, IDS[3:])]:
            self.page()
            self.heading(f'3.3 Curvas de aprendizagem - painel {part}')
            self.figure(f'learning_curves_{part}.png', f'Figura {5+part} - Perda e acurácia de treino e validação: '+', '.join(NAMES[IDS.index(i)] for i in ids)+'.', 515)
            self.p('Treino é medido durante a otimização, com dropout e, na CNN, BatchNorm em modo treino; validação ocorre após a época em modo avaliação. Com pesos, apenas a perda de treino é ponderada. Um desnível isolado entre curvas não demonstra sobreajuste.', 'small')
            for id_ in ids:
                h = self.histories[id_]
                m = self.runs[id_]
                self.p(f'{NAMES[IDS.index(id_)]}: época salva {m["best_epoch"]}/{m["epochs_trained"]}, F1 val. {dec(m["val_macro_f1"])}; acurácias de treino/validação na última época {pct(h.iloc[-1].train_accuracy)}/{pct(h.iloc[-1].val_accuracy)}.', 'small')
            if part == 1:
                self.p('A MLP sem pesos reduz perdas e melhora a acurácia, mas estabiliza em desempenho limitado. Com pesos, a validação oscila e para antes. Na CNN, o treino continua melhorando enquanto a validação piora após o checkpoint: comportamento compatível com sobreajuste e instabilidade, que justifica recuperar a época 10.', 'small')
            else:
                self.p('Nas três CNNs deste painel, a perda de treino cai, mas a validação apresenta picos e não sustenta a melhora. A resolução maior não resolve essa instabilidade. O modelo 512 foi retomado após interrupção na época 11, preservando otimizador e estados aleatórios; os intervalos ociosos não entram no tempo informado.', 'small')
        # 8 - seis matrizes, contagens e porcentagens por linha.
        self.page()
        self.heading('3.4 Matrizes de confusão')
        self.figure('confusion_matrices.png', 'Figura 8 - Seis matrizes de teste: contagens e porcentagens por linha; classe real nas linhas e prevista nas colunas.', 570)
        self.p('A ordem das classes é fixa em todos os painéis. A diagonal representa acertos; as demais células distinguem falhas específicas que a acurácia agregada pode ocultar. As porcentagens usam como denominador o suporte da classe real, com pequenos desvios de 100% pela apresentação arredondada.', 'small')
        pred = self.predictions[self.best_id]
        wrong = pred[pred.y_true != pred.y_pred]
        pair = wrong.groupby(['y_true','y_pred']).size().sort_values(ascending=False)
        if len(pair):
            pairs = [f'{CLASSES[int(a)]} → {CLASSES[int(z)]} ({int(n)} imagens)' for (a,z), n in pair.iloc[:3].items()]
            self.p(f'No modelo escolhido por validação ({self.best_name}), as confusões direcionais mais frequentes foram '+ '; '.join(pairs)+'. A inspeção visual dos erros complementa essas contagens na Figura 9.', 'small')
        # 9 - erros, limites, referência.
        self.page()
        self.heading('4 Discussão dos erros e das limitações')
        self.p(f'O modelo selecionado foi {self.best_name}, com F1 macro de validação {dec(b["val_macro_f1"])}. Ele errou {len(wrong)} das {test_n} imagens de teste. A Figura 9 apresenta até seis erros de maior probabilidade atribuída à classe prevista, com desempate pelo caminho do arquivo. Esta amostra evidencia erros confiantes; não é uma seleção aleatória nem representa toda a distribuição de falhas.')
        self.figure('selected_model_errors.png', f'Figura 9 - Erros de teste de {self.best_name}: classe real, previsão e probabilidade softmax da classe prevista.', 270)
        self.p('Em plastic364 e metal97, rótulos impressos ocupam parte relevante da imagem e podem aproximá-los visualmente de paper. Cardboard277 mostra uma caixa impressa; cardboard33, uma superfície clara e lisa. Glass501 tem corpo transparente e etiqueta, compatíveis visualmente com embalagens plásticas. Metal37 aparece em recorte com reflexo suave. Essas observações sugerem hipóteses de confusão, sem provar quais sinais a rede utilizou. A probabilidade softmax não foi calibrada: confiança alta não garante acerto.')
        self.p('A coleta em fundo predominantemente claro e com objetos isolados não cobre adequadamente lixo misturado, oclusão, fundos urbanos, sujeira ou iluminação variável. O conjunto tem apenas 2.527 imagens, rótulos conflitantes conhecidos e não informa uma identidade de objeto para particionar diferentes fotografias do mesmo item. Agrupar hashes exatos não resolve todas essas dependências. Uma única semente, uma única divisão e tempos em CPU compartilhada limitam generalizações e comparações de custo.')
        self.heading('4.1 Comparação contextual com o estudo de referência')
        self.p('Yang e Thung (2016) relataram 63% de acurácia com SVM em seis classes e 22% com CNN em cinco, após excluir trash. O SVM usou divisão 70/30; a CNN, 70/13/17 e imagens 256 × 256. Mudanças de classes, partições, proporção de imagem e treinamento impedem tratar esses números como um confronto experimental direto. Os aproximadamente 75% mencionados no README do TrashNet pertencem à continuação posterior dos experimentos, não ao resultado publicado naquele artigo (Thung; Yang, [2016]).')
        # 10 - síntese e referências com URLs rastreáveis.
        self.page()
        self.heading('5 Conclusão')
        self.p(f'Os seis experimentos foram executados sob um protocolo comum, preservando as classes e o teste. O modelo escolhido por validação foi {self.best_name}, com {pct(b["test_accuracy"])} de acurácia e F1 macro {dec(b["test_macro_f1"])} no teste. As tabelas por classe mostram que o efeito da ponderação deve ser avaliado além da acurácia. Neste protocolo, as resoluções maiores pioraram o F1 macro e aumentaram tempo e memória; 128 × 96 ofereceu a melhor relação entre desempenho e custo entre as CNNs avaliadas. Os resultados sustentam uma comparação descritiva deste conjunto e não estabelecem superioridade universal.')
        self.p('Próximos estudos devem repetir sementes e partições, revisar os rótulos conflitantes, ampliar a diversidade de imagens e testar aumento de dados ou transferência de aprendizagem em um protocolo separado. A seleção dessas melhorias deve continuar restrita à validação; uma nova investigação após observar este teste exige avaliação final independente.')
        self.heading('Referências')
        refs = [
            ('HARRIS, Charles R. et al. Array programming with NumPy. <b>Nature</b>, v. 585, p. 357-362, 2020.', 'https://doi.org/10.1038/s41586-020-2649-2'),
            ('HUNTER, John D. Matplotlib: a 2D graphics environment. <b>Computing in Science &amp; Engineering</b>, v. 9, n. 3, p. 90-95, 2007.', 'https://doi.org/10.1109/MCSE.2007.55'),
            ('KINGMA, Diederik P.; BA, Jimmy. Adam: a method for stochastic optimization. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 3., 2015, San Diego. <b>Proceedings [...]</b>. [S. l.: s. n.], 2015.', 'https://arxiv.org/abs/1412.6980'),
            ('MCKINNEY, Wes. Data structures for statistical computing in Python. In: PYTHON IN SCIENCE CONFERENCE, 9., 2010. <b>Proceedings [...]</b>. [S. l.], 2010. p. 56-61.', 'https://doi.org/10.25080/Majora-92bf1922-00a'),
            ('PASZKE, Adam et al. PyTorch: an imperative style, high-performance deep learning library. <b>Advances in Neural Information Processing Systems</b>, v. 32, 2019.', 'https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library'),
            ('PEDREGOSA, Fabian et al. Scikit-learn: machine learning in Python. <b>Journal of Machine Learning Research</b>, v. 12, n. 85, p. 2825-2830, 2011.', 'https://jmlr.org/papers/v12/pedregosa11a.html'),
            ('PILLOW. <b>Pillow documentation</b>. Versão 13.0.0. [S. l.], 2026.', 'https://pillow.readthedocs.io/en/stable/'),
            ('THUNG, Gary; YANG, Mindy. <b>TrashNet</b>. GitHub, [2016]. Repositório e conjunto de dados.', 'https://github.com/garythung/trashnet'),
            ('YANG, Mindy; THUNG, Gary. <b>Classification of trash for recyclability status</b>. Stanford: Stanford University, 2016. Relatório CS229.', 'https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf'),
        ]
        for citation, link in refs:
            self.p(citation+f' Disponível em: <link href="{link}" color="#173f63">{link}</link>. Acesso em: {('22 out. 2026' if citation.startswith(('HARRIS', 'HUNTER', 'MCKINNEY', 'PILLOW')) else self.meta['access_date'])}.', 'reference')
        target = self.root / 'docs/artigo.pdf'
        def footer(canvas, doc):
            canvas.setFont('Times-Roman', 9)
            canvas.drawRightString(A4[0]-54, 29, str(doc.page))
        document = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=54, leftMargin=54, topMargin=42, bottomMargin=48, title=self.meta['title'], author='; '.join(a['name'] for a in self.meta['authors']))
        document.build(self.story, onFirstPage=footer, onLaterPages=footer)
        (self.root / 'docs/artigo.md').write_text('\n'.join(self.markdown), encoding='utf-8')
        from pypdf import PdfReader
        count = len(PdfReader(target).pages)
        if count != 10:
            raise RuntimeError(f'O artigo gerado tem {count} páginas; revisar o layout antes da entrega.')
        print(f'Artigo e texto editável gerados: {target} ({count} páginas)')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    Article(p.parse_args().root.resolve()).build()

if __name__ == '__main__':
    main()
