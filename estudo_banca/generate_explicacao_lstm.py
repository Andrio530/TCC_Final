# -*- coding: utf-8 -*-
"""
Gera o documento TeX da explicação didática da LSTM e compila para PDF.
"""
from pathlib import Path

tex_path = Path("estudo_banca") / "explicacao_lstm_decisoes.tex"

tex_content = r'''\documentclass[10pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazilian]{babel}
\usepackage{lmodern}
\usepackage[margin=1.6cm,top=1.8cm,bottom=1.8cm]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs,tabularx,array}
\usepackage{float}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{fancyhdr}
\usepackage[most,breakable,skins]{tcolorbox}
\usepackage[colorlinks=true,linkcolor=NavyBlue,citecolor=NavyBlue,urlcolor=NavyBlue]{hyperref}

% Cores institucionais
\definecolor{NavyBlue}{RGB}{16, 44, 87}
\definecolor{MidnightBlue}{RGB}{24, 43, 73}
\definecolor{DarkSlateGray}{RGB}{38, 70, 83}
\definecolor{TealAccent}{RGB}{42, 157, 143}
\definecolor{AlertRed}{RGB}{180, 40, 40}
\definecolor{LightBg}{RGB}{248, 249, 250}
\definecolor{MathBg}{RGB}{238, 244, 251}
\definecolor{WarningBg}{RGB}{254, 243, 243}
\definecolor{LightGreenBg}{RGB}{235, 247, 238}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\textbf{\textsf{GUIA EXPLICATIVO: MODELAGEM TEMPORAL E PROJEÇÃO PREDITIVA (LSTM)}}}
\fancyhead[R]{\footnotesize\textsf{Andrio Gabriel Epp | TCC Engenharia da Computação}}
\fancyfoot[L]{\footnotesize\textsf{Decisões de Engenharia, Janelamento, Normalização e Horizontes}}
\fancyfoot[R]{\footnotesize\textbf{\thepage}}
\renewcommand{\headrulewidth}{0.6pt}
\renewcommand{\footrulewidth}{0.4pt}

\newtcolorbox{conceitobox}[1][]{
  breakable,
  colback=LightBg,
  colframe=NavyBlue,
  fonttitle=\bfseries\sffamily\color{white},
  coltitle=white,
  boxrule=0.7pt,
  arc=2mm,
  title={#1},
  left=6pt,right=6pt,top=4pt,bottom=4pt
}

\newtcolorbox{decisaobox}[1][]{
  breakable,
  colback=MathBg,
  colframe=MidnightBlue,
  fonttitle=\bfseries\sffamily\color{white},
  coltitle=white,
  boxrule=0.7pt,
  arc=2mm,
  title={#1},
  left=6pt,right=6pt,top=4pt,bottom=4pt
}

\newtcolorbox{bancabox}[1][]{
  breakable,
  colback=LightGreenBg,
  colframe=TealAccent,
  fonttitle=\bfseries\sffamily\color{white},
  coltitle=white,
  boxrule=0.7pt,
  arc=2mm,
  title={#1},
  left=6pt,right=6pt,top=4pt,bottom=4pt
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{4.5pt}

\begin{document}

\begin{center}
  {\Large\textbf{\textsf{\color{NavyBlue}GUIA DE ENGENHARIA DA MODELAGEM TEMPORAL (LSTM)}}}\\[3pt]
  {\large\textbf{\textsf{Entendimento Conceitual, Decisões de Projeto e Argumentação de Defesa}}}\\[4pt]
  \textbf{Aluno:} Andrio Gabriel Epp \quad|\quad \textbf{Curso:} Engenharia da Computação (Unisinos) \quad|\quad \textbf{Foco:} Módulo Temporal (Track A)
\end{center}

\vspace{-4pt}
\noindent\makebox[\linewidth]{\rule{\textwidth}{1.0pt}}

\section{Visão Geral: O que este módulo faz e por quê?}

No nosso TCC, antes de classificar o tipo de falha do transformador com o XGBoost, nós construímos um \textbf{módulo de previsão temporal} baseado em rede neural profunda recorrente (\textbf{Stacked LSTM}).

A pergunta prática de engenharia que este módulo responde é simples:
\begin{quote}
\textit{``Olhando o histórico recente das medições dos sensores de gases dissolvidos no óleo de um transformador de potência, quais serão as concentrações desses gases daqui a 7 dias ($h=14$) e daqui a 14 dias ($h=28$)?''}
\end{quote}

A ideia original da arquitetura híbrida era: se entregarmos ao classificador não apenas a foto do instante atual (\textit{snapshot}), mas também \textbf{para onde os gases estão caminhando no futuro}, o modelo terá muito mais inteligência para identificar falhas graves antes que o transformador queime.

Abaixo estão explicadas, de forma direta e sem formalismo matemático desnecessário, as 5 grandes decisões de engenharia desta etapa.

---

\section{Decisão 1: Por que a janela deslizante ótima é $w = 50$ (25 dias)?}

\begin{conceitobox}[O que é a Janela Deslizante ($w$)?]
Pense na janela deslizante como um \textbf{buffer circular (fila FIFO)} na memória do computador. 
\begin{itemize}[leftmargin=*,itemsep=1pt]
  \item Nossos dados registram medições a cada \textbf{12 horas} (1 passo temporal = 12 horas).
  \item Uma janela $w = 10$ armazena 10 medições passadas = \textbf{5 dias de histórico}.
  \item Uma janela $w = 50$ armazena 50 medições passadas = \textbf{25 dias de histórico}.
\end{itemize}
\end{conceitobox}

\begin{decisaobox}[Como chegamos em $w=50$ e qual foi a lógica de escolha?]
Nós não chutamos esse número. Foi realizado um \textbf{Grid Search} (busca em grade exaustiva), treinando e avaliando a rede LSTM com quatro tamanhos de buffer diferentes:
$$w \in \{10, 20, 30, 50\}$$

A física do transformador e os testes computacionais ditaram o resultado:
\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item \textbf{Janelas curtas ($w=10$ e $w=20$ / 5 a 10 dias):} Transformadores de potência operam imersos em dezenas de toneladas de óleo isolante. A difusão dos gases e o aquecimento são fenômenos físicos de inércia muito lenta. Em 5 ou 10 dias, a rede enxerga apenas pequenas flutuações diárias de carga e temperatura ambiente. O erro de predição foi significativamente mais alto porque a rede confundia ruído com tendência real.
  \item \textbf{Janela longa ($w=50$ / 25 dias):} Com 25 dias contínuos de histórico, o buffer contém quase um mês de operação. A rede consegue enxergar a inclinação real da curva de degradação térmica do óleo. 
  \item \textbf{Por que não testar janelas ainda maiores (ex: $w=100$)?} Cada série da base Katser possui 420 passos no total. Se usássemos uma janela de 100 passos, perderíamos quase um quarto dos dados apenas para ``encher a fila'' antes de gerar o primeiro alvo, além de aumentar o consumo de memória RAM e GPU sem trazer ganho perceptível.
\end{enumerate}
\textbf{Resultado Numérico Real:} $w^* = 50$ alcançou o menor Erro Absoluto Médio global ($\text{MAE} = 5,59 \times 10^{-5}$ normalizado) e coeficiente de determinação quase perfeito ($R^2 > 0,999$ no primeiro passo), sendo o vencedor absoluto do teste.
\end{decisaobox}

---

\section{Decisão 2: Para que serve a Normalização Min-Max e por que é estritamente no Treino?}

\begin{conceitobox}[O Problema Físico da Escala dos Gases]
Em um transformador em serviço, os gases dissolvidos coexistem em ordens de grandeza absurdamente desiguais:
\begin{itemize}[leftmargin=*,itemsep=1pt]
  \item O Monóxido de Carbono ($\text{CO}$) atinge facilmente \textbf{$1.000$ a $3.000$ ppm} (partes por milhão).
  \item O Acetileno ($\text{C}_2\text{H}_2$) aparece em concentrações diminutas de \textbf{$1$ a $15$ ppm}.
\end{itemize}
\textbf{O perigo computacional:} Se você injetar esses números brutos na rede neural, o valor de $3.000$ do $\text{CO}$ domina o cálculo do gradiente e do erro quadrático. A rede fica cega para o Acetileno, porque um erro de $2$ ppm no acetileno parece insignificante perto de $3.000$ ppm do CO.\\
\textbf{A gravidade na Engenharia Elétrica:} O Acetileno ($\text{C}_2\text{H}_2$) é exatamente o gás mais perigoso de todos! Qualquer fração de ppm dele indica a ocorrência de arco elétrico em altíssima temperatura (> 1.000°C), com risco iminente de explosão do transformador. A rede jamais poderia negligenciar o acetileno.
\end{conceitobox}

\begin{decisaobox}[A Solução: ScalerPerGas e a Blindagem de Treino]
Para resolver isso, implementamos o \texttt{ScalerPerGas}, que aplica a fórmula Min-Max $[0, 1]$ \textbf{individualmente para cada gás}:
$$x_{\text{norm}}^{(g)} = \frac{x^{(g)} - \min(x^{(g)})}{\max(x^{(g)}) - \min(x^{(g)}) + \epsilon}$$
Agora, tanto o $\text{CO}$ quanto o $\text{C}_2\text{H}_2$ variam entre 0 e 1. Para o algoritmo de otimização, uma variação no acetileno ganha o mesmo peso matemático que uma variação no monóxido de carbono.

\textbf{Por que calcular os limites ESTRITAMENTE na partição de treino?}
Isso é um princípio fundamental de Engenharia de Software e Machine Learning chamado \textbf{Prevenção de Vazamento de Dados (\textit{Data Leakage})}:
\begin{itemize}[leftmargin=*,itemsep=1pt]
  \item Se calcularmos o máximo e o mínimo usando o dataset inteiro (treino + teste), o modelo já estará ``sabendo o futuro'' (ele saberá qual é o pico máximo que só ocorre no teste).
  \item Na vida real, um sistema em produção numa subestação em 2024 só conhece o histórico do passado; ele não tem como prever o valor máximo de 2025.
  \item Por isso, salvamos a régua de $(\min, \max)$ calculada \textbf{apenas nas 1.785 séries de treino}. Os dados de validação e de teste são transformados passivamente usando essa mesma régua do passado.
\end{itemize}
\end{decisaobox}

---

\section{Decisão 3: O que significa uma LSTM de 2 camadas com 64 unidades e Dropout 0,20?}

\begin{conceitobox}[Entendendo a Arquitetura em Termos de Computação]
A rede LSTM (\textit{Long Short-Term Memory}) é uma evolução das redes neurais tradicionais projetada para reter contexto ao longo do tempo através de portas lógicas internas (\textit{Gates}: Esquecer, Guardar e Passar adiante).

\begin{itemize}[leftmargin=*,itemsep=2pt]
  \item \textbf{O que são as 64 unidades?} Cada unidade é uma célula de memória com seus próprios registradores de estado. Ter 64 unidades significa que a camada mantém um vetor interno de 64 números para descrever o estado dinâmico do transformador naquele instante (se os gases estão subindo com aceleração, se estão estáveis, se houve oscilação cíclica, etc.). Sessenta e quatro é um valor equilibrado: tem capacidade de memória de sobra para 4 gases, sem sobrecarregar o processador.
  \item \textbf{Por que 2 Camadas Empilhadas (\textit{Stacked})?}
    \begin{itemize}[leftmargin=*,itemsep=1pt]
      \item \textit{1ª Camada (64 unidades, return\_sequences = True):} Funciona como um filtro digital que processa a sequência temporal passo a passo e extrai padrões de curto prazo (variações entre uma medição de 12h e a seguinte).
      \item \textit{2ª Camada (64 unidades, return\_sequences = False):} Lê a sequência de saídas da 1ª camada e consolida tudo em um único vetor resumo de 64 números, representando a \textbf{tendência macro} do equipamento nos 25 dias.
    \end{itemize}
  \item \textbf{O que é o Dropout de 0,20?} Durante o treinamento, a cada iteração, \textbf{20\% das conexões entre neurônios são desligadas aleatoriamente}. Isso impede que neurônios vizinhos combinem respostas decoradas para transformadores específicos. A rede é forçada a aprender padrões robustos e generalizáveis, evitando o temido sobreajuste (\textit{overfitting}).
  \item \textbf{Camada Densa Final (4 neurônios com ativação Linear):} Pega o vetor resumo de 64 características e projeta nas 4 saídas contínuas normalizadas correspondentes aos 4 gases no próximo instante. A ativação é linear para permitir a predição de grandezas físicas contínuas sem corte artificial.
\end{itemize}
\end{conceitobox}

---

\section{Decisão 4: Como funciona a Previsão Recursiva ($h=14$ e $h=28$)?}

\begin{decisaobox}[O Mecanismo de Malha Fechada / Feedback Loop]
Nossa rede LSTM foi treinada para prever \textbf{apenas 1 passo à frente} ($h=1$, equivalente a 12 horas). 

Como fazemos então para olhar 7 dias ($h=14$) ou 14 dias ($h=28$) no futuro?
Utilizamos a técnica de \textbf{Previsão Recursiva (\textit{Autoregressive Roll-forward})}:
\begin{enumerate}[leftmargin=*,itemsep=1pt]
  \item Injetamos as 50 medições reais observadas no buffer da rede. A rede cospe a previsão do passo $t+1$ (daqui a 12h).
  \item Pegamos essa previsão, \textbf{colocamos no final do buffer} e descartamos a medição mais antiga do início da fila (deslizando a janela).
  \item Agora o buffer contém 49 dados reais e 1 previsão gerada pela própria rede. Alimentamos a rede novamente, e ela cospe o passo $t+2$ (daqui a 24h).
  \item Repetimos esse laço recursivo:
    \begin{itemize}[leftmargin=*,itemsep=0pt]
      \item Para $h=14$ passos (7 dias / 168 horas): o laço roda \textbf{14 vezes}.
      \item Para $h=28$ passos (14 dias / 336 horas): o laço roda \textbf{28 vezes}.
    \end{itemize}
\end{enumerate}
\end{decisaobox}

---

\section{Decisão 5: Avaliação até 112 Passos e a Escolha de $h=14$ e $h=28$}

\begin{conceitobox}[O Efeito ``Telefone Sem Fio'' da Recursão e o Teste de Estresse]
Na previsão recursiva, qualquer pequeno erro no passo 1 entra como dado de entrada no passo 2. O passo 2 acumula um erro ligeiramente maior e joga no passo 3. Com o tempo, o erro se propaga como em um jogo de telefone sem fio.

Para descobrir até onde a rede se mantinha confiável antes de começar a ``alucinar'', nós submetemos o modelo a um teste de estresse em 11 horizontes futuros diferentes:
$$h \in [1, 2, 4, 7, 14, 28, 42, 56, 70, 84, 98, 112 \text{ passos}]$$

Definimos uma regra de corte de engenharia clara:
\begin{quote}
\textbf{Critério de Confiabilidade Técnica:} A previsão só é aceita se o $R^2$ médio entre os gases for \textbf{$\ge 0,80$} (isto é, se a rede ainda explicar pelo menos 80\% do comportamento real dos gases).
\end{quote}
\end{conceitobox}

\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{c c c c c c c c X}
\toprule
\textbf{Passo ($h$)} & \textbf{Tempo Físico} & \textbf{Semanas} & \textbf{$R^2$ Médio} & \textbf{$R^2$ $\text{H}_2$} & \textbf{$R^2$ $\text{CO}$} & \textbf{$R^2$ $\text{C}_2\text{H}_4$} & \textbf{$R^2$ $\text{C}_2\text{H}_2$} & \textbf{Avaliação Operacional} \\
\midrule
$h=1$ & 12 horas & 0,07 sem & \textbf{0,9996} & 0,9995 & 0,9996 & 0,9996 & 0,9996 & Precisão quase absoluta ($e < 0,5\%$) \\
$h=2$ & 24 horas (1 dia) & 0,14 sem & \textbf{0,9996} & 0,9995 & 0,9996 & 0,9996 & 0,9996 & Confiabilidade máxima \\
$h=4$ & 48 horas (2 dias) & 0,29 sem & \textbf{0,9994} & 0,9993 & 0,9994 & 0,9994 & 0,9994 & Confiabilidade máxima \\
$h=7$ & 84 horas (3,5 dias) & 0,50 sem & \textbf{0,9991} & 0,9991 & 0,9992 & 0,9992 & 0,9991 & Confiabilidade máxima \\
\textbf{$h=14$} & \textbf{7 dias (1 sem)} & \textbf{1,00 sem} & \textbf{0,9979} & 0,9976 & 0,9981 & 0,9980 & 0,9978 & \textbf{[ADOTADO NO XGBOOST] (Erro 2\%)} \\
\textbf{$h=28$} & \textbf{14 dias (2 sem)} & \textbf{2,00 sem} & \textbf{0,9928} & 0,9914 & 0,9939 & 0,9932 & 0,9928 & \textbf{[ADOTADO NO XGBOOST] (Erro 3,7\%)} \\
$h=42$ & 21 dias (3 sem) & 3,00 sem & \textbf{0,9836} & 0,9797 & 0,9865 & 0,9843 & 0,9839 & Excelente aderência \\
$h=56$ & 28 dias (4 sem) & 4,00 sem & \textbf{0,9684} & 0,9598 & 0,9748 & 0,9700 & 0,9691 & Excelente aderência \\
$h=70$ & 35 dias (5 sem) & 5,00 sem & \textbf{0,9350} & 0,9179 & 0,9486 & 0,9388 & 0,9346 & Boa aderência (Erro 10\%) \\
$h=84$ & 42 dias (6 sem) & 6,00 sem & \textbf{0,9163} & 0,8923 & 0,9349 & 0,9204 & 0,9177 & Degradação progressiva \\
$h=98$ & 49 dias (7 sem) & 7,00 sem & \textbf{0,8798} & 0,8458 & 0,9068 & 0,8859 & 0,8807 & Degradação progressiva \\
\textbf{$h=112$} & \textbf{56 dias (8 sem)} & \textbf{8,00 sem} & \textbf{0,8380} & 0,7877 & 0,8744 & 0,8493 & 0,8407 & \textbf{LIMITE MÁXIMO ACEITÁVEL ($R^2 \ge 0,80$)} \\
\bottomrule
\end{tabularx}
\end{table}

\begin{decisaobox}[Por que escolhemos $h=14$ e $h=28$ para o XGBoost em vez de $h=112$?]
Os números reais do projeto mostram que a rede é tão potente que ainda segura $R^2 = 83,8\%$ aos \textbf{112 passos (8 semanas / quase dois meses)}.
Contudo, selecionamos \textbf{$h=14$ (1 semana) e $h=28$ (2 semanas)} para o classificador por dois motivos pragmáticos de engenharia:
\begin{enumerate}[leftmargin=*,itemsep=1pt]
  \item \textbf{Pureza de Sinal:} Em 7 e 14 dias, o $R^2$ é de \textbf{$99,79\%$ e $99,28\%$}, com erro médio entre $2\%$ e $3,7\%$. Não queríamos poluir o classificador com projeções que já carregassem $17\%$ de desvio acumulado (como ocorre em $h=112$).
  \item \textbf{Janela Operacional da Subestação:} Uma ou duas semanas é a janela de tempo perfeita para uma concessionária de energia programar uma equipe de manutenção, fazer manobra de alívio de carga na rede e abrir o transformador com total segurança sem causar apagão na cidade.
\end{enumerate}
\end{decisaobox}

---

\section{Guia Rápido de Respostas para a Banca Examinadora}

\begin{bancabox}[Pergunta 1: ``Por que fixar w=50 e não uma janela menor de 5 dias?'']
\textbf{Resposta Ideal:} ``Professor, em transformadores de potência, a quebra térmica do óleo e a difusão dos gases são processos de inércia física lenta. Uma janela de 5 dias ($w=10$) é curta demais e capta apenas variações diárias de carga e temperatura ambiente. Nós executamos um Grid Search entre 10, 20, 30 e 50 passos: o tamanho de 50 passos (25 dias) alcançou o menor erro quadrático e forneceu à rede o histórico necessário para distinguir uma oscilação passageira de uma tendência real de degradação.''
\end{bancabox}

\begin{bancabox}[Pergunta 2: ``Por que a normalização Min-Max foi calculada estritamente no treino?'']
\textbf{Resposta Ideal:} ``Para evitar vazamento de dados (\textit{Data Leakage}). Na operação real de uma subestação, o sistema não sabe qual será o pico máximo de gás no ano seguinte. Se utilizássemos o dataset inteiro para extrair os valores máximos, estaríamos contaminando a rede com informações do futuro. Por isso, ajustamos o escalonador unicamente sobre as séries de treino e aplicamos essa mesma régua fixa nas partições de teste.''
\end{bancabox}

\begin{bancabox}[Pergunta 3: ``Por que usar uma Stacked LSTM com Dropout em vez de uma rede mais profunda?'']
\textbf{Resposta Ideal:} ``Por parcimônia arquitetural e controle de variância. Nosso problema possui 4 variáveis de entrada em 50 passos temporais. Duas camadas de 64 unidades formam a profundidade ideal: a primeira extrai a dinâmica sequencial de curto prazo e a segunda consolida o vetor de contexto global do equipamento. Adicionar mais camadas causaria sobreajuste (\textit{overfitting}) e lentidão sem ganho preditivo. O Dropout de 0,20 desliga 20\% das vias no treino, impedindo a memorização de máquinas específicas.''
\end{bancabox}

\begin{bancabox}[Pergunta 4: ``Se a rede aguentou prever até 112 passos, por que vocês só passaram h=14 e h=28 para o XGBoost?'']
\textbf{Resposta Ideal:} ``Nosso teste de estresse provou que a LSTM mantém $R^2 \ge 80\%$ até 112 passos (8 semanas à frente), demonstrando a alta capacidade da rede recorrente. Porém, para alimentar o classificador, selecionamos 7 e 14 dias ($h=14$ e $28$) porque nesses horizontes o $R^2$ ainda é cirúrgico ($> 99\%$, com erro de apenas 2\% a 3,7\%), evitando a injeção de ruído no XGBoost. Além disso, 7 a 14 dias é o tempo operacional padrão para uma concessionária planejar a manutenção preventiva de campo.''
\end{bancabox}

\vspace{8pt}
\noindent\makebox[\linewidth]{\rule{\textwidth}{0.8pt}}
\begin{center}
  \footnotesize\textsf{\color{NavyBlue}\textbf{DOCUMENTO DE APOIO E PREPARAÇÃO PARA A DEFESA DE TCC -- ENGENHARIA DA COMPUTAÇÃO}}
\end{center}

\end{document}
'''

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(tex_content)

print(f"Arquivo TeX gerado em: {tex_path}")
