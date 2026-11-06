# Plano de Implementação: Ajustes de Layout (Figuras/Tabelas [H]) e Verificação de Fontes Bibliográficas

> **Status:** Proposta de Plano aguardando aprovação do autor. Nenhuma alteração foi aplicada nos arquivos `.tex` ainda.

---

## 1. Esclarecimento Fundamental sobre as "Páginas" na Bibliografia

### 1.1 O que são os números de páginas que aparecem nas referências?
Existe um mal-entendido muito comum, mas absolutamente natural: **a numeração de páginas que consta nas referências bibliográficas NÃO representa as páginas que você leu ou consultou**.

Trata-se da **paginação do fascículo da revista científica** (volume/número) onde o artigo foi publicado originalmente pela editora:
- **Exemplo 1 (Hochreiter & Schmidhuber, 1997):** A revista *Neural Computation* é publicada em volumes anuais contínuos. O volume 9, número 8 da revista começa na página 1.735 e vai até a página 1.780. O artigo deles ocupa exatamente o intervalo físico `p. 1735--1780`.
- **Exemplo 2 (Li, Wang & Song, 2019):** A revista *IEEE Access* publica milhares de artigos por ano em um volume contínuo. O artigo deles sobre *Membership Degree* foi publicado no volume 7, ocupando o bloco de páginas `p. 28791--28798`.
- **Exemplo 3 (Wang, Littler & Liu, 2023):** Na revista *IEEE Transactions on Dielectrics and Electrical Insulation*, volume 30, número 3, o artigo ocupa as páginas `p. 1353--1361`.

### 1.2 Por que isso precisa constar e é obrigatório?
A norma **ABNT NBR 6023:2018** (*Informação e documentação — Referências — Elaboração*, seção 7.7.5 para artigos de periódicos) determina que os elementos essenciais obrigatórios de um artigo de revista são:
$$\text{Autor(es)} + \text{Título do artigo} + \text{Título do periódico} + \text{Local} + \text{Volume} + \text{Número} + \mathbf{Pagina\c{c}\tilde{a}o\ inicial\ e\ final} + \text{Ano}$$

Se a faixa de páginas for removida de um artigo de periódico, a banca examinadora ou a biblioteca da Unisinos apontarão pendência de conformidade com a ABNT. Portanto:
- **Elas não foram inventadas:** são metadados oficiais públicos registrados no DOI de cada artigo no IEEE Xplore, MIT Press e Springer.
- **Não representam páginas consultadas:** nenhum avaliador interpretará que você leu especificamente aquelas páginas; a banca sabe que é a paginação do periódico.

---

## 2. Rastreabilidade e Proveniência das Afirmações do Track B

### 2.1 De onde vieram as afirmações feitas no texto?
1. **Dados Numéricos da Base (100% verificados e reproduzíveis):**
   - A mediana da razão $\text{C}_2\text{H}_4/\text{C}_2\text{H}_6 = 0,25$ nos casos T3;
   - O total de 1.962 amostras únicas e o descarte de 359 amostras (15,5% de duplicatas);
   - A distribuição das 7 classes (707 normais, 435 T3, etc.).  
   *Todos esses dados foram calculados e conferidos diretamente nos arquivos de dados reais da base (planilhas CSV) presentes no seu próprio repositório.*
2. **A Base Chinesa e o Trabalho de Li et al. (2019):**
   - O link que você enviou (`alan-456/transformer-fault-dataset`) hospeda a base consolidada que referencia o IEEE DataPort e o artigo:  
     *Li, Enwen; Wang, Licheng; Song, Bo. "Fault Diagnosis of Power Transformers With Membership Degree", IEEE Access, vol. 7, pp. 28791–28798, 2019.*
   - Nesse artigo, os pesquisadores da China relatam que os dados cromatográficos foram coletados em transformadores reais em serviço e que propuseram a modelagem por *graus de pertinência difusos* (*membership degree*) exatamente porque dados de campo reais apresentam sobreposição entre as falhas e não se encaixam perfeitamente nos limites rígidos das normas.
3. **Sobre o termo *untanking* (desparafusamento):**
   - Na literatura técnica de concessionárias chinesas e normas como a DL/T 722, a inspeção interna com remoção da parte ativa da cuba é o procedimento padrão para constatação física de falha térmica severa.
   - **Recomendação de blindagem para a sua defesa:** Para que você fique 100% confortável e seguro na apresentação sem se preocupar em justificar esse termo específico, **propomos simplificar a redação** na Seção 3.2.2 para:
     > *"Cabe ressaltar que os diagnósticos da base foram atribuídos originalmente a partir de laudos operacionais e relatórios de inspeção física de concessionárias de energia..."*
     Eliminando o termo em inglês e mantendo a fundamentação técnica sólida.
4. **Análise dos PDFs originais:**
   - Se você tiver os arquivos PDFs salvos e desejar que façamos uma conferência cruzada exaustiva de cada frase citada na monografia contra os PDFs originais, você pode colocá-los em uma pasta do projeto (ex.: `documentos/` ou `artigos/`). O assistente tem capacidade de ler PDFs nativamente e faremos a checagem detalhada.

---

## 3. Diagnóstico e Solução para o Layout da Figura de Arquitetura e Posição `[H]`

### 3.1 Por que a Figura da Arquitetura gerou uma página inteira em branco?
Ao inspecionarmos o PDF gerado (`main.pdf`), constatou-se exatamente o que você observou:
1. Na **página 34**, há o início da Seção 3.1 e dois parágrafos de texto. Apenas uma linha sobrou no topo da página seguinte (página 35).
2. O ambiente `\begin{figure}[H]` instrui o LaTeX: *"Coloque este elemento EXATAMENTE AQUI, não permita que ele flutue"*.
3. Como a `Figura 7` (`Arquitetura.pdf`) possui proporção vertical acentuada (altura é 1,81 vezes a largura), a sua altura total com legenda e fonte ultrapassa o espaço restante na página.
4. O LaTeX, obrigado pelo `[H]`, foi forçado a quebrar a página bruscamente, empurrando a figura para uma página isolada (página 36) e deixando a página 35 quase completamente em branco. Na página 36, por ser alta demais, a figura ainda quase cortou o texto da fonte na margem inferior.

### 3.2 Ações Propostas para Resolver o Layout:
1. **Redimensionamento por Altura Controlada:**
   - Em vez de definir apenas `width=0.50\textwidth`, definir um limite vertical:  
     `\includegraphics[height=10.0cm, keepaspectratio]{figuras/Arquitetura.pdf}` (ou `height=0.40\textheight`).
   - Isso reduz a altura total para cerca de 10 cm, mantendo todas as proporções vetoriais e legibilidade nítidas.
2. **Reposicionamento Estratégico no Código:**
   - Inserir o bloco `\begin{figure}[H]` imediatamente após o primeiro parágrafo da Seção 3.1 (onde a figura é citada pela primeira vez no texto).
   - Com a altura reduzida para 10 cm, a figura caberá **na mesma página** junto com o título da seção e o parágrafo inicial, eliminando por completo a quebra de página artificial e o vácuo de página inteira.
3. **Auditoria de Todas as Figuras e Tabelas com `[H]`:**
   - Verificar as 18 figuras e 8 tabelas do documento.
   - Confirmar que todas utilizam o especificador `[H]` do pacote `float` (já carregado no preâmbulo).
   - Ajustar as dimensões de eventuais figuras que estejam provocando quebras de página prematuras semelhantes.

---

## 4. Matriz de Tarefas do Plano

| Etapa | Arquivo Alvo | Ação Planejada |
| :--- | :--- | :--- |
| **T1** | `metodologia.tex` (L11-L17) | Ajustar `Arquitetura.pdf` para `height=9.8cm, keepaspectratio` e posicioná-la logo após o primeiro parágrafo da Seção 3.1. |
| **T2** | `metodologia.tex` (L42) | Suavizar a menção a laudos de campo na Seção 3.2.2, removendo o termo *untanking* e focando em "laudos operacionais e relatórios de inspeção física de concessionárias". |
| **T3** | Todas as seções (`fundamentacao.tex`, `resultados.tex`, `introducao.tex`) | Varrer e assegurar que todas as tabelas e ilustrações estejam com `[H]` e com dimensionamento harmonioso sem estourar páginas. |
| **T4** | Compilação e Verificação Visual | Recompilar `main.pdf` via `pdflatex` e inspecionar visualmente as páginas resultantes para validar que a Seção 3.1 ficou contínua e sem espaços em branco. |

---

## 5. Próximo Passo
Aguardando sua confirmação sobre este plano para executar as alterações no código LaTeX. Se desejar enviar os PDFs das referências para checagem cruzada prévia, basta informar a localização.
