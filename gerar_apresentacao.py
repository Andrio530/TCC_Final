# -*- coding: utf-8 -*-
r"""
Gerador Reproduzível da Apresentação de Defesa de TCC — Engenharia da Computação (UNISINOS)
===========================================================================================

Autor do TCC: Ândrio Gabriel Epping
Orientadora: Profa. Me. Janaína Conceição Sutil Lemos
Título: PREDIÇÃO DINÂMICA DE FALHAS EM TRANSFORMADORES: Abordagem Híbrida de Previsão
        Temporal (LSTM) e Classificação Otimizada (IGA-XGBoost)

Descrição Técnica e Nota sobre python-pptx:
-------------------------------------------
O modelo institucional fornecido (`apt_modelo-institucional_2027.pptx`) aplica todos os
34 slides de exemplo sobre o layout `Blank` (`slideLayout7.xml`), incorporando fundos gráficos
e logotipos vetoriais nativos do PowerPoint (`asvg:svgBlip` para `image3.svg` branco e
`image7.svg` azul UNISINOS) diretamente na árvore OpenXML de cada slide.
Como a API de alto nível da biblioteca `python-pptx` não oferece suporte nativo para
inserir `asvg:svgBlip` nem para clonar slides-base entre apresentações preservando relações
SVG, este script implementa um construtor OpenXML determinístico sobre `lxml.etree` e
`zipfile` (e utiliza `PIL` + Inkscape quando disponível para renderizar os SVGs do TCC em
PNG 300 DPI sem distorção). Todos os textos, cartões, diagramas, setas, tabelas e notas
do apresentador são gerados como objetos nativos 100% editáveis no PowerPoint.

Execução:
    & "C:\Program Files\Inkscape\bin\python.exe" gerar_apresentacao.py
"""

from __future__ import annotations

import copy
import os
import posixpath
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path
from typing import Any

from lxml import etree
from PIL import Image

# ---------------------------------------------------------------------------
# Caminhos e constantes do projeto
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
TEMPLATE_PPTX = ROOT_DIR / "apt_modelo-institucional_2027.pptx"
OUTPUT_PPTX = ROOT_DIR / "defesa_tcc.pptx"

FIG_DIR = ROOT_DIR / "tcc-unisinos" / "figuras"
FIG_F6_DIR = FIG_DIR / "fase6"
SCRATCH_DIR = Path(
    r"C:\Users\ResTIC55\.gemini\antigravity\brain\46b1bc26-d49d-4e75-99e1-76cb9be8eaaa\scratch"
)

EMU_PER_INCH = 914400
SLIDE_W_IN = 13.333333333333334
SLIDE_H_IN = 7.5

# Paleta institucional extraída de apt_modelo-institucional_2027.pptx
C_NAVY = "253786"        # Azul UNISINOS principal
C_BLUE = "21409A"        # Azul UNISINOS acento
C_ORANGE = "FE4900"      # Laranja destaque do modelo
C_CORAL = "FE5D21"       # Accent 5
C_TEAL = "00BAC8"        # Accent 2 (Ciano / Temporal LSTM)
C_TEAL_DARK = "0E7490"   # Ciano escuro legível em texto
C_GREEN = "00875A"       # Verde escuro legível (Saudável / Convergência)
C_GREEN_LIGHT = "E6F7F0" # Fundo verde suave
C_AMBER = "C67D00"       # Âmbar legível (Alertas / Tradicionais)
C_AMBER_LIGHT = "FFF7E6" # Fundo âmbar suave
C_RED = "C0392B"         # Vermelho técnico (Erro / Negativo / Limite)
C_RED_LIGHT = "FDEDEC"   # Fundo vermelho suave
C_DARK = "262626"        # Texto principal (dk2)
C_MUTED = "5A6577"       # Texto secundário / legendas
C_WHITE = "FFFFFF"
C_BG_LIGHT = "F5F7FB"    # Fundo cartão claro
C_BG_BLUE = "EAF0FA"     # Fundo azul suave
C_BG_TEAL = "E6F8FA"     # Fundo ciano suave
C_BG_ORANGE = "FFF1EB"   # Fundo laranja suave
C_BORDER = "CFD8E8"      # Borda sutil

FONT_MAIN = "Trebuchet MS"

# Namespaces OpenXML
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS_ASVG = "http://schemas.microsoft.com/office/drawing/2016/SVG/main"
NS_A16 = "http://schemas.microsoft.com/office/drawing/2014/main"

NSMAP_SLD = {"a": NS_A, "r": NS_R, "p": NS_P}


def in_emu(inches: float) -> int:
    return int(round(inches * EMU_PER_INCH))


def qn(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def new_guid() -> str:
    return "{" + str(uuid.uuid4()).upper() + "}"


def ensure_cadeia_causal_png() -> Path:
    """Garante PNG 300 DPI de Cadeia_causal.svg sem modificar arquivos originais."""
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    target = SCRATCH_DIR / "Cadeia_causal_300dpi.png"
    if target.exists() and target.stat().st_size > 10000:
        return target
    svg_src = FIG_DIR / "Cadeia_causal.svg"
    inkscape_exe = Path(r"C:\Program Files\Inkscape\bin\inkscape.com")
    if svg_src.exists() and inkscape_exe.exists():
        subprocess.run(
            [
                str(inkscape_exe),
                str(svg_src),
                "--export-type=png",
                "--export-dpi=300",
                f"--export-filename={target}",
            ],
            check=False,
            capture_output=True,
        )
        if target.exists() and target.stat().st_size > 10000:
            return target
    fallback = FIG_DIR / "cadeia_causal.png"
    return fallback


# ---------------------------------------------------------------------------
# Construtor de Slides OpenXML baseado em apt_modelo-institucional_2027.pptx
# ---------------------------------------------------------------------------
class OpenXMLSlide:
    """Representa um slide editável e suas relações/notas."""

    def __init__(self, slide_num: int, base_xml: bytes, base_rels_xml: bytes, notes_text: str = ""):
        self.slide_num = slide_num
        self.root = etree.fromstring(base_xml)
        self.sp_tree = self.root.find("p:cSld/p:spTree", NSMAP_SLD)
        self.rels_root = etree.fromstring(base_rels_xml)
        self.notes_text = notes_text

        # Descobre maior shape id e maior rId no slide-base
        max_id = 1
        for el in self.root.xpath(".//p:cNvPr", namespaces=NSMAP_SLD):
            try:
                max_id = max(max_id, int(el.get("id", "1")))
            except ValueError:
                pass
        self._next_id = max_id + 1

        max_rid = 0
        for rel in self.rels_root:
            rid = rel.get("Id", "")
            if rid.startswith("rId"):
                try:
                    max_rid = max(max_rid, int(rid[3:]))
                except ValueError:
                    pass
        self._next_rid = max_rid + 1
        self.extra_media: list[tuple[str, bytes]] = []

    def alloc_id(self) -> str:
        val = str(self._next_id)
        self._next_id += 1
        return val

    def alloc_rid(self) -> str:
        val = f"rId{self._next_rid}"
        self._next_rid += 1
        return val

    def remove_text_shapes(self, keep_pics: bool = True, remove_non_pic_shapes: bool = True) -> None:
        """Remove caixas de texto/retângulos de exemplo do template, mantendo fundos e logo SVG."""
        to_remove = []
        for child in list(self.sp_tree):
            tag = etree.QName(child).localname
            if tag == "sp" and remove_non_pic_shapes:
                to_remove.append(child)
            elif tag == "pic" and not keep_pics:
                to_remove.append(child)
        for child in to_remove:
            self.sp_tree.remove(child)

    def add_rel(self, rel_type: str, target: str) -> str:
        rid = self.alloc_rid()
        rel = etree.SubElement(self.rels_root, qn(NS_REL, "Relationship"))
        rel.set("Id", rid)
        rel.set("Type", rel_type)
        rel.set("Target", target)
        return rid

    def add_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str | None = None,
        line_color: str | None = None,
        line_pt: float = 1.0,
        rounded: bool = False,
        radius_adj: int = 8000,
        dash: str | None = None,
        name: str = "Box",
    ) -> etree._Element:
        sp = etree.SubElement(self.sp_tree, qn(NS_P, "sp"))
        nv = etree.SubElement(sp, qn(NS_P, "nvSpPr"))
        cNv = etree.SubElement(nv, qn(NS_P, "cNvPr"), id=self.alloc_id(), name=name)
        etree.SubElement(nv, qn(NS_P, "cNvSpPr"))
        etree.SubElement(nv, qn(NS_P, "nvPr"))

        spPr = etree.SubElement(sp, qn(NS_P, "spPr"))
        xfrm = etree.SubElement(spPr, qn(NS_A, "xfrm"))
        etree.SubElement(xfrm, qn(NS_A, "off"), x=str(in_emu(x)), y=str(in_emu(y)))
        etree.SubElement(xfrm, qn(NS_A, "ext"), cx=str(in_emu(w)), cy=str(in_emu(h)))

        prst = "roundRect" if rounded else "rect"
        geom = etree.SubElement(spPr, qn(NS_A, "prstGeom"), prst=prst)
        av = etree.SubElement(geom, qn(NS_A, "avLst"))
        if rounded:
            etree.SubElement(av, qn(NS_A, "gd"), name="adj", fmla=f"val {radius_adj}")

        if fill is not None:
            sf = etree.SubElement(spPr, qn(NS_A, "solidFill"))
            etree.SubElement(sf, qn(NS_A, "srgbClr"), val=fill)
        else:
            etree.SubElement(spPr, qn(NS_A, "noFill"))

        ln = etree.SubElement(spPr, qn(NS_A, "ln"), w=str(int(round(line_pt * 12700))))
        if line_color is not None:
            lf = etree.SubElement(ln, qn(NS_A, "solidFill"))
            etree.SubElement(lf, qn(NS_A, "srgbClr"), val=line_color)
            if dash:
                etree.SubElement(ln, qn(NS_A, "prstDash"), val=dash)
        else:
            etree.SubElement(ln, qn(NS_A, "noFill"))

        txBody = etree.SubElement(sp, qn(NS_P, "txBody"))
        etree.SubElement(txBody, qn(NS_A, "bodyPr"), rtlCol="0", anchor="ctr")
        etree.SubElement(txBody, qn(NS_A, "lstStyle"))
        p = etree.SubElement(txBody, qn(NS_A, "p"))
        etree.SubElement(p, qn(NS_A, "endParaRPr"), lang="pt-BR")
        return sp

    def add_arrow(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        direction: str = "right",
        fill: str = C_NAVY,
        name: str = "Arrow",
    ) -> etree._Element:
        prst_map = {
            "right": "rightArrow",
            "left": "leftArrow",
            "down": "downArrow",
            "up": "upArrow",
        }
        sp = etree.SubElement(self.sp_tree, qn(NS_P, "sp"))
        nv = etree.SubElement(sp, qn(NS_P, "nvSpPr"))
        etree.SubElement(nv, qn(NS_P, "cNvPr"), id=self.alloc_id(), name=name)
        etree.SubElement(nv, qn(NS_P, "cNvSpPr"))
        etree.SubElement(nv, qn(NS_P, "nvPr"))

        spPr = etree.SubElement(sp, qn(NS_P, "spPr"))
        xfrm = etree.SubElement(spPr, qn(NS_A, "xfrm"))
        etree.SubElement(xfrm, qn(NS_A, "off"), x=str(in_emu(x)), y=str(in_emu(y)))
        etree.SubElement(xfrm, qn(NS_A, "ext"), cx=str(in_emu(w)), cy=str(in_emu(h)))
        geom = etree.SubElement(spPr, qn(NS_A, "prstGeom"), prst=prst_map.get(direction, "rightArrow"))
        etree.SubElement(geom, qn(NS_A, "avLst"))
        sf = etree.SubElement(spPr, qn(NS_A, "solidFill"))
        etree.SubElement(sf, qn(NS_A, "srgbClr"), val=fill)
        ln = etree.SubElement(spPr, qn(NS_A, "ln"))
        etree.SubElement(ln, qn(NS_A, "noFill"))

        txBody = etree.SubElement(sp, qn(NS_P, "txBody"))
        etree.SubElement(txBody, qn(NS_A, "bodyPr"), rtlCol="0", anchor="ctr")
        etree.SubElement(txBody, qn(NS_A, "lstStyle"))
        p = etree.SubElement(txBody, qn(NS_A, "p"))
        etree.SubElement(p, qn(NS_A, "endParaRPr"), lang="pt-BR")
        return sp

    def add_line_arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = C_NAVY,
        width_pt: float = 1.6,
        dash: str | None = None,
        end_arrow: bool = True,
        name: str = "Connector",
    ) -> etree._Element:
        """Adiciona uma linha reta (com ou sem seta na ponta) entre (x1,y1) e (x2,y2)."""
        cxn = etree.SubElement(self.sp_tree, qn(NS_P, "cxnSp"))
        nv = etree.SubElement(cxn, qn(NS_P, "nvCxnSpPr"))
        etree.SubElement(nv, qn(NS_P, "cNvPr"), id=self.alloc_id(), name=name)
        etree.SubElement(nv, qn(NS_P, "cNvCxnSpPr"))
        etree.SubElement(nv, qn(NS_P, "nvPr"))

        spPr = etree.SubElement(cxn, qn(NS_P, "spPr"))
        x0 = min(x1, x2)
        y0 = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        xfrm_attrs: dict[str, str] = {}
        if x2 < x1:
            xfrm_attrs["flipH"] = "1"
        if y2 < y1:
            xfrm_attrs["flipV"] = "1"
        xfrm = etree.SubElement(spPr, qn(NS_A, "xfrm"), **xfrm_attrs)
        etree.SubElement(xfrm, qn(NS_A, "off"), x=str(in_emu(x0)), y=str(in_emu(y0)))
        etree.SubElement(xfrm, qn(NS_A, "ext"), cx=str(in_emu(w)), cy=str(in_emu(h)))
        geom = etree.SubElement(spPr, qn(NS_A, "prstGeom"), prst="line")
        etree.SubElement(geom, qn(NS_A, "avLst"))

        ln = etree.SubElement(spPr, qn(NS_A, "ln"), w=str(int(round(width_pt * 12700))))
        sf = etree.SubElement(ln, qn(NS_A, "solidFill"))
        etree.SubElement(sf, qn(NS_A, "srgbClr"), val=color)
        if dash:
            etree.SubElement(ln, qn(NS_A, "prstDash"), val=dash)
        if end_arrow:
            etree.SubElement(ln, qn(NS_A, "tailEnd"), type="triangle", w="med", len="med")
        return cxn

    def add_textbox(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        paragraphs: list[dict[str, Any]],
        fill: str | None = None,
        line_color: str | None = None,
        line_pt: float = 1.0,
        rounded: bool = False,
        radius_adj: int = 7000,
        valign: str = "t",
        margins: tuple[float, float, float, float] = (0.06, 0.04, 0.06, 0.04),
        name: str = "TextBox",
    ) -> etree._Element:
        """
        Adiciona caixa de texto editável (ou cartão com fundo/borda + texto).
        Cada item de `paragraphs` aceita:
          - runs: list de dicts {text, size, bold, italic, color, font}
            OU atalhos text, size, bold, italic, color, font
          - align: 'l' | 'ctr' | 'r' | 'just'
          - space_before: pt
          - space_after: pt
          - line_spacing: pct (ex.: 115 para 1.15x)
          - bullet: bool ou str (ex.: '▪')
        """
        sp = etree.SubElement(self.sp_tree, qn(NS_P, "sp"))
        nv = etree.SubElement(sp, qn(NS_P, "nvSpPr"))
        etree.SubElement(nv, qn(NS_P, "cNvPr"), id=self.alloc_id(), name=name)
        etree.SubElement(nv, qn(NS_P, "cNvSpPr"), txBox="1")
        etree.SubElement(nv, qn(NS_P, "nvPr"))

        spPr = etree.SubElement(sp, qn(NS_P, "spPr"))
        xfrm = etree.SubElement(spPr, qn(NS_A, "xfrm"))
        etree.SubElement(xfrm, qn(NS_A, "off"), x=str(in_emu(x)), y=str(in_emu(y)))
        etree.SubElement(xfrm, qn(NS_A, "ext"), cx=str(in_emu(w)), cy=str(in_emu(h)))
        prst = "roundRect" if rounded else "rect"
        geom = etree.SubElement(spPr, qn(NS_A, "prstGeom"), prst=prst)
        av = etree.SubElement(geom, qn(NS_A, "avLst"))
        if rounded:
            etree.SubElement(av, qn(NS_A, "gd"), name="adj", fmla=f"val {radius_adj}")

        if fill is not None:
            sf = etree.SubElement(spPr, qn(NS_A, "solidFill"))
            etree.SubElement(sf, qn(NS_A, "srgbClr"), val=fill)
        else:
            etree.SubElement(spPr, qn(NS_A, "noFill"))

        if line_color is not None:
            ln = etree.SubElement(spPr, qn(NS_A, "ln"), w=str(int(round(line_pt * 12700))))
            lf = etree.SubElement(ln, qn(NS_A, "solidFill"))
            etree.SubElement(lf, qn(NS_A, "srgbClr"), val=line_color)
        else:
            ln = etree.SubElement(spPr, qn(NS_A, "ln"))
            etree.SubElement(ln, qn(NS_A, "noFill"))

        txBody = etree.SubElement(sp, qn(NS_P, "txBody"))
        lIns, tIns, rIns, bIns = [str(in_emu(m)) for m in margins]
        bodyPr = etree.SubElement(
            txBody,
            qn(NS_A, "bodyPr"),
            wrap="square",
            lIns=lIns,
            tIns=tIns,
            rIns=rIns,
            bIns=bIns,
            rtlCol="0",
            anchor=valign,
        )
        etree.SubElement(bodyPr, qn(NS_A, "noAutofit"))
        etree.SubElement(txBody, qn(NS_A, "lstStyle"))

        if not paragraphs:
            paragraphs = [{"runs": [{"text": "", "size": 12}]}]

        for pinfo in paragraphs:
            p = etree.SubElement(txBody, qn(NS_A, "p"))
            align = pinfo.get("align", "l")
            pPr_attrs = {"algn": align}
            bullet = pinfo.get("bullet")
            if bullet:
                pPr_attrs["marL"] = str(in_emu(0.18))
                pPr_attrs["indent"] = str(-in_emu(0.15))
            pPr = etree.SubElement(p, qn(NS_A, "pPr"), **pPr_attrs)

            if "line_spacing" in pinfo:
                lnSpc = etree.SubElement(pPr, qn(NS_A, "lnSpc"))
                etree.SubElement(lnSpc, qn(NS_A, "spcPct"), val=str(int(pinfo["line_spacing"] * 1000)))
            if "space_before" in pinfo:
                spcBef = etree.SubElement(pPr, qn(NS_A, "spcBef"))
                etree.SubElement(spcBef, qn(NS_A, "spcPts"), val=str(int(pinfo["space_before"] * 100)))
            if "space_after" in pinfo:
                spcAft = etree.SubElement(pPr, qn(NS_A, "spcAft"))
                etree.SubElement(spcAft, qn(NS_A, "spcPts"), val=str(int(pinfo["space_after"] * 100)))

            if bullet:
                bchar = bullet if isinstance(bullet, str) else "•"
                bclr = pinfo.get("bullet_color", C_ORANGE)
                buClr = etree.SubElement(pPr, qn(NS_A, "buClr"))
                etree.SubElement(buClr, qn(NS_A, "srgbClr"), val=bclr)
                etree.SubElement(pPr, qn(NS_A, "buFont"), typeface=FONT_MAIN)
                etree.SubElement(pPr, qn(NS_A, "buChar"), char=bchar)
            else:
                etree.SubElement(pPr, qn(NS_A, "buNone"))

            runs = pinfo.get("runs")
            if runs is None:
                runs = [
                    {
                        "text": pinfo.get("text", ""),
                        "size": pinfo.get("size", 14),
                        "bold": pinfo.get("bold", False),
                        "italic": pinfo.get("italic", False),
                        "color": pinfo.get("color", C_DARK),
                        "font": pinfo.get("font", FONT_MAIN),
                    }
                ]

            last_sz = 1200
            for rinfo in runs:
                r = etree.SubElement(p, qn(NS_A, "r"))
                sz = int(round(rinfo.get("size", 14) * 100))
                last_sz = sz
                rPr_attrs = {"lang": "pt-BR", "sz": str(sz), "dirty": "0"}
                if rinfo.get("bold", False):
                    rPr_attrs["b"] = "1"
                if rinfo.get("italic", False):
                    rPr_attrs["i"] = "1"
                rPr = etree.SubElement(r, qn(NS_A, "rPr"), **rPr_attrs)
                clr = rinfo.get("color", C_DARK)
                sf = etree.SubElement(rPr, qn(NS_A, "solidFill"))
                etree.SubElement(sf, qn(NS_A, "srgbClr"), val=clr)
                fnt = rinfo.get("font", FONT_MAIN)
                etree.SubElement(rPr, qn(NS_A, "latin"), typeface=fnt)
                etree.SubElement(rPr, qn(NS_A, "ea"), typeface=fnt)
                etree.SubElement(rPr, qn(NS_A, "cs"), typeface=fnt)
                t = etree.SubElement(r, qn(NS_A, "t"))
                t.text = rinfo.get("text", "")

            etree.SubElement(p, qn(NS_A, "endParaRPr"), lang="pt-BR", sz=str(last_sz))
        return sp

    def add_card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body_paragraphs: list[dict[str, Any]],
        accent: str = C_NAVY,
        fill: str = C_BG_LIGHT,
        title_size: float = 13.5,
        header_h: float = 0.36,
        name: str = "Card",
    ) -> None:
        """Cria um cartão visual com borda sutil, barra de título colorida e conteúdo editável."""
        self.add_rect(x, y, w, h, fill=fill, line_color=C_BORDER, line_pt=1.0, rounded=True, radius_adj=3500, name=f"{name}_bg")
        self.add_textbox(
            x,
            y,
            w,
            header_h,
            [{"text": title, "size": title_size, "bold": True, "color": C_WHITE, "align": "l"}],
            fill=accent,
            line_color=accent,
            line_pt=0.5,
            rounded=False,
            valign="ctr",
            margins=(0.10, 0.02, 0.08, 0.02),
            name=f"{name}_hdr",
        )
        self.add_textbox(
            x + 0.03,
            y + header_h + 0.04,
            w - 0.06,
            h - header_h - 0.08,
            body_paragraphs,
            fill=None,
            line_color=None,
            valign="t",
            margins=(0.08, 0.03, 0.08, 0.03),
            name=f"{name}_body",
        )

    def add_table(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        headers: list[str],
        rows: list[list[str]],
        col_widths_frac: list[float] | None = None,
        header_fill: str = C_NAVY,
        font_size: float = 10.5,
        header_size: float = 10.5,
        highlight_rows: dict[int, str] | None = None,
        aligns: list[str] | None = None,
        row_height_in: float | None = None,
        name: str = "Table",
    ) -> etree._Element:
        """Cria uma tabela nativa editável do PowerPoint (p:graphicFrame -> a:tbl)."""
        highlight_rows = highlight_rows or {}
        n_cols = len(headers)
        n_rows = 1 + len(rows)

        if col_widths_frac is None:
            col_widths = [in_emu(w / n_cols)] * n_cols
        else:
            s = sum(col_widths_frac)
            col_widths = [in_emu(w * (f / s)) for f in col_widths_frac]
            col_widths[-1] += in_emu(w) - sum(col_widths)

        if row_height_in is not None:
            hdr_h = in_emu(min(row_height_in * 1.15, h * 0.25))
            body_h = in_emu(row_height_in)
        else:
            hdr_h = in_emu(min(0.42, max(0.28, h / max(n_rows, 1) * 1.12)))
            body_h = int((in_emu(h) - hdr_h) / max(len(rows), 1))

        gf = etree.SubElement(self.sp_tree, qn(NS_P, "graphicFrame"))
        nv = etree.SubElement(gf, qn(NS_P, "nvGraphicFramePr"))
        etree.SubElement(nv, qn(NS_P, "cNvPr"), id=self.alloc_id(), name=name)
        cNvGf = etree.SubElement(nv, qn(NS_P, "cNvGraphicFramePr"))
        etree.SubElement(cNvGf, qn(NS_A, "graphicFrameLocks"), noGrp="1")
        etree.SubElement(nv, qn(NS_P, "nvPr"))

        xfrm = etree.SubElement(gf, qn(NS_P, "xfrm"))
        etree.SubElement(xfrm, qn(NS_A, "off"), x=str(in_emu(x)), y=str(in_emu(y)))
        etree.SubElement(xfrm, qn(NS_A, "ext"), cx=str(in_emu(w)), cy=str(in_emu(h)))

        graphic = etree.SubElement(gf, qn(NS_A, "graphic"))
        graphicData = etree.SubElement(
            graphic,
            qn(NS_A, "graphicData"),
            uri="http://schemas.openxmlformats.org/drawingml/2006/table",
        )
        tbl = etree.SubElement(graphicData, qn(NS_A, "tbl"))
        tblPr = etree.SubElement(tbl, qn(NS_A, "tblPr"), firstRow="1", bandRow="1")
        stId = etree.SubElement(tblPr, qn(NS_A, "tableStyleId"))
        stId.text = "{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"

        tblGrid = etree.SubElement(tbl, qn(NS_A, "tblGrid"))
        for cw in col_widths:
            etree.SubElement(tblGrid, qn(NS_A, "gridCol"), w=str(cw))

        def _make_cell(
            tr: etree._Element,
            text: str,
            bg: str,
            fg: str,
            sz_pt: float,
            bold: bool,
            algn: str,
            border_clr: str = "D5DFED",
        ) -> None:
            tc = etree.SubElement(tr, qn(NS_A, "tc"))
            txBody = etree.SubElement(tc, qn(NS_A, "txBody"))
            etree.SubElement(txBody, qn(NS_A, "bodyPr"), wrap="square", anchor="ctr")
            etree.SubElement(txBody, qn(NS_A, "lstStyle"))

            lines = str(text).split("\n")
            sz_hundreds = str(int(round(sz_pt * 100)))
            for line in lines:
                p = etree.SubElement(txBody, qn(NS_A, "p"))
                etree.SubElement(p, qn(NS_A, "pPr"), algn=algn)
                r = etree.SubElement(p, qn(NS_A, "r"))
                rPr_attr = {"lang": "pt-BR", "sz": sz_hundreds, "dirty": "0"}
                if bold:
                    rPr_attr["b"] = "1"
                rPr = etree.SubElement(r, qn(NS_A, "rPr"), **rPr_attr)
                sf = etree.SubElement(rPr, qn(NS_A, "solidFill"))
                etree.SubElement(sf, qn(NS_A, "srgbClr"), val=fg)
                etree.SubElement(rPr, qn(NS_A, "latin"), typeface=FONT_MAIN)
                etree.SubElement(rPr, qn(NS_A, "ea"), typeface=FONT_MAIN)
                etree.SubElement(rPr, qn(NS_A, "cs"), typeface=FONT_MAIN)
                t = etree.SubElement(r, qn(NS_A, "t"))
                t.text = line
                etree.SubElement(p, qn(NS_A, "endParaRPr"), lang="pt-BR", sz=sz_hundreds)

            tcPr = etree.SubElement(
                tc,
                qn(NS_A, "tcPr"),
                marL="60000",
                marR="60000",
                marT="32000",
                marB="32000",
                anchor="ctr",
            )
            for side in ("lnL", "lnR", "lnT", "lnB"):
                ln = etree.SubElement(tcPr, qn(NS_A, side), w="6350", cap="flat", cmpd="sng", algn="ctr")
                sf_b = etree.SubElement(ln, qn(NS_A, "solidFill"))
                etree.SubElement(sf_b, qn(NS_A, "srgbClr"), val=border_clr)
            sf_bg = etree.SubElement(tcPr, qn(NS_A, "solidFill"))
            etree.SubElement(sf_bg, qn(NS_A, "srgbClr"), val=bg)

        # Linha de cabeçalho
        tr_h = etree.SubElement(tbl, qn(NS_A, "tr"), h=str(hdr_h))
        for j, htxt in enumerate(headers):
            al = aligns[j] if aligns and j < len(aligns) else ("l" if j == 0 else "ctr")
            _make_cell(tr_h, htxt, header_fill, C_WHITE, header_size, True, al, border_clr=header_fill)

        # Linhas de dados
        for i, row in enumerate(rows):
            tr = etree.SubElement(tbl, qn(NS_A, "tr"), h=str(body_h))
            bg_row = highlight_rows.get(i, C_WHITE if i % 2 == 0 else "F3F6FB")
            is_hi = i in highlight_rows
            for j, val in enumerate(row):
                al = aligns[j] if aligns and j < len(aligns) else ("l" if j == 0 else "ctr")
                is_bold = is_hi or (j == 0 and str(val).startswith(("C1", "C3", "Média", "Total")))
                _make_cell(tr, val, bg_row, C_DARK, font_size, is_bold, al)

        return gf

    def add_image_fit(
        self,
        img_path: Path,
        x: float,
        y: float,
        max_w: float,
        max_h: float,
        center_h: bool = True,
        center_v: bool = True,
        border_color: str | None = None,
        name: str = "Figure",
    ) -> tuple[float, float, float, float]:
        """Insere imagem mantendo a razão de aspecto exata em pixels, sem distorção."""
        if not img_path.exists():
            raise FileNotFoundError(f"Figura não encontrada: {img_path}")

        with Image.open(img_path) as im:
            px_w, px_h = im.size
        aspect = px_w / px_h

        if max_w / max_h > aspect:
            draw_h = max_h
            draw_w = max_h * aspect
        else:
            draw_w = max_w
            draw_h = max_w / aspect

        draw_x = x + (max_w - draw_w) / 2.0 if center_h else x
        draw_y = y + (max_h - draw_h) / 2.0 if center_v else y

        ext = img_path.suffix.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        media_name = f"added_s{self.slide_num}_{len(self.extra_media) + 1}.{ext}"
        self.extra_media.append((media_name, img_path.read_bytes()))
        rid = self.add_rel(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            f"../media/{media_name}",
        )

        pic = etree.SubElement(self.sp_tree, qn(NS_P, "pic"))
        nv = etree.SubElement(pic, qn(NS_P, "nvPicPr"))
        etree.SubElement(nv, qn(NS_P, "cNvPr"), id=self.alloc_id(), name=name)
        cNvPic = etree.SubElement(nv, qn(NS_P, "cNvPicPr"))
        etree.SubElement(cNvPic, qn(NS_A, "picLocks"), noChangeAspect="1")
        etree.SubElement(nv, qn(NS_P, "nvPr"))

        blipFill = etree.SubElement(pic, qn(NS_P, "blipFill"))
        etree.SubElement(blipFill, qn(NS_A, "blip"), {qn(NS_R, "embed"): rid})
        stretch = etree.SubElement(blipFill, qn(NS_A, "stretch"))
        etree.SubElement(stretch, qn(NS_A, "fillRect"))

        spPr = etree.SubElement(pic, qn(NS_P, "spPr"))
        xfrm = etree.SubElement(spPr, qn(NS_A, "xfrm"))
        etree.SubElement(xfrm, qn(NS_A, "off"), x=str(in_emu(draw_x)), y=str(in_emu(draw_y)))
        etree.SubElement(xfrm, qn(NS_A, "ext"), cx=str(in_emu(draw_w)), cy=str(in_emu(draw_h)))
        geom = etree.SubElement(spPr, qn(NS_A, "prstGeom"), prst="rect")
        etree.SubElement(geom, qn(NS_A, "avLst"))
        if border_color:
            ln = etree.SubElement(spPr, qn(NS_A, "ln"), w="9525")
            lf = etree.SubElement(ln, qn(NS_A, "solidFill"))
            etree.SubElement(lf, qn(NS_A, "srgbClr"), val=border_color)

        return draw_x, draw_y, draw_w, draw_h

    def add_header_and_footer(
        self,
        tag: str,
        title: str,
        footer_num: str,
        title_size: float = 22.0,
    ) -> None:
        """Padroniza cabeçalho institucional e rodapé sem colidir com logo ou ondas."""
        self.add_textbox(
            0.55,
            0.16,
            11.40,
            0.24,
            [{"text": tag.upper(), "size": 9.5, "bold": True, "color": C_ORANGE}],
            margins=(0.0, 0.0, 0.0, 0.0),
            name="Tag",
        )
        self.add_textbox(
            0.55,
            0.38,
            11.45,
            0.54,
            [{"text": title, "size": title_size, "bold": True, "color": C_NAVY}],
            valign="ctr",
            margins=(0.0, 0.0, 0.0, 0.0),
            name="Title",
        )
        # Barra de destaque institucional laranja + linha cinza
        self.add_rect(0.55, 0.96, 1.35, 0.038, fill=C_ORANGE, line_color=None, name="AccentBar")
        self.add_rect(1.94, 0.972, 10.84, 0.015, fill="D6DFEC", line_color=None, name="RuleBar")

        # Rodapé discreto antes da zona das ondas (x <= 11.40 in)
        self.add_textbox(
            0.55,
            7.13,
            9.20,
            0.26,
            [
                {
                    "text": "TCC II — Engenharia da Computação (UNISINOS)  |  Ândrio Gabriel Epping (2026)",
                    "size": 9.5,
                    "color": C_MUTED,
                }
            ],
            margins=(0.0, 0.0, 0.0, 0.0),
            name="FooterLeft",
        )
        self.add_textbox(
            9.80,
            7.13,
            1.65,
            0.26,
            [{"text": footer_num, "size": 9.5, "bold": True, "color": C_NAVY, "align": "r"}],
            margins=(0.0, 0.0, 0.0, 0.0),
            name="FooterRight",
        )


# ---------------------------------------------------------------------------
# Construção individual dos 23 slides
# ---------------------------------------------------------------------------
def r(text: str, size: float = 13.0, bold: bool = False, italic: bool = False, color: str = C_DARK) -> dict[str, Any]:
    return {"text": text, "size": size, "bold": bold, "italic": italic, "color": color, "font": FONT_MAIN}


def build_slides(tpl_archive: zipfile.ZipFile, cadeia_png: Path) -> list[OpenXMLSlide]:
    slides: list[OpenXMLSlide] = []

    def new_from(tpl_slide_idx: int, notes: str) -> OpenXMLSlide:
        idx = len(slides) + 1
        s_xml = tpl_archive.read(f"ppt/slides/slide{tpl_slide_idx}.xml")
        r_xml = tpl_archive.read(f"ppt/slides/_rels/slide{tpl_slide_idx}.xml.rels")
        sl = OpenXMLSlide(idx, s_xml, r_xml, notes)
        sl.remove_text_shapes(keep_pics=True, remove_non_pic_shapes=True)
        return sl

    # =========================================================================
    # SLIDE 1 — CAPA (20s) — Base: template slide2.xml
    # =========================================================================
    s1 = new_from(
        2,
        notes="""[OBJETIVO] Apresentar formalmente o trabalho, o autor, a orientadora e o contexto institucional.
[TEMPO PREVISTO] 20 segundos (Acumulado: 0:20).
[MENSAGEM CENTRAL] Investigação científica e rigorosa sobre previsão temporal e classificação explicável para diagnóstico de transformadores via DGA.
[FALA SUGERIDA] "Bom dia/boa noite aos membros da banca examinadora e à professora orientadora Janaína Lemos. Sou Ândrio Gabriel Epping, formando em Engenharia da Computação pela UNISINOS, e apresento meu Trabalho de Conclusão de Curso intitulado Predição Dinâmica de Falhas em Transformadores: Abordagem Híbrida de Previsão Temporal (LSTM) e Classificação Otimizada (IGA-XGBoost)."
[TRANSIÇÃO] "Para iniciar, apresento por que o monitoramento de transformadores por gases dissolvidos é um problema estratégico."
[PERGUNTAS POSSÍVEIS] Nenhuma específica neste slide.""",
    )
    s1.add_textbox(
        0.92,
        0.55,
        9.0,
        0.40,
        [
            r(
                "UNIVERSIDADE DO VALE DO RIO DOS SINOS — UNISINOS  |  ENGENHARIA DA COMPUTAÇÃO",
                size=11.5,
                bold=True,
                color="D6E2FF",
            )
        ],
        margins=(0.0, 0.0, 0.0, 0.0),
        name="InstHeader",
    )
    s1.add_textbox(
        0.92,
        2.15,
        9.35,
        2.45,
        [
            {"text": "PREDIÇÃO DINÂMICA DE FALHAS\nEM TRANSFORMADORES", "size": 34.0, "bold": True, "color": C_WHITE, "line_spacing": 105},
            {
                "text": "Abordagem Híbrida de Previsão Temporal (LSTM) e Classificação Otimizada (IGA-XGBoost)",
                "size": 18.5,
                "bold": False,
                "color": "E8EEFF",
                "space_before": 12,
                "line_spacing": 115,
            },
        ],
        valign="b",
        margins=(0.0, 0.0, 0.0, 0.0),
        name="CoverTitle",
    )
    # Barra institucional laranja (#FE4900) do slide 2 do modelo
    s1.add_rect(0.92, 4.78, 2.15, 0.065, fill=C_ORANGE, line_color=None, name="CoverOrangeBar")
    s1.add_textbox(
        0.92,
        5.05,
        8.60,
        1.55,
        [
            {"runs": [r("Autor: ", 15.0, True, False, "FFE3D6"), r("Ândrio Gabriel Epping", 15.0, True, False, C_WHITE)], "space_after": 4},
            {"runs": [r("Orientadora: ", 14.5, True, False, "FFE3D6"), r("Profa. Me. Janaína Conceição Sutil Lemos", 14.5, False, False, C_WHITE)], "space_after": 4},
            {"runs": [r("Banca de Defesa de TCC II  |  São Leopoldo, RS — 2026", 13.0, False, False, "CBD9FA")]},
        ],
        valign="t",
        margins=(0.0, 0.0, 0.0, 0.0),
        name="CoverMeta",
    )
    slides.append(s1)

    # =========================================================================
    # SLIDE 2 — CONTEXTUALIZAÇÃO: TRANSFORMADORES E DGA (1min 20s)
    # =========================================================================
    s2 = new_from(
        13,
        notes="""[OBJETIVO] Situar a relevância física e operacional dos transformadores de potência e da Análise de Gases Dissolvidos (DGA).
[TEMPO PREVISTO] 1 minuto e 20 segundos (Acumulado: 1:40).
[MENSAGEM CENTRAL] Estresses térmicos e elétricos degradam o óleo mineral e o papel celulósico, gerando assinatura gasosa proporcional ao mecanismo e à severidade da falha.
[FALA SUGERIDA] "Transformadores de potência são ativos críticos de transmissão e distribuição. Sob estresse térmico, elétrico e envelhecimento contínuo, o óleo mineral e o papel Kraft sofrem decomposição molecular, liberando gases que se dissolvem no óleo. A Análise de Gases Dissolvidos, ou DGA, funciona como um exame de sangue do equipamento. Como vemos na cadeia causal à esquerda, descargas parciais geram hidrogênio e metano; sobreaquecimentos evoluem de metano e etano para etileno em temperaturas acima de 700 °C; e arcos elétricos acima de 1.000 °C produzem acetileno. Além disso, o acompanhamento contínuo via sensores online motivou investigar se projeções temporais contribuem para antecipar o diagnóstico em relação às regras normativas reativas."
[TRANSIÇÃO] "Contudo, transformar essas leituras de DGA em diagnósticos confiáveis esbarra em limitações práticas importantes, que definem o nosso problema de pesquisa."
[PERGUNTAS POSSÍVEIS] "Quais gases são gerados pelo papel celulósico vs. óleo mineral?" -> CO e CO2 vêm principalmente da celulose; os hidrocarbonetos (CH4, C2H6, C2H4, C2H2) e H2 vêm da decomposição do óleo mineral isolante.""",
    )
    s2.add_header_and_footer("Contextualização", "Transformadores de Potência e Análise de Gases Dissolvidos (DGA)", "02 / 15")
    s2.add_image_fit(cadeia_png, 0.55, 1.15, 7.25, 5.75, border_color=C_BORDER, name="FigCadeiaCausal")

    s2.add_card(
        8.00,
        1.15,
        4.75,
        1.75,
        "1. Ativos Críticos e Degradação do Isolamento",
        [
            {"runs": [r("• Sistema óleo-papel: ", 12.5, True, False, C_NAVY), r("óleo mineral (dielétrico/resfriamento) e papel Kraft/pressboard.", 12.5)], "space_after": 4},
            {"runs": [r("• Estresses termoelétricos ", 12.5, True, False, C_NAVY), r("quebram ligações químicas e comprometem a rigidez dielétrica.", 12.5)]},
        ],
        accent=C_NAVY,
        fill=C_BG_LIGHT,
        name="CardCtx1",
    )
    s2.add_card(
        8.00,
        3.05,
        4.75,
        2.15,
        "2. Assinatura Físico-Química dos Gases (DGA)",
        [
            {"runs": [r("• Descargas Parciais (PD): ", 12.0, True, False, C_BLUE), r("predomínio de H₂ e CH₄.", 12.0)], "space_after": 4},
            {"runs": [r("• Falhas Térmicas (T1–T3): ", 12.0, True, False, C_AMBER), r("CH₄, C₂H₆ (<300 °C) evoluindo para C₂H₄ (>700 °C); CO/CO₂ do papel.", 12.0)], "space_after": 4},
            {"runs": [r("• Arcos Elétricos (D1/D2): ", 12.0, True, False, C_RED), r("picos de C₂H₂ acima de 1.000 °C.", 12.0)]},
        ],
        accent=C_ORANGE,
        fill=C_BG_ORANGE,
        name="CardCtx2",
    )
    s2.add_card(
        8.00,
        5.35,
        4.75,
        1.55,
        "3. Por Que Modelar a Evolução Temporal?",
        [
            {"runs": [r("• Sensores online ", 12.0, True, False, C_TEAL_DARK), r("registram séries contínuas além de coletas laboratoriais isoladas.", 12.0)], "space_after": 3},
            {"runs": [r("• Hipótese motivadora: ", 12.0, True, False, C_TEAL_DARK), r("projetar concentrações futuras auxiliaria a detecção precoce?", 12.0)]},
        ],
        accent=C_TEAL_DARK,
        fill=C_BG_TEAL,
        name="CardCtx3",
    )
    slides.append(s2)

    # =========================================================================
    # SLIDE 3 — PROBLEMA DE PESQUISA E MOTIVAÇÃO (1min 20s)
    # =========================================================================
    s3 = new_from(
        13,
        notes="""[OBJETIVO] Definir com clareza as lacunas científicas da área e a pergunta central que o TCC responde.
[TEMPO PREVISTO] 1 minuto e 20 segundos (Acumulado: 3:00).
[MENSAGEM CENTRAL] Normas clássicas têm fronteiras rígidas e não contemplam a classe Normal; estudos recentes de IA ou carecem de ablação temporal controlada ou apresentam métricas infladas em bases pequenas/não auditadas.
[FALA SUGERIDA] "Ao revisar as abordagens existentes, identificamos três lacunas objetivas. Primeiro, métodos normativos como o Triângulo de Duval e as razões da IEC 60599 utilizam limites fixos, não possuem categoria para operação Normal e geram alta taxa de laudos Não Determinados. Segundo, estudos de aprendizado de máquina estático frequentemente reportam acurácias acima de 99% usando amostras minúsculas — como 25 casos de teste em Wu et al. (2021) — ou sem auditar registros duplicados, além de omitirem intervalos de confiança estatísticos. Terceiro, abordagens temporais com LSTM, como Wang et al. (2023), avaliam a regressão isoladamente sem testar se as projeções futuras realmente melhoram o classificador. Isso define nossa pergunta científica: projeções futuras da LSTM, seleção evolutiva por IGA e interpretabilidade via TreeSHAP agregam ganho mensurável e coerência física em bases industriais auditadas?"
[TRANSIÇÃO] "A partir dessa pergunta científica, estruturamos o objetivo geral e os cinco objetivos específicos do trabalho."
[PERGUNTAS POSSÍVEIS] "Por que o Triângulo de Duval não detecta o estado Normal?" -> Ele parte da premissa normativa de que as concentrações já excederam os limites típicos de atenção (IEC 60599), projetando todo ponto sobre 6 regiões de falha ou zona mista DT.""",
    )
    s3.add_header_and_footer("Problema de Pesquisa", "Limitações Atuais e Pergunta Científica Investigada", "03 / 15")

    s3.add_card(
        0.55,
        1.18,
        3.90,
        3.40,
        "Normas Clássicas (IEC 60599 / Duval)",
        [
            {"runs": [r("• Sem classe Normal: ", 12.5, True, False, C_RED), r("toda amostra projetada no Triângulo recebe rótulo de defeito.", 12.5)], "space_after": 6},
            {"runs": [r("• Laudos inconclusivos: ", 12.5, True, False, C_RED), r("combinações fora da tabela IEC resultam em Não Determinado (ND).", 12.5)], "space_after": 6},
            {"runs": [r("• Fronteiras rígidas: ", 12.5, True, False, C_RED), r("alta sensibilidade a ruídos em zonas limítrofes e visão estática.", 12.5)]},
        ],
        accent=C_RED,
        fill=C_RED_LIGHT,
        name="CardGap1",
    )
    s3.add_card(
        4.70,
        1.18,
        3.90,
        3.40,
        "Literatura de ML Estático",
        [
            {"runs": [r("• Amostras reduzidas: ", 12.5, True, False, C_AMBER), r("ex.: Wu et al. (2021) testaram apenas 25 amostras (>99% acc).", 12.5)], "space_after": 6},
            {"runs": [r("• Risco de duplicatas: ", 12.5, True, False, C_AMBER), r("bases compiladas possuem réplicas que vazam entre treino e teste.", 12.5)], "space_after": 6},
            {"runs": [r("• Sem rigor estatístico: ", 12.5, True, False, C_AMBER), r("ausência de intervalos de confiança pareados e caixa-preta.", 12.5)]},
        ],
        accent=C_AMBER,
        fill=C_AMBER_LIGHT,
        name="CardGap2",
    )
    s3.add_card(
        8.85,
        1.18,
        3.90,
        3.40,
        "Modelos Temporais em DGA",
        [
            {"runs": [r("• Avaliação desacoplada: ", 12.5, True, False, C_TEAL_DARK), r("Wang et al. (2023) não testam ablação no classificador.", 12.5)], "space_after": 6},
            {"runs": [r("• Bases fragmentadas: ", 12.5, True, False, C_TEAL_DARK), r("nenhuma base aberta reúne séries longas + 5 gases + 7 classes IEC.", 12.5)], "space_after": 6},
            {"runs": [r("• Lacuna de validação: ", 12.5, True, False, C_TEAL_DARK), r("falta evidência empírica se o futuro previsto agrega ao snapshot atual.", 12.5)]},
        ],
        accent=C_TEAL_DARK,
        fill=C_BG_TEAL,
        name="CardGap3",
    )

    # Caixa de destaque da Pergunta Científica
    s3.add_rect(0.55, 4.82, 12.20, 2.08, fill=C_NAVY, line_color=C_ORANGE, line_pt=2.2, rounded=True, radius_adj=3000, name="ResearchQuestionBox")
    s3.add_textbox(
        0.75,
        4.94,
        11.80,
        1.85,
        [
            {"text": "PERGUNTA CIENTÍFICA CENTRAL DO TCC", "size": 12.0, "bold": True, "color": "FFB899", "space_after": 6},
            {
                "text": "Em bases reais auditadas de DGA, o acoplamento de previsões temporais (LSTM) e a otimização evolutiva (IGA) acoplados ao XGBoost agregam ganho estatisticamente comprovado ao diagnóstico em comparação a modelos estáticos e regras normativas, mantendo coerência físico-química verificável via TreeSHAP?",
                "size": 15.5,
                "bold": True,
                "color": C_WHITE,
                "line_spacing": 120,
            },
        ],
        valign="ctr",
        margins=(0.05, 0.02, 0.05, 0.02),
        name="ResearchQuestionText",
    )
    slides.append(s3)

    # =========================================================================
    # SLIDE 4 — OBJETIVO E CONTRIBUIÇÕES (1min 00s)
    # =========================================================================
    s4 = new_from(
        13,
        notes="""[OBJETIVO] Apresentar o objetivo geral fiel ao documento, os objetivos específicos e separar claramente proposta, implementação e evidência empírica.
[TEMPO PREVISTO] 1 minuto (Acumulado: 4:00).
[MENSAGEM CENTRAL] O trabalho separa de forma transparente o que foi proposto como arquitetura híbrida do que os experimentos rigorosos efetivamente demonstraram — incluindo um resultado negativo relevante.
[FALA SUGERIDA] "O objetivo geral deste TCC é desenvolver e avaliar uma metodologia computacional para diagnóstico em transformadores via DGA, investigando a contribuição de previsões temporais por LSTM, a simplificação dimensional via IGA-XGBoost e a coerência físico-química via TreeSHAP frente às normas do setor elétrico. Nos cinco objetivos específicos, destacam-se a auditoria de dados em dois eixos (Track A e Track B), o módulo LSTM multi-horizonte, o teste pareado por bootstrap, a otimização conjunta IGA e a auditoria via TreeSHAP. Na tabela à direita, diferenciamos honestamente o que foi proposto e implementado do que foi comprovado: não afirmamos ganho da LSTM no diagnóstico pontual — os dados mostraram redundância temporal —, mas demonstramos redução de 53% dos atributos no Track A, equivalência com o estado da arte em base saneada, superioridade sobre normas no Track B e recuperação autônoma de fronteiras de Duval pelo TreeSHAP."
[TRANSIÇÃO] "A seguir, detalhamos como essa metodologia foi materializada na arquitetura geral do sistema."
[PERGUNTAS POSSÍVEIS] "Por que estruturar em dois eixos em vez de rodar a LSTM na base de 7 classes?" -> Porque a base de 7 classes (1.962 amostras) é estática (sem timestamps/séries), enquanto a base temporal de Katser tem 4 gases e 4 classes constantes.""",
    )
    s4.add_header_and_footer("Objetivos e Escopo", "Objetivo Geral, Específicos e Síntese das Contribuições", "04 / 15")

    s4.add_textbox(
        0.55,
        1.14,
        12.20,
        0.92,
        [
            {"runs": [r("Objetivo Geral: ", 13.5, True, False, "FFD166"), r("Desenvolver e avaliar uma metodologia computacional baseada em DGA, investigando a contribuição de previsões temporais (LSTM), a simplificação de atributos por IGA-XGBoost e a coerência físico-química dos padrões aprendidos frente à IEC 60599 e ao Triângulo de Duval.", 13.0, False, False, C_WHITE)], "line_spacing": 115},
        ],
        fill=C_NAVY,
        line_color=C_NAVY,
        rounded=True,
        radius_adj=2500,
        valign="ctr",
        margins=(0.14, 0.06, 0.14, 0.06),
        name="ObjGeralBox",
    )

    s4.add_card(
        0.55,
        2.18,
        5.55,
        4.75,
        "Objetivos Específicos (Seção 1.3)",
        [
            {"bullet": "1", "bullet_color": C_ORANGE, "runs": [r("Auditar e preparar bases ", 12.5, True, False, C_NAVY), r("removendo duplicatas e estruturando os eixos Track A (temporal) e Track B (multiclasse IEC).", 12.0)], "space_after": 7},
            {"bullet": "2", "bullet_color": C_ORANGE, "runs": [r("Implementar módulo LSTM ", 12.5, True, False, C_NAVY), r("para projeção multivariada de concentrações gasosas em múltiplos horizontes futuros.", 12.0)], "space_after": 7},
            {"bullet": "3", "bullet_color": C_ORANGE, "runs": [r("Testar hipótese híbrida via bootstrap ", 12.5, True, False, C_NAVY), r("verificando se projeções LSTM melhoram o F1-macro do XGBoost no diagnóstico pontual.", 12.0)], "space_after": 7},
            {"bullet": "4", "bullet_color": C_ORANGE, "runs": [r("Avaliar simplificação pelo IGA ", 12.5, True, False, C_NAVY), r("na seleção conjunta de variáveis e hiperparâmetros sem perda diagnóstica.", 12.0)], "space_after": 7},
            {"bullet": "5", "bullet_color": C_ORANGE, "runs": [r("Auditar física das decisões (TreeSHAP) ", 12.5, True, False, C_NAVY), r("confrontando limiares aprendidos com IEC 60599 e Duval.", 12.0)]},
        ],
        accent=C_BLUE,
        fill=C_BG_LIGHT,
        name="CardObjEsp",
    )

    s4.add_table(
        6.30,
        2.18,
        6.45,
        4.75,
        ["Proposto e Implementado", "Demonstrado Empiricamente"],
        [
            ["Auditoria anti-vazamento e CustomSMOTE termodinâmico", "Remoção de 15,5% de réplicas/conflitos (2.321 -> 1.962) evitou vazamento."],
            ["Stacked LSTM recursiva (w=50; h=14, 28) acoplada ao XGBoost", "Alta precisão regressiva (R²>0,99), porém sem ganho no diagnóstico pontual (ΔF1=+0,0056, p=0,238)."],
            ["Otimização conjunta IGA-XGBoost (máscara + 9 hiperparâmetros)", "Redução de 53% dos atributos no Track A (17 -> 8) mantendo F1 ≈ 0,891; equivalência no Track B."],
            ["Benchmark multiclasse (7 e 6 classes) com 5.000 bootstraps", "Supera Duval e IEC (F1=0,800 vs. 0,450/0,330); revela teto empírico (0,76–0,80)."],
            ["Protocolo XAI via TreeSHAP (log-odds, interações, estabilidade)", "Alta estabilidade (ρ=0,954); recupera corte de 23% de etileno de Duval e explica erros."],
        ],
        col_widths_frac=[0.44, 0.56],
        header_fill=C_NAVY,
        font_size=11.0,
        header_size=11.5,
        highlight_rows={1: C_AMBER_LIGHT, 4: C_GREEN_LIGHT},
        aligns=["l", "l"],
        name="TableContrib",
    )
    slides.append(s4)

    # =========================================================================
    # SLIDE 5 — ARQUITETURA GERAL PROPOSTA (1min 30s)
    # =========================================================================
    s5 = new_from(
        13,
        notes="""[OBJETIVO] Reconstruir visualmente a arquitetura real implementada no código, distinguindo entrada, previsão, classificação, saída e baselines experimentais.
[TEMPO PREVISTO] 1 minuto e 30 segundos (Acumulado: 5:30).
[MENSAGEM CENTRAL] O sistema organiza dois eixos rigorosamente isolados de vazamento de dados, permitindo testar tanto o acoplamento temporal LSTM → IGA-XGBoost (Track A) quanto o benchmark multiclasse IEC + XAI (Track B).
[FALA SUGERIDA] "Este diagrama reconstrói o pipeline real implementado no código-fonte. Na faixa superior azul temos o Eixo Track A: 3.000 séries temporais de 4 gases passam por ScalerPerGas ajustado apenas no treino e janelamento walk-forward de 50 passos. A rede Stacked LSTM prevê concentrações futuras em h=14 e h=28 passos (8 atributos). Esses valores juntam-se aos 4 gases observados em t=420 e a 5 razões adaptadas, formando 17 candidatos balanceados via sample_weight para o IGA-XGBoost (C1) e para os controles de ablação (C2, C5a, C5b e C6). Na faixa inferior verde temos o Eixo Track B: 2.321 registros são deduplicados para 1.962 amostras únicas em ppm de 5 gases. Calculamos 14 variáveis físico-químicas antes da transformação log(x+1) e Min-Max, somando 19 candidatos. Dentro de cada dobra de CV do IGA roda o ImbPipeline com CustomSMOTE, gerando o modelo C3 (16 atributos) confrontado com C4, Wu et al. (M3), Duval (M2) e IEC 60599 (M1). Ambos os eixos convergem à direita para bootstrap pareado e TreeSHAP."
[TRANSIÇÃO] "Vamos examinar em detalhe o tratamento das duas bases de dados e as garantias contra vazamento de informação."
[PERGUNTAS POSSÍVEIS] "Por que os dois tracks não compartilham o mesmo vetor de atributos?" -> Porque Katser (Track A) não mede CH4 e C2H6, enquanto a base estática (Track B) mede os 5 hidrocarbonetos/H2 e não mede CO nem séries temporais.""",
    )
    s5.add_header_and_footer("Arquitetura Implementada", "Arquitetura Geral do Sistema Híbrido e Fluxos de Referência", "05 / 15")

    # Legenda de cores no topo
    leg_items = [
        ("Entrada / Dados", C_BLUE, C_BG_BLUE, 1.56),
        ("Previsão LSTM", C_TEAL_DARK, C_BG_TEAL, 1.48),
        ("Eng. Físico-Química", C_AMBER, C_AMBER_LIGHT, 1.88),
        ("Classificação IGA-XGBoost", C_GREEN, C_GREEN_LIGHT, 2.38),
        ("Baselines / Ablação", C_MUTED, C_BG_LIGHT, 1.82),
        ("Saída / Bootstrap + TreeSHAP", C_NAVY, "E8ECF8", 2.70),
    ]
    lx = 0.55
    for lbl, bclr, fclr, lw in leg_items:
        s5.add_textbox(
            lx,
            1.08,
            lw,
            0.27,
            [{"text": lbl, "size": 9.0, "bold": True, "color": bclr, "align": "ctr"}],
            fill=fclr,
            line_color=bclr,
            line_pt=1.0,
            rounded=True,
            valign="ctr",
            margins=(0.02, 0.0, 0.02, 0.0),
        )
        lx += lw + 0.08

    # Faixa Track A (Superior)
    s5.add_rect(0.55, 1.44, 12.22, 2.66, fill="F8FAFE", line_color=C_BLUE, line_pt=1.2, rounded=True, radius_adj=2500, name="TrackAContainer")
    s5.add_textbox(0.65, 1.48, 3.80, 0.24, [{"text": "TRACK A — EIXO TEMPORAL (Base Katser et al., 2022)", "size": 10.0, "bold": True, "color": C_BLUE}], margins=(0.0, 0.0, 0.0, 0.0))

    # Blocos Track A
    s5.add_textbox(
        0.68, 1.78, 2.05, 1.38,
        [
            {"text": "ENTRADA (3.000 séries)", "size": 9.5, "bold": True, "color": C_WHITE, "align": "ctr", "space_after": 3},
            {"text": "4 gases: H₂, CO, C₂H₄, C₂H₂\n420 passos (Δt=12h)\nSplit: 1785 / 315 / 900\nScalerPerGas (só treino)", "size": 9.5, "color": C_WHITE, "align": "ctr"},
        ],
        fill=C_BLUE, line_color=C_NAVY, rounded=True, valign="ctr", margins=(0.04, 0.03, 0.04, 0.03),
    )
    s5.add_line_arrow(2.74, 2.47, 2.98, 2.47, color=C_TEAL_DARK, width_pt=2.0)

    s5.add_textbox(
        3.00, 1.78, 2.22, 1.38,
        [
            {"text": "PREVISÃO LSTM (Estágio 1)", "size": 9.5, "bold": True, "color": C_WHITE, "align": "ctr", "space_after": 3},
            {"text": "Janela w*=50 (371..420)\n2×LSTM(64) + Dropout(0,2)\nDense(4) -> loop recursivo\nh=14 (7d) e h=28 (14d) -> 8 vars", "size": 9.5, "color": C_WHITE, "align": "ctr"},
        ],
        fill=C_TEAL_DARK, line_color=C_TEAL_DARK, rounded=True, valign="ctr", margins=(0.04, 0.03, 0.04, 0.03),
    )
    s5.add_line_arrow(5.23, 2.47, 5.47, 2.47, color=C_AMBER, width_pt=2.0)

    s5.add_textbox(
        5.49, 1.78, 2.18, 1.38,
        [
            {"text": "ATRIBUTOS TRACK A", "size": 9.5, "bold": True, "color": C_DARK, "align": "ctr", "space_after": 3},
            {"text": "4 obs. (t=420) + 5 razões/TH\n+ 8 previsões LSTM\n= 17 candidatos\n+ sample_weight analítico", "size": 9.5, "color": C_DARK, "align": "ctr"},
        ],
        fill=C_AMBER_LIGHT, line_color=C_AMBER, line_pt=1.3, rounded=True, valign="ctr", margins=(0.04, 0.03, 0.04, 0.03),
    )
    s5.add_line_arrow(7.68, 2.47, 7.92, 2.47, color=C_GREEN, width_pt=2.0)

    s5.add_textbox(
        7.94, 1.78, 2.22, 1.38,
        [
            {"text": "IGA-XGBOOST (C1)", "size": 9.5, "bold": True, "color": C_WHITE, "align": "ctr", "space_after": 3},
            {"text": "Cromossomo (17 bin + 9 cont.)\nFitness: F1-macro (CV 5-fold)\nSeleciona 8 de 17 atrib.\n4 classes: Normal, PD, D1, T1", "size": 9.5, "color": C_WHITE, "align": "ctr"},
        ],
        fill=C_GREEN, line_color=C_GREEN, rounded=True, valign="ctr", margins=(0.04, 0.03, 0.04, 0.03),
    )
    s5.add_line_arrow(10.17, 2.47, 10.41, 2.47, color=C_NAVY, width_pt=2.0)

    s5.add_textbox(
        10.43, 1.78, 2.20, 1.38,
        [
            {"text": "SAÍDA & XAI TRACK A", "size": 9.5, "bold": True, "color": C_WHITE, "align": "ctr", "space_after": 3},
            {"text": "Teste isolado (N=900)\nBootstrap pareado (5.000×)\nTreeSHAP: migração de\ncrédito H₂ (h14 <-> h28)", "size": 9.5, "color": C_WHITE, "align": "ctr"},
        ],
        fill=C_NAVY, line_color=C_NAVY, rounded=True, valign="ctr", margins=(0.04, 0.03, 0.04, 0.03),
    )

    # Linha de referência experimental (Ablação Track A)
    s5.add_textbox(
        3.00, 3.30, 7.16, 0.68,
        [
            {
                "runs": [
                    r("Fluxos Alternativos / Ablação Track A: ", 10.0, True, False, C_DARK),
                    r("C2 (Ref. fixa, 17 atrib.)  |  C5a (Ref. fixa sem LSTM, 9 obs.)  |  C5b (IGA sem LSTM, 8 obs.)  |  C6 (Só LSTM, 8 prev.)", 9.5, False, False, C_MUTED),
                ],
                "align": "ctr",
            }
        ],
        fill="EDF1F7", line_color=C_MUTED, line_pt=1.0, rounded=True, valign="ctr", margins=(0.06, 0.02, 0.06, 0.02),
    )
    s5.add_line_arrow(6.58, 3.16, 6.58, 3.30, color=C_MUTED, width_pt=1.3, dash="dash")
    s5.add_line_arrow(10.16, 3.64, 11.50, 3.64, color=C_MUTED, width_pt=1.2, dash="dash", end_arrow=False)
    s5.add_line_arrow(11.50, 3.64, 11.50, 3.18, color=C_MUTED, width_pt=1.2, dash="dash")

    # Faixa Track B (Inferior)
    s5.add_rect(0.55, 4.24, 12.22, 2.66, fill="F6FBF9", line_color=C_GREEN, line_pt=1.2, rounded=True, radius_adj=2500, name="TrackBContainer")
    s5.add_textbox(0.65, 4.28, 4.40, 0.24, [{"text": "TRACK B — EIXO ESTÁTICO MULTICLASSE (IEC 60599)", "size": 10.0, "bold": True, "color": C_GREEN}], margins=(0.0, 0.0, 0.0, 0.0))

    s5.add_textbox(
        0.68, 4.58, 2.15, 1.38,
        [
            {"text": "ENTRADA & AUDITORIA", "size": 9.5, "bold": True, "color": C_WHITE, "align": "ctr", "space_after": 3},
            {"text": "2.321 brutas -> -359 (-15,5%)\n= 1.962 únicas (5 gases ppm)\nH₂, CH₄, C₂H₆, C₂H₄, C₂H₂\nSplit 70/15/15: 1372/295/295", "size": 9.5, "color": C_WHITE, "align": "ctr"},
        ],
        fill=C_BLUE, line_color=C_NAVY, rounded=True, valign="ctr", margins=(0.04, 0.03, 0.04, 0.03),
    )
    s5.add_line_arrow(2.84, 5.27, 3.12, 5.27, color=C_AMBER, width_pt=2.0)

    s5.add_textbox(
        3.14, 4.58, 2.40, 1.38,
        [
            {"text": "ENGENHARIA + PRÉ-PROC.", "size": 9.5, "bold": True, "color": C_DARK, "align": "ctr", "space_after": 3},
            {"text": "Razões em ppm bruto ->\n5 razões + 9 TH/TH1 = 19 atrib.\nln(x+1) + Min-Max (só treino)\nCustomSMOTE dentro do CV", "size": 9.5, "color": C_DARK, "align": "ctr"},
        ],
        fill=C_AMBER_LIGHT, line_color=C_AMBER, line_pt=1.3, rounded=True, valign="ctr", margins=(0.04, 0.03, 0.04, 0.03),
    )
    s5.add_line_arrow(5.55, 5.27, 5.83, 5.27, color=C_GREEN, width_pt=2.0)

    s5.add_textbox(
        5.85, 4.58, 3.90, 1.38,
        [
            {"text": "CLASSIFICAÇÃO OTIMIZADA IGA-XGBOOST (C3)", "size": 9.5, "bold": True, "color": C_WHITE, "align": "ctr", "space_after": 3},
            {"text": "ImbPipeline: [CustomSMOTE -> Select -> XGBClassifier]\nCromossomo (19 bin + 9 cont.) -> Seleciona 16 de 19 atrib.\n7 classes: Normal, PD, D1, D2, T1, T2, T3", "size": 9.5, "color": C_WHITE, "align": "ctr"},
        ],
        fill=C_GREEN, line_color=C_GREEN, rounded=True, valign="ctr", margins=(0.05, 0.03, 0.05, 0.03),
    )
    s5.add_line_arrow(9.76, 5.27, 10.04, 5.27, color=C_NAVY, width_pt=2.0)

    s5.add_textbox(
        10.06, 4.58, 2.57, 1.38,
        [
            {"text": "SAÍDA & VALIDAÇÃO FÍSICA", "size": 9.5, "bold": True, "color": C_WHITE, "align": "ctr", "space_after": 3},
            {"text": "Teste 7 classes (N=295) e\n6 falhas (N=189) + CV 5×5\nBootstrap pareado (5.000×)\nTreeSHAP (limiares vs. norma)", "size": 9.5, "color": C_WHITE, "align": "ctr"},
        ],
        fill=C_NAVY, line_color=C_NAVY, rounded=True, valign="ctr", margins=(0.04, 0.03, 0.04, 0.03),
    )

    s5.add_textbox(
        3.14, 6.10, 6.61, 0.68,
        [
            {
                "runs": [
                    r("Referências Track B: ", 10.0, True, False, C_DARK),
                    r("C4 (XGBoost fixo, 19 atrib.)  |  M3 (Wu et al. 2021 + IGA, 14/18)  |  Linha base (5 gases ppm)  |  M2 (Duval)  |  M1 (IEC 60599)", 9.5, False, False, C_MUTED),
                ],
                "align": "ctr",
            }
        ],
        fill="EDF1F7", line_color=C_MUTED, line_pt=1.0, rounded=True, valign="ctr", margins=(0.06, 0.02, 0.06, 0.02),
    )
    s5.add_line_arrow(9.75, 6.44, 11.35, 6.44, color=C_MUTED, width_pt=1.2, dash="dash", end_arrow=False)
    s5.add_line_arrow(11.35, 6.44, 11.35, 5.98, color=C_MUTED, width_pt=1.2, dash="dash")
    slides.append(s5)

    # =========================================================================
    # SLIDE 6 — DATASET E PRÉ-PROCESSAMENTO (1min 30s)
    # =========================================================================
    s6 = new_from(
        13,
        notes="""[OBJETIVO] Apresentar os dados verificáveis das bases (registros, gases, classes, divisões) e destacar as decisões metodológicas contra vazamento de informação.
[TEMPO PREVISTO] 1 minuto e 30 segundos (Acumulado: 7:00).
[MENSAGEM CENTRAL] A auditoria prévia (remoção de 359 amostras no Track B e MD5 no Track A) e o encapsulamento estrito de escalonadores e CustomSMOTE no treino garantem testes livres de vazamento.
[FALA SUGERIDA] "No Slide 6 apresentamos os dados auditados de cada eixo. No Track A usamos 3.000 séries de Katser et al. com intervalo de 12 horas por 210 dias; os sensores fornecem 4 gases em escala normalizada (10⁻³) e 0 duplicatas exatas via hash MD5. Dividimos em 1.785 séries de treino, 315 de validação e 900 de teste oficial. No Track B, a base compilada original tinha 2.321 registros em ppm dos 5 gases normativos. Nossa auditoria detectou e removeu 303 réplicas idênticas e 56 amostras com mesma concentração e rótulos conflitantes — 15,5% da base — restando 1.962 amostras únicas, divididas de forma estratificada em 1.372 treino, 295 validação e 295 teste. À direita destacamos quatro barreiras anti-vazamento: (1) razões molares calculadas estritamente em ppm bruto antes de aplicar log e Min-Max; (2) scalers ajustados apenas no treino; (3) janelamento walk-forward interno por transformador; e (4) o CustomSMOTE, que sintetiza apenas os 5 gases brutos dentro de cada dobra da CV e recalcula analiticamente as 14 variáveis derivadas."
[TRANSIÇÃO] "Com os dados saneados e normalizados, analisemos primeiro o módulo de previsão temporal LSTM."
[PERGUNTAS POSSÍVEIS] "Por que não aplicar SMOTE padrão nas 19 colunas?" -> Porque a interpolação linear direta em razões e percentuais viola as equações físicas (a razão das médias difere da média das razões); o CustomSMOTE interpola só nos 5 gases e recalcula as 14 derivadas.""",
    )
    s6.add_header_and_footer("Dados e Pré-Processamento", "Bases Auditadas, Classes e Blindagem contra Vazamento", "06 / 15")

    s6.add_table(
        0.55,
        1.14,
        7.55,
        4.05,
        ["Característica", "Track A (Katser et al., 2022)", "Track B (Li et al., 2019 / DGALab)"],
        [
            ["Natureza e Escala", "Séries online (Δt = 12h, T = 420 pts)\nEscala pré-normalizada (~10^-3)", "Medições pontuais estáticas\nConcentrações absolutas em ppm"],
            ["Auditoria / Limpeza", "3.000 séries verificadas via MD5\n(0 duplicatas exatas)", "2.321 brutas -> -303 réplicas - 56 conflitos\n(-15,5%) -> 1.962 amostras únicas"],
            ["Gases Utilizados", "4 gases: H₂, CO, C₂H₄, C₂H₂\n(sem CH₄ e C₂H₆)", "5 gases: H₂, CH₄, C₂H₆, C₂H₄, C₂H₂\n(cobertura completa Duval / IEC)"],
            ["Divisão (Train/Val/Test)", "2.100 desenv. (1.785 tr / 315 val)\n+ 900 teste oficial isolado", "70/15/15 estratificado:\n1.372 treino / 295 val / 295 teste"],
            ["Classes e Suporte\nno Teste Final", "4 classes (rótulo constante na série):\nNormal(731), PD(38), D1(49), T1(82)", "7 classes IEC (N=295 | Falhas=189):\nNorm(106), PD(17), D1(41), D2(33), T1(16), T2(16), T3(66)"],
            ["Normalização", "ScalerPerGas Min-Max [0, 1]\n(ajustado apenas nas 1.785 de treino)", "Razões em ppm -> ln(x+1) -> Min-Max\n(ajustado apenas nas 1.372 de treino)"],
            ["Desbalanceamento", "sample_weight: w_c = N / (C · N_c)\n(sem síntese nas séries temporais)", "CustomSMOTE (k=5) só nos 5 gases\n+ recálculo analítico no CV"],
        ],
        col_widths_frac=[0.24, 0.38, 0.38],
        header_fill=C_NAVY,
        font_size=10.0,
        header_size=10.5,
        highlight_rows={1: C_AMBER_LIGHT, 5: C_BG_BLUE, 6: C_GREEN_LIGHT},
        aligns=["l", "l", "l"],
        name="TableDatasets",
    )

    # Box compacto de Engenharia de Atributos sob a tabela
    s6.add_textbox(
        0.55,
        5.32,
        7.55,
        1.62,
        [
            {"text": "ESPAÇOS DE ATRIBUTOS CANDIDATOS", "size": 11.0, "bold": True, "color": C_NAVY, "space_after": 4},
            {
                "runs": [
                    r("• Track A (17): ", 10.5, True, False, C_BLUE),
                    r("4 obs. (t=420) + 5 derivados (C₂H₂/C₂H₄, H₂/C₂H₂, TH_katser, C₂H₂/TH_k, C₂H₄/TH_k) + 8 previsões LSTM (h=14, 28).", 10.5),
                ],
                "space_after": 4,
            },
            {
                "runs": [
                    r("• Track B (19): ", 10.5, True, False, C_GREEN),
                    r("5 brutos + 5 razões (CH₄/H₂, C₂H₂/C₂H₄, C₂H₄/C₂H₆, C₂H₂/CH₄, C₂H₂/C₂H₆) + 2 somas (TH, TH₁) + 7 frações %.", 10.5),
                ]
            },
        ],
        fill=C_BG_LIGHT,
        line_color=C_BORDER,
        rounded=True,
        valign="ctr",
        margins=(0.10, 0.05, 0.10, 0.05),
        name="BoxFeatures",
    )

    # Coluna direita: 4 barreiras contra vazamento
    s6.add_card(
        8.30,
        1.14,
        4.45,
        5.80,
        "Blindagem contra Data Leakage",
        [
            {"runs": [r("1. Ordem Física Inviolável\n", 11.5, True, False, C_NAVY), r("Todas as razões e proporções (TH, TH₁) são calculadas em ppm bruto antes de ln(x+1) e Min-Max (denominador nulo -> 0,0).", 10.8)], "space_after": 8},
            {"runs": [r("2. Escalonamento Estrito no Treino\n", 11.5, True, False, C_NAVY), r("x_min e x_max vêm exclusivamente do treino; validação e teste não são truncados artificialmente se >1,0.", 10.8)], "space_after": 8},
            {"runs": [r("3. Janelamento Walk-Forward\n", 11.5, True, False, C_NAVY), r("Janelas w=50 geradas dentro de cada série sem cruzar transformadores (660.450 tr / 116.550 val / 333.000 te).", 10.8)], "space_after": 8},
            {"runs": [r("4. CustomSMOTE no Pipeline\n", 11.5, True, False, C_GREEN), r("Encapsulado em imblearn.Pipeline dentro de cada dobra (K=5): interpola 5 gases no treino, desnormaliza, recalcula as 14 derivadas e deixa a validação 100% real.", 10.8)]},
        ],
        accent=C_ORANGE,
        fill=C_BG_ORANGE,
        name="CardLeakage",
    )
    slides.append(s6)

    # =========================================================================
    # SLIDE 7 — MODELO LSTM: PREVISÃO TEMPORAL (2min 00s)
    # =========================================================================
    s7 = new_from(
        13,
        notes="""[OBJETIVO] Explicar a arquitetura efetivamente implementada da LSTM no código (src/lstm/model.py e trainer.py), dimensões dos tensores, papel da camada densa e extrapolação recursiva.
[TEMPO PREVISTO] 2 minutos (Acumulado: 9:00).
[MENSAGEM CENTRAL] A Stacked LSTM de 2 camadas (64 unidades + Dropout 0,20) aprende a dinâmica de 1 passo à frente (h=1, 12h) sobre janelas de 50 passos e projeta múltiplos horizontes por realimentação recursiva.
[FALA SUGERIDA] "Para modelar a evolução multivariada dos quatro gases no Track A, avaliamos na validação janelas candidatas de 10, 20, 30 e 50 passos. A janela w*=50 — equivalente a 25 dias de histórico com leituras a cada 12 horas — obteve o menor MAE e foi fixada como entrada (50×4). Conforme implementado em src/lstm/model.py, a arquitetura é uma Stacked LSTM composta por duas camadas recorrentes de 64 unidades cada, utilizando ativação tangente hiperbólica e sigmoide nas portas internas. A primeira camada retorna toda a sequência de estados ocultos (50×64), passa por Dropout de 0,20 para regularização e alimenta a segunda camada, que consolida o vetor de estado oculto final de dimensão 64 (h_50). Após novo Dropout de 0,20, uma camada Densa com ativação estritamente linear e 4 neurônios projeta a representação latente de 64 dimensões nos quatro gases previstos para o próximo instante (h=1, 12 horas). O treinamento minimizou MSE com otimizador Adam (lr=10⁻³), lote de 1.024 janelas, ReduceLROnPlateau e Early Stopping com paciência de 15 épocas (convergência típica em ~16 épocas). Para alcançar horizontes de 1 e 2 semanas (h=14 e h=28), aplicamos extrapolação recursiva, realimentando cada previsão no fim da fila FIFO de 50 passos."
[TRANSIÇÃO] "Geradas as projeções temporais, passemos ao estágio de otimização evolutiva e classificação com IGA-XGBoost."
[PERGUNTAS POSSÍVEIS] "Por que usar camada densa linear em vez de ReLU na saída?" -> Como os dados normalizados podem extrapolar ligeiramente [0,1] no teste ou exigir regressão contínua sem truncamento artificial negativo, a projeção linear evita saturação de gradiente.""",
    )
    s7.add_header_and_footer("Módulo Temporal (Track A)", "Arquitetura da Stacked LSTM e Extrapolação Recursiva", "07 / 15")

    # Diagrama editável da arquitetura real (model.py)
    s7.add_rect(0.55, 1.15, 12.22, 2.38, fill=C_BG_LIGHT, line_color=C_TEAL_DARK, line_pt=1.3, rounded=True, radius_adj=2500, name="LSTMArchBg")
    s7.add_textbox(0.70, 1.20, 8.00, 0.25, [{"text": "ARQUITETURA EFETIVAMENTE IMPLEMENTADA (src/lstm/model.py)", "size": 10.5, "bold": True, "color": C_TEAL_DARK}], margins=(0, 0, 0, 0))

    blocks = [
        ("Entrada Temporal\nInput(50, 4)", "Janela w*=50 passos\n(25 dias × 4 gases:\nH₂, CO, C₂H₄, C₂H₂)", C_BLUE, C_WHITE, 1.85),
        ("LSTM Camada 1\n64 unidades", "return_sequences=True\nAtiv: tanh / sigmoid\nSaída: (B, 50, 64)", C_TEAL_DARK, C_WHITE, 2.00),
        ("Dropout (0,20)\nRegularização", "Desconexão estocástica\n20% no treino\nSaída: (B, 50, 64)", C_MUTED, C_WHITE, 1.65),
        ("LSTM Camada 2\n64 unidades", "return_sequences=False\nConsolida estado oculto\nh₅₀ em R⁶⁴", C_TEAL_DARK, C_WHITE, 2.00),
        ("Dropout (0,20)\n+ Dense(4, linear)", "Projeta h₅₀ (64) nas\n4 concentrações contínuas\nSaída: (B, 4)", C_NAVY, C_WHITE, 2.05),
        ("Previsão ŷ_{t+1}\n(h = 1 -> 12h)", "4 gases no próximo\ntimestep (escala norm.\ninvertida ao final)", C_ORANGE, C_WHITE, 1.72),
    ]
    bx = 0.68
    for i, (ttl, sub, bg, fg, bw) in enumerate(blocks):
        s7.add_textbox(
            bx,
            1.53,
            bw,
            1.80,
            [
                {"text": ttl, "size": 10.5, "bold": True, "color": fg, "align": "ctr", "space_after": 5},
                {"text": sub, "size": 9.5, "color": fg, "align": "ctr", "line_spacing": 115},
            ],
            fill=bg,
            line_color=bg,
            rounded=True,
            radius_adj=6000,
            valign="ctr",
            margins=(0.04, 0.04, 0.04, 0.04),
            name=f"LSTMBlock{i}",
        )
        if i < len(blocks) - 1:
            s7.add_arrow(bx + bw + 0.03, 2.28, 0.14, 0.28, direction="right", fill=C_NAVY, name=f"LSTMArrow{i}")
            bx += bw + 0.20

    # Inferior esquerda: Loop Recursivo Multi-horizonte
    s7.add_card(
        0.55,
        3.68,
        6.10,
        3.25,
        "Extrapolação Recursiva Multi-Horizonte (multi_horizon_eval.py)",
        [
            {"runs": [r("• Passo 1 (t+1): ", 11.5, True, False, C_NAVY), r("janela [x_{t-49}, ..., x_t] -> prevê ŷ_{t+1}.", 11.5)], "space_after": 5},
            {"runs": [r("• Passo k (t+k): ", 11.5, True, False, C_NAVY), r("descarta x mais antigo e insere ŷ anterior na fila FIFO:\n  X_{t+k} = [x_{t-49+k}, ..., x_t, ŷ_{t+1}, ..., ŷ_{t+k-1}].", 11.5)], "space_after": 6},
            {"runs": [r("• Horizontes avaliados: ", 11.5, True, False, C_TEAL_DARK), r("12 horizontes de h=1 (12h) até h=112 (8 semanas).", 11.5)], "space_after": 6},
            {"runs": [r("• Seleção para o XGBoost: ", 11.5, True, False, C_ORANGE), r("h=14 (1 semana / 168h) e h=28 (2 semanas / 336h) — 8 atributos com R² > 0,99 e MAE% < 4%.", 11.5)]},
        ],
        accent=C_TEAL_DARK,
        fill=C_BG_TEAL,
        name="CardRecursive",
    )

    # Inferior direita: Tabela de parâmetros de treino
    s7.add_table(
        6.85,
        3.68,
        5.92,
        3.25,
        ["Parâmetro de Treinamento", "Valor Efetivo (Código)", "Critério / Observação"],
        [
            ["Janela temporal (w*)", "50 passos (25 dias)", "Grid Search {10, 20, 30, 50} na validação"],
            ["Função de perda", "MSE (Mean Squared Error)", "Avaliada por MAE, RMSE e R² no teste"],
            ["Otimizador", "Adam (lr = 10^-3)", "ReduceLROnPlateau (fator 0,5; pac. 7; min 10^-6)"],
            ["Lote / Épocas máx.", "batch_size = 1.024 / 200", "660.450 janelas de treino (1.785 séries)"],
            ["Parada antecipada", "EarlyStopping (pac. = 15)", "val_loss, min_delta=10^-6; típ. ~16 épocas"],
            ["Reprodutibilidade", "5 sementes independentes", "Seeds: [42, 123, 456, 789, 2024]"],
        ],
        col_widths_frac=[0.33, 0.30, 0.37],
        header_fill=C_NAVY,
        font_size=10.5,
        header_size=10.8,
        highlight_rows={0: C_BG_TEAL, 1: C_AMBER_LIGHT},
        aligns=["l", "ctr", "l"],
        name="TableLSTMParams",
    )
    slides.append(s7)

    # =========================================================================
    # SLIDE 8 — IGA E XGBOOST: CLASSIFICAÇÃO (2min 00s)
    # =========================================================================
    s8 = new_from(
        13,
        notes="""[OBJETIVO] Explicar como o IGA otimiza conjuntamente atributos e hiperparâmetros, diferenciar função objetivo interna do XGBoost de métrica de fitness/avaliação e ilustrar o ciclo evolutivo.
[TEMPO PREVISTO] 2 minutos (Acumulado: 11:00).
[MENSAGEM CENTRAL] O cromossomo bipartido acopla seleção de atributos (binário) e sintonia de 9 hiperparâmetros do XGBoost (contínuo); o XGBoost minimiza mlogloss regularizada enquanto o IGA maximiza F1-macro em CV 5-fold no treino.
[FALA SUGERIDA] "Na etapa de classificação, o Algoritmo Genético Melhorado — IGA — otimiza simultaneamente quais variáveis entram no modelo e quais hiperparâmetros controlam as árvores do XGBoost. Como ilustrado no topo, cada candidato é codificado por um cromossomo bipartido: uma máscara binária de tamanho p (17 no Track A ou 19 no Track B) e um vetor contínuo com 9 genes mapeados linearmente aos hiperparâmetros do XGBoost. Na inicialização (população de 50 indivíduos), aplicamos um viés de domínio que ativa razões normativas IEC com probabilidade de 0,80 contra 0,50 nos demais atributos. O ciclo evolutivo avalia cada indivíduo pela média do F1-macro em validação cruzada estratificada de 5 dobras estritamente sobre o conjunto de treino — no Track B com o CustomSMOTE dentro de cada dobra. Aplicamos seleção por torneio de tamanho 3, elitismo dos 2 melhores, crossover uniforme na máscara binária e BLX-alfa (0,50) nos contínuos com taxa 0,90, além de mutação bit-flip (0,10) e gaussiana (sigma 0,10), parando após 100 gerações ou 20 sem melhoria. É fundamental não confundir as métricas: internamente ao construir as árvores, o XGBoost minimiza a função objetivo mlogloss regularizada (L1, L2 e gamma), enquanto o IGA e a avaliação final utilizam o F1-macro para garantir equidade entre classes raras e frequentes."
[TRANSIÇÃO] "Agora veremos como essa arquitetura foi organizada experimentalmente nos Tracks A e B."
[PERGUNTAS POSSÍVEIS] "Por que árvores de profundidade 12 no Track A (C1) e Track B (C3) não sofreram overfitting?" -> Porque o IGA compensou com regularização: no Track B, colsample_bytree caiu para 0,3433 (forçando diversidade tipo Random Forest), reg_lambda subiu para 3,94 e gamma=1,44 + min_child_weight=3 impuseram poda rigorosa.""",
    )
    s8.add_header_and_footer("Classificação e Otimização Evolutiva", "IGA-XGBoost: Cromossomo Bipartido, Ciclo Evolutivo e Hiperparâmetros", "08 / 15", title_size=21.0)

    # Cromossomo Bipartido (Topo)
    s8.add_rect(0.55, 1.12, 12.22, 1.05, fill=C_BG_LIGHT, line_color=C_NAVY, line_pt=1.2, rounded=True, radius_adj=2500, name="ChromBg")
    s8.add_textbox(0.70, 1.16, 8.50, 0.22, [{"text": "REPRESENTAÇÃO DO CROMOSSOMO HÍBRIDO (src/iga_xgboost/cromossomo.py)", "size": 9.8, "bold": True, "color": C_NAVY}], margins=(0, 0, 0, 0))
    s8.add_textbox(
        0.70, 1.43, 5.25, 0.62,
        [
            {"text": "Parte Binária: Máscara de Atributos (p bits: 17 no Track A | 19 no Track B)", "size": 10.5, "bold": True, "color": C_WHITE, "align": "ctr"},
            {"text": "Bit ≥ 0,5 -> ativo  |  Mín. 2 ativos  |  Inicialização enviesada: Razões IEC (p=0,80) vs. demais (0,50)", "size": 9.0, "color": "E6EEFF", "align": "ctr"},
        ],
        fill=C_BLUE, line_color=C_NAVY, rounded=True, valign="ctr", margins=(0.04, 0.02, 0.04, 0.02),
    )
    s8.add_textbox(
        6.05, 1.43, 6.55, 0.62,
        [
            {"text": "Parte Contínua: 9 Genes Reais em [0, 1] -> Hiperparâmetros do XGBoost", "size": 10.5, "bold": True, "color": C_WHITE, "align": "ctr"},
            {"text": "learning_rate · max_depth · n_estimators · subsample · colsample_bytree · gamma · min_child_weight · reg_alpha · reg_lambda", "size": 9.0, "color": "FFF5EB", "align": "ctr"},
        ],
        fill=C_ORANGE, line_color=C_ORANGE, rounded=True, valign="ctr", margins=(0.04, 0.02, 0.04, 0.02),
    )

    # Fluxograma do ciclo evolutivo (Meio)
    flow_steps = [
        ("1. População\nInicial", "N = 50 indivíduos\nViés IEC = 0,80", C_BLUE),
        ("2. Avaliação\n(Fitness)", "StratifiedKFold (5)\nF1-macro no treino\n(+CustomSMOTE em B)", C_TEAL_DARK),
        ("3. Seleção &\nElitismo", "Torneio (k = 3)\nPreserva 2 melhores\n(Elitismo direto)", C_NAVY),
        ("4. Operadores\nGenéticos", "Pc=0,90 (Unif. + BLX-0,5)\nPm=0,10 (Bit-flip +\nGauss σ=0,10)", C_AMBER),
        ("5. Critério de\nParada", "G_max = 100 ger.\nou 20 gerações\nsem melhoria", C_MUTED),
        ("6. Modelo\nFinal", "Treina XGBoost\nótimo no treino todo\ne avalia no teste", C_GREEN),
    ]
    fx = 0.55
    fw = 1.85
    for i, (ft, fb, fc) in enumerate(flow_steps):
        s8.add_textbox(
            fx, 2.30, fw, 1.25,
            [
                {"text": ft, "size": 10.5, "bold": True, "color": C_WHITE, "align": "ctr", "space_after": 3},
                {"text": fb, "size": 9.2, "color": C_WHITE, "align": "ctr", "line_spacing": 110},
            ],
            fill=fc, line_color=fc, rounded=True, radius_adj=5000, valign="ctr", margins=(0.04, 0.03, 0.04, 0.03),
            name=f"IGAFlow{i}",
        )
        if i < len(flow_steps) - 1:
            s8.add_arrow(fx + fw + 0.03, 2.80, 0.15, 0.25, direction="right", fill=C_NAVY, name=f"IGAArrow{i}")
            fx += fw + 0.22

    # Inferior esquerda: Tabela de hiperparâmetros reais
    s8.add_table(
        0.55,
        3.70,
        7.55,
        3.22,
        ["Hiperparâmetro XGBoost", "Faixa de Busca", "Ref. Fixa (C2/C4)", "Track A (C1)", "Track B (C3)"],
        [
            ["learning_rate", "[0,01; 0,30]", "0,10", "0,1282", "0,0596"],
            ["max_depth", "[3; 12]", "6", "12", "12"],
            ["n_estimators", "[50; 500]", "100", "422", "342"],
            ["subsample / colsample", "[0,5; 1] / [0,3; 1]", "0,90 / 1,00", "1,00 / 0,9453", "0,9654 / 0,3433"],
            ["gamma / min_child_weight", "[0; 5] / [1; 10]", "0,00 / 1", "0,00 / 3", "1,4356 / 3"],
            ["reg_alpha (L₁) / reg_lambda (L₂)", "[0; 1] / [0; 5]", "0,00 / 1,00", "0,1054 / 1,5461", "0,3182 / 3,9425"],
        ],
        col_widths_frac=[0.30, 0.20, 0.17, 0.16, 0.17],
        header_fill=C_NAVY,
        font_size=9.8,
        header_size=10.0,
        highlight_rows={3: C_BG_BLUE, 4: C_GREEN_LIGHT, 5: C_GREEN_LIGHT},
        aligns=["l", "ctr", "ctr", "ctr", "ctr"],
        name="TableHparams",
    )

    # Inferior direita: Distinção Objetivo XGBoost vs Fitness / Métrica + Características do Ensemble
    s8.add_card(
        8.30,
        3.70,
        4.47,
        3.22,
        "Função Objetivo vs. Avaliação",
        [
            {"runs": [r("• Objetivo Interno do XGBoost:\n", 11.2, True, False, C_NAVY), r("  Minimiza multi-class log-loss (mlogloss) com regularização de 2ª ordem: Ω(f) = γT_L + ½λ||w||² + α||w||₁ (tree_method='hist').", 10.5)], "space_after": 6},
            {"runs": [r("• Aptidão do IGA e Avaliação Final:\n", 11.2, True, False, C_GREEN), r("  Maximiza F1-macro (média não ponderada das classes), garantindo sensibilidade às falhas raras.", 10.5)], "space_after": 6},
            {"runs": [r("• Regularização Aprendida (C3):\n", 11.2, True, False, C_ORANGE), r("  Baixo colsample (0,34) + alto λ (3,94) + γ (1,44) evitam sobreajuste mesmo com max_depth=12.", 10.5)]},
        ],
        accent=C_NAVY,
        fill=C_BG_LIGHT,
        name="CardXGBObj",
    )
    slides.append(s8)

    # =========================================================================
    # SLIDE 9 — TRACK A E TRACK B: DESENHO EXPERIMENTAL (2min 00s)
    # =========================================================================
    s9 = new_from(
        13,
        notes="""[OBJETIVO] Esclarecer exatamente por que existem dois tracks, seus objetivos distintos, configurações avaliadas e protocolos de comparação justa.
[TEMPO PREVISTO] 2 minutos (Acumulado: 13:00).
[MENSAGEM CENTRAL] Track A isola o impacto das projeções LSTM no diagnóstico por meio de ablação controlada (5 configurações); Track B estabelece o benchmark multiclasse completo (7 e 6 classes) frente a Wu et al. (2021) e às normas IEC 60599 e Duval.
[FALA SUGERIDA] "Para evitar confundir o papel de cada experimento, este quadro resume o desenho experimental dos dois eixos. O Track A responde à pergunta temporal: prever o futuro dos gases ajuda a classificar o estado atual em séries contínuas? Como os rótulos das séries de Katser permanecem constantes nos 420 passos, extraímos o snapshot em t=420 com janela prévia w=50 e avaliamos cinco configurações no mesmo teste (N=900): C2 (referência fixa com todos os 17 atributos), C1 (IGA-XGBoost sobre os 17), C5a e C5b (apenas os 9 atributos estáticos observados, sem LSTM, sob parâmetros fixos e IGA) e C6 (somente as 8 projeções da LSTM). O Track B responde à pergunta multiclasse normativa sobre 7 categorias com os 5 gases completos: confrontamos nosso IGA-XGBoost (C3, 16 de 19 atributos) contra o XGBoost fixo (C4, 19), a reimplementação do espaço de Wu et al. (2021) com IGA (M3, 14 de 18), a linha de base com os 5 gases brutos em ppm, o Triângulo 1 de Duval (M2) e as Razões IEC 60599 (M1). Crucialmente, como Duval e IEC não definem classe Normal, avaliamos o Track B também no subconjunto estrito de 189 falhas ativas (6 classes) para uma comparação justa."
[TRANSIÇÃO] "A seguir, detalhamos exatamente quais métricas e testes estatísticos sustentam essas comparações."
[PERGUNTAS POSSÍVEIS] "Qual a diferença exata entre os 19 candidatos do C3 e os 18 de Wu et al. (M3)?" -> C3 inclui a razão de Doernenburg C2H2/CH4, ausente no conjunto de 18 variáveis de Wu et al. (2021).""",
    )
    s9.add_header_and_footer("Desenho Experimental", "Track A vs. Track B: Objetivos, Configurações e Baselines", "09 / 15")

    s9.add_table(
        0.55,
        1.10,
        12.22,
        4.60,
        ["Critério Metodológico", "Track A — Acoplamento Temporal-Estático", "Track B — Benchmark Multiclasse IEC 60599"],
        [
            ["Pergunta de Pesquisa", "Projeções futuras da LSTM melhoram a classificação do estado operacional frente ao snapshot atual?", "Como o IGA-XGBoost se compara a Wu et al. (2021) e às normas IEC/Duval em base saneada?"],
            ["Entradas e Papel da LSTM", "4 gases (H₂, CO, C₂H₄, C₂H₂). LSTM gera 8 previsões (h=14, 28) somadas a 4 obs. (t=420) + 5 razões = 17 atrib.", "5 gases em ppm + 5 razões + 9 frações TH/TH₁ = 19 atrib. Sem LSTM (dados estáticos pontuais)."],
            ["Escopo do Diagnóstico", "4 classes (Normal, PD, D1, T1) — rótulo constante ao longo da série (não mede RUL).", "7 classes IEC (Normal, PD, D1, D2, T1, T2, T3) + recorte justo de 6 falhas ativas (N=189)."],
            ["Configurações / Ablação", "• C2: XGBoost fixo (17 atrib. c/ LSTM)\n• C1: IGA-XGBoost (17 -> 8 atrib.)\n• C5a / C5b: Fixo / IGA sem LSTM (9 obs.)\n• C6: Fixo só LSTM (8 prev.) [C1-L pós-hoc]", "• C3: IGA-XGBoost (19 -> 16 atrib.)\n• C4: XGBoost fixo (19 atrib.)\n• M3: Wu et al. (2021) + IGA (18 -> 14 atrib.)\n• Linha base (5 gases ppm) | M2 Duval | M1 IEC"],
            ["Desbalanceamento", "sample_weight analítico no XGBoost\nw_c = N / (C · N_c)", "CustomSMOTE (k=5) nos 5 gases brutos dentro do CV (ImbPipeline); teste 100% real."],
            ["Protocolo de Validação", "Teste oficial isolado (N=900) · 5 sementes · Bootstrap pareado estratificado (5.000×).", "Teste isolado (N=295 e N=189) · 5 seeds · Bootstrap pareado (5.000×) · Repeated CV 5×5 (1.667 desenv.)."],
        ],
        col_widths_frac=[0.20, 0.40, 0.40],
        header_fill=C_NAVY,
        font_size=9.7,
        header_size=10.5,
        highlight_rows={0: C_BG_BLUE, 3: C_AMBER_LIGHT},
        aligns=["l", "l", "l"],
        name="TableTracks",
    )

    s9.add_textbox(
        0.55,
        5.82,
        12.22,
        1.14,
        [
            {"text": "POR QUE DOIS PROTOCOLOS SÃO NECESSÁRIOS PARA RESPONDER ÀS PERGUNTAS DO TCC?", "size": 11.0, "bold": True, "color": C_ORANGE, "space_after": 4},
            {
                "runs": [
                    r("1) Isolamento causal da hipótese temporal (Track A): ", 11.0, True, False, C_NAVY),
                    r("as ablações C2×C5a e C1×C5b testam diretamente se a LSTM agrega informação além do instante presente.  ", 11.0),
                    r("2) Comparação normativa justa (Track B): ", 11.0, True, False, C_GREEN),
                    r("avalia o modelo nos 5 gases normativos e separa o cenário de 6 falhas ativas (sem Normal), eliminando a desvantagem estrutural das normas.", 11.0),
                ],
                "line_spacing": 118,
            },
        ],
        fill=C_BG_LIGHT,
        line_color=C_NAVY,
        line_pt=1.2,
        rounded=True,
        valign="ctr",
        margins=(0.12, 0.05, 0.12, 0.05),
        name="BoxWhyTracks",
    )
    slides.append(s9)

    # =========================================================================
    # SLIDE 10 — MÉTRICAS E VALIDAÇÃO (1min 00s)
    # =========================================================================
    s10 = new_from(
        13,
        notes="""[OBJETIVO] Diferenciar métrica de otimização, validação e teste final; justificar F1-macro; distinguir CV, multi-seed e bootstrap pareado.
[TEMPO PREVISTO] 1 minuto (Acumulado: 14:00).
[MENSAGEM CENTRAL] Cada etapa usa a métrica estatisticamente adequada; o desvio entre 5 sementes mede apenas estabilidade algorítmica, enquanto o bootstrap pareado estratificado (5.000×) quantifica a incerteza amostral do conjunto de teste.
[FALA SUGERIDA] "Para garantir rigor metodológico, separamos claramente três papéis de métricas. Durante o treino, a LSTM minimiza o MSE e o XGBoost minimiza a log-loss multiclasse regularizada. Na validação e seleção, escolhemos a janela da LSTM pelo menor MAE e avaliamos os indivíduos do IGA pelo F1-macro médio em 5 dobras sobre o treino — utilizando também a validação cruzada repetida 5×5 no Track B para medir estabilidade paramétrica. No teste final isolado, adotamos o F1-macro como métrica principal por atribuir peso idêntico às sete classes, evitando que a classe Normal (81% do Track A) mascare falhas raras como PD, T1 e T2. Por fim, não confundimos variância entre sementes com incerteza amostral: aplicamos bootstrap pareado e estratificado com 5.000 reamostragens no teste, avaliando os modelos nos mesmos índices reamostrados e concluindo significância somente quando o IC de 95% de Delta F1 exclui o zero."
[TRANSIÇÃO] "Passamos agora aos resultados experimentais obtidos, iniciando pelo módulo LSTM."
[PERGUNTAS POSSÍVEIS] "Por que usar bootstrap estratificado em vez de bootstrap simples?" -> Porque classes minoritárias como T1 e T2 têm apenas 16 amostras no teste; reamostrar estratificadamente preserva o suporte exato por classe e evita réplicas degeneradas sem exemplos de uma falha.""",
    )
    s10.add_header_and_footer("Métricas e Rigor Estatístico", "Separação de Métricas, Justificativa do F1-Macro e Bootstrap Pareado", "10 / 15")

    s10.add_card(
        0.55,
        1.14,
        3.88,
        3.55,
        "1. Otimização (Treinamento)",
        [
            {"runs": [r("• LSTM (Regressão): ", 12.0, True, False, C_TEAL_DARK), r("minimiza MSE com Adam (lr=10^-3) sobre janelas normalizadas.", 11.5)], "space_after": 8},
            {"runs": [r("• XGBoost (Classificação): ", 12.0, True, False, C_NAVY), r("minimiza mlogloss regularizada (L₁, L₂, γ) por derivadas de 1ª/2ª ordem.", 11.5)], "space_after": 8},
            {"runs": [r("• Nota técnica: ", 11.5, True, False, C_ORANGE), r("mlogloss calibra probabilidades internas; não é a métrica de seleção.", 11.0)]},
        ],
        accent=C_TEAL_DARK,
        fill=C_BG_TEAL,
        name="CardMet1",
    )
    s10.add_card(
        4.72,
        1.14,
        3.88,
        3.55,
        "2. Validação e Seleção",
        [
            {"runs": [r("• Seleção w* (LSTM): ", 12.0, True, False, C_BLUE), r("menor MAE agregado nas 315 séries de validação.", 11.5)], "space_after": 8},
            {"runs": [r("• Fitness IGA (Treino): ", 12.0, True, False, C_BLUE), r("F1-macro médio em StratifiedKFold (K=5) sem tocar validação/teste.", 11.5)], "space_after": 8},
            {"runs": [r("• Estabilidade (Track B): ", 12.0, True, False, C_BLUE), r("Repeated Stratified K-Fold 5×5 (25 dobras) nas 1.667 amostras de desenv.", 11.5)]},
        ],
        accent=C_BLUE,
        fill=C_BG_BLUE,
        name="CardMet2",
    )
    s10.add_card(
        8.89,
        1.14,
        3.88,
        3.55,
        "3. Teste Final e Inferência",
        [
            {"runs": [r("• Regressão LSTM: ", 12.0, True, False, C_GREEN), r("MAE, RMSE, MAE% e R² (corte operacional R² ≥ 0,80).", 11.5)], "space_after": 8},
            {"runs": [r("• Classificação: ", 12.0, True, False, C_GREEN), r("F1-macro médio em 5 sementes no teste cego (N=900 / 295 / 189).", 11.5)], "space_after": 8},
            {"runs": [r("• Inferência Estatística: ", 12.0, True, False, C_GREEN), r("Bootstrap pareado estratificado (5.000 réplicas) -> IC 95% de ΔF1.", 11.5)]},
        ],
        accent=C_GREEN,
        fill=C_GREEN_LIGHT,
        name="CardMet3",
    )

    # Rodapé técnico de justificativas
    s10.add_textbox(
        0.55,
        4.88,
        5.95,
        2.05,
        [
            {"text": "POR QUE ADOTAR F1-MACRO COMO MÉTRICA CENTRAL?", "size": 11.5, "bold": True, "color": C_NAVY, "space_after": 5},
            {"runs": [r("• Acurácia global engana em DGA: ", 11.5, True, False, C_RED), r("no Track A, 81,2% do teste é Normal (731/900); um classificador trivial já superaria 81% de acurácia errando todas as falhas.", 11.5)], "space_after": 5},
            {"runs": [r("• Peso equitativo: ", 11.5, True, False, C_GREEN), r("F1_macro = (1/C) Σ F1_c exige equilíbrio entre precisão e recall em classes críticas raras (PD, T1, T2).", 11.5)]},
        ],
        fill=C_BG_LIGHT,
        line_color=C_BORDER,
        rounded=True,
        valign="ctr",
        margins=(0.10, 0.06, 0.10, 0.06),
        name="BoxF1Macro",
    )
    s10.add_textbox(
        6.72,
        4.88,
        6.05,
        2.05,
        [
            {"text": "DISTINÇÃO: CROSS-VALIDATION vs. MULTI-SEED vs. BOOTSTRAP", "size": 11.5, "bold": True, "color": C_ORANGE, "space_after": 5},
            {"runs": [r("• 5 Seeds (±σ): ", 11.5, True, False, C_NAVY), r("mede apenas aleatoriedade estocástica de treino (σ ≈ 0,001–0,006).", 11.5)], "space_after": 4},
            {"runs": [r("• CV Repetida 5×5: ", 11.5, True, False, C_NAVY), r("mede variabilidade entre partições de desenvolvimento (25 dobras).", 11.5)], "space_after": 4},
            {"runs": [r("• Bootstrap Pareado (5.000×): ", 11.5, True, False, C_NAVY), r("mede incerteza amostral do teste nos mesmos índices; ΔF1 é significativo só se o IC 95% excluir zero.", 11.5)]},
        ],
        fill=C_BG_ORANGE,
        line_color=C_ORANGE,
        rounded=True,
        valign="ctr",
        margins=(0.10, 0.06, 0.10, 0.06),
        name="BoxBootVsCV",
    )
    slides.append(s10)

    # =========================================================================
    # SLIDE 11 — RESULTADOS DA PREVISÃO TEMPORAL (1min 30s)
    # =========================================================================
    s11 = new_from(
        13,
        notes="""[OBJETIVO] Apresentar os resultados reais da previsão LSTM (h=1 e multi-horizonte), confrontá-los com o baseline de persistência e interpretar criticamente a inércia temporal.
[TEMPO PREVISTO] 1 minuto e 30 segundos (Acumulado: 15:30).
[MENSAGEM CENTRAL] A LSTM prevê com alta fidelidade quantitativa até 7 semanas (h=98, R² ≥ 0,80), mas a persistência ingênua já atinge R² = 0,9970 em 1 semana devido à altíssima inércia das séries.
[FALA SUGERIDA] "No teste oficial de 900 séries independentes (333.000 janelas em h=1), a LSTM alcançou MAE médio de 5,59×10⁻⁵ e R² médio de 0,9996 em cinco sementes, mantendo consistência nos quatro gases. Na análise multi-horizonte recursiva (tabela inferior e gráfico superior direito), a rede sustenta R² acima de 0,99 até 2 semanas (h=28, MAE% de 3,76%), e o limite operacional uniforme de R² ≥ 0,80 é atendido até 7 semanas (h=98, com H₂ em 0,8491), rompendo-se em 8 semanas quando o hidrogênio cai para 0,7877. Contudo, o achado científico mais revelador surge ao comparar a LSTM contra um preditor ingênuo de persistência (ŷ_{t+h} = y_t): em 1 e 2 semanas, a simples repetição do valor atual já obtém R² de 0,9970 e 0,9886! Portanto, embora as curvas previstas acompanhem de perto as séries reais, o ganho sobre a persistência é marginal (ΔR² ≈ +0,001 a +0,004), evidenciando fortíssima inércia temporal — o que explica o comportamento observado em seguida no classificador."
[TRANSIÇÃO] "Veremos agora se essas projeções temporais, quando fornecidas ao XGBoost, agregaram valor ao diagnóstico nos Tracks A e B."
[PERGUNTAS POSSÍVEIS] "Por que o CO tem MAE 85× maior que o C2H2 em h=1?" -> Por diferença de amplitude física na escala original (Katser ~10⁻³): CO varia em ordem bem maior que C2H2; quando olhamos o R² (0,9997 vs. 0,9996) e o MAE% (0,88% vs. 1,00% em h=1), ambos os gases têm precisão relativa equivalente.""",
    )
    s11.add_header_and_footer("Resultados — Estágio 1 (LSTM)", "Previsão Temporal: Desempenho h=1, Multi-Horizonte e Persistência", "11 / 15", title_size=21.5)

    # Tabela 1: h=1
    s11.add_table(
        0.55,
        1.12,
        6.10,
        2.15,
        ["Gás (h=1, N=333k janelas)", "MAE (×10^-5)", "RMSE (×10^-4)", "R² (5 seeds)"],
        [
            ["Hidrogênio (H₂)", "1,44 ± 0,13", "0,18 ± 0,02", "0,9995 ± 0,0001"],
            ["Monóxido de Carbono (CO)", "16,15 ± 2,72", "2,06 ± 0,31", "0,9997 ± 0,0001"],
            ["Etileno (C₂H₄)", "4,58 ± 0,59", "0,58 ± 0,07", "0,9996 ± 0,0001"],
            ["Acetileno (C₂H₂)", "0,19 ± 0,01", "0,02 ± 0,01", "0,9996 ± 0,0001"],
            ["Média Agregada", "5,59 ± 0,74", "1,07 ± 0,15", "0,9996 ± 0,0001"],
        ],
        col_widths_frac=[0.38, 0.21, 0.21, 0.20],
        header_fill=C_NAVY,
        font_size=9.5,
        header_size=9.8,
        highlight_rows={4: C_BG_TEAL},
        name="TableLSTMh1",
    )

    # Tabela 2: Multi-horizonte vs Persistência
    s11.add_table(
        0.55,
        3.38,
        6.10,
        2.42,
        ["Horizonte (h)", "R² Persist.", "R² LSTM", "MAE% LSTM", "Status (R²≥0,80)"],
        [
            ["h = 1 (12h)", "1,0000", "0,9996", "0,90%", "Conforme"],
            ["h = 14 (1 sem.) *", "0,9970", "0,9979", "2,05%", "Adotado XGB"],
            ["h = 28 (2 sem.) *", "0,9886", "0,9928", "3,76%", "Adotado XGB"],
            ["h = 56 (4 sem.)", "0,9588", "0,9691", "7,76%", "Conforme"],
            ["h = 98 (7 sem.)", "0,8940", "0,8846", "14,96%", "Máx. Uniforme"],
            ["h = 112 (8 sem.)", "0,8696", "0,8380", "17,68%", "H₂=0,788 < 0,80"],
        ],
        col_widths_frac=[0.24, 0.16, 0.16, 0.17, 0.27],
        header_fill=C_TEAL_DARK,
        font_size=9.2,
        header_size=9.5,
        highlight_rows={1: C_GREEN_LIGHT, 2: C_GREEN_LIGHT, 5: C_AMBER_LIGHT},
        name="TableLSTMMulti",
    )

    # Interpretação crítica abaixo das tabelas
    s11.add_textbox(
        0.55,
        5.92,
        6.10,
        1.05,
        [
            {
                "runs": [
                    r("Interpretação Crítica (Inércia Temporal): ", 10.5, True, False, C_RED),
                    r("Embora o ajuste visual pareça próximo e R²>0,99 até 2 semanas, o baseline de persistência (ŷ_{t+h}=y_t) já alcança R²=0,9970 (1 sem.) e 0,9886 (2 sem.). As concentrações evoluem lentamente, tornando as projeções quase colineares ao valor atual.", 10.0),
                ],
                "line_spacing": 112,
            }
        ],
        fill=C_RED_LIGHT,
        line_color=C_RED,
        line_pt=1.1,
        rounded=True,
        valign="ctr",
        margins=(0.08, 0.04, 0.08, 0.04),
        name="BoxLSTMInterp",
    )

    # Figuras à direita
    s11.add_image_fit(FIG_DIR / "lstm_multi_horizon_r2_norm.png", 6.80, 1.12, 5.95, 2.80, border_color=C_BORDER, name="FigMultiHorizon")
    s11.add_image_fit(FIG_DIR / "lstm_predictions_sample_norm.png", 6.80, 4.02, 5.95, 2.95, border_color=C_BORDER, name="FigPredSample")
    slides.append(s11)

    # =========================================================================
    # SLIDE 12 — RESULTADOS COMPARATIVOS DE CLASSIFICAÇÃO (1min 30s)
    # =========================================================================
    s12 = new_from(
        13,
        notes="""[OBJETIVO] Apresentar os resultados quantitativos centrais de classificação dos Tracks A e B, respeitando protocolos distintos.
[TEMPO PREVISTO] 1 minuto e 30 segundos (Acumulado: 17:00).
[MENSAGEM CENTRAL] No Track A, o IGA reduz 53% dos atributos mantendo F1=0,891, mas modelos sem LSTM (C5a=0,886) igualam o desempenho; no Track B, os modelos baseados em árvores superam largamente as normas (0,800 vs. 0,450), mas estabilizam em um teto empírico comum (~0,76–0,80).
[FALA SUGERIDA] "Este é o slide central de resultados comparativos, separando rigorosamente os dois protocolos incompatíveis entre si. À esquerda superior, no Track A (N=900, 4 classes), observamos que C2 (fixo com LSTM, 17 atributos) atinge F1-macro de 0,8911 e C1 (IGA com LSTM, 8 atributos) atinge 0,8909 — redução de 53% na dimensão com o mesmo escore. Porém, ao retirar a LSTM desde a origem em C5a (9 atributos estáticos observados), o F1 é 0,8855, com total sobreposição dos intervalos de confiança de 95%. Ressaltamos que a queda da linha 'C1 sem LSTM (pós-hoc)' para 0,8280 é apenas um artefato: o IGA havia selecionado previsões da LSTM no lugar de H₂, C₂H₄ e C₂H₂ observados; o teste justo é C5a/C5b. À esquerda inferior, no Track B (7 classes, N=295; e 6 falhas ativas, N=189), o IGA-XGBoost C3 atinge F1 de 0,7970 no teste global e 0,8000 em falhas ativas, contra 0,4499 de Duval e 0,3301 das Razões IEC 60599. Contudo, C4 (fixo, 0,7843) e Wu et al. (M3, 0,7829) são equivalentes a C3, e na validação cruzada repetida 5×5 todos estabilizam entre 0,752 e 0,765, superando a linha base bruta (0,7250 teste / 0,7082 CV), como sintetiza o gráfico de complexidade versus F1 à direita."
[TRANSIÇÃO] "Para confirmar o que é diferença estatística real e explicar fisicamente esses números, passamos ao bootstrap pareado e ao TreeSHAP."
[PERGUNTAS POSSÍVEIS] "Por que C3 teve F1=0,7970 no teste e 0,7516 na CV 5×5?" -> Na CV 5×5 não usamos CustomSMOTE dentro das dobras (para medir estabilidade pura), e o teste tem N=295 com apenas 16 amostras em T1 e T2; na validação única foi 0,7038, dentro da variância amostral esperada.""",
    )
    s12.add_header_and_footer("Resultados Comparativos", "Classificação: Ablação (Track A) e Multiclasse IEC (Track B)", "12 / 15", title_size=21.5)

    # Track A tabela
    s12.add_textbox(0.55, 1.07, 6.80, 0.24, [{"text": "TRACK A — Ablação Temporal (Katser, N=900 teste, 4 classes, 5 seeds)", "size": 10.2, "bold": True, "color": C_BLUE}], margins=(0, 0, 0, 0))
    s12.add_table(
        0.55,
        1.32,
        6.95,
        2.32,
        ["Configuração (Track A)", "Atrib.", "F1-macro (±σ)", "IC 95% Bootstrap"],
        [
            ["C2: Ref. fixa com LSTM", "17", "0,8911 ± 0,0034", "[0,8574; 0,9229]"],
            ["C1: IGA com LSTM", "8", "0,8909 ± 0,0011", "[0,8549; 0,9230]"],
            ["C6: Ref. fixa apenas LSTM", "8", "0,8888 ± 0,0011", "[0,8547; 0,9212]"],
            ["C5a: Ref. fixa sem LSTM", "9", "0,8855 ± 0,0029", "[0,8507; 0,9177]"],
            ["C5b: IGA sem LSTM", "8", "0,8796 ± 0,0051", "[0,8469; 0,9110]"],
            ["C1 s/ LSTM (pós-hoc)*", "4", "0,8280 ± 0,0030", "[0,7876; 0,8656]"],
        ],
        col_widths_frac=[0.39, 0.11, 0.23, 0.27],
        header_fill=C_BLUE,
        font_size=9.2,
        header_size=9.5,
        highlight_rows={1: C_BG_BLUE, 3: C_GREEN_LIGHT, 5: C_AMBER_LIGHT},
        name="TableTrackA",
    )

    # Track B tabela
    s12.add_textbox(0.55, 3.74, 6.95, 0.24, [{"text": "TRACK B — Diagnóstico Multiclasse (Base Auditada, N=295 / Falhas N=189)", "size": 10.2, "bold": True, "color": C_GREEN}], margins=(0, 0, 0, 0))
    s12.add_table(
        0.55,
        3.99,
        6.95,
        2.58,
        ["Modelo / Método (Track B)", "Atrib.", "F1 Teste (7 cls.)", "F1 Falhas (6 cls.)", "F1 CV 5×5"],
        [
            ["C3: IGA-XGBoost (Proposto)", "16", "0,7970 [0,739; 0,852]", "0,8000 [0,732; 0,858]", "0,7516 ± 0,028"],
            ["C4: XGBoost Ref. Fixa", "19", "0,7843 [0,726; 0,838]", "0,7780 [0,710; 0,840]", "0,7616 ± 0,023"],
            ["M3: Wu et al. (2021) + IGA", "14", "0,7829 [0,726; 0,838]", "0,7845 [0,718; 0,845]", "0,7654 ± 0,025"],
            ["Linha Base (5 gases ppm)", "5", "0,7250 [0,659; 0,785]", "--", "0,7082 ± 0,030"],
            ["M2: Triângulo 1 de Duval", "3", "0,3634 [0,310; 0,414]", "0,4499 [0,382; 0,516]", "-- (90,5% cob.)"],
            ["M1: Razões IEC 60599", "5", "0,2569 [0,194; 0,314]", "0,3301 [0,252; 0,402]", "-- (60,3% cob.)"],
        ],
        col_widths_frac=[0.30, 0.08, 0.23, 0.22, 0.17],
        header_fill=C_GREEN,
        font_size=8.8,
        header_size=9.2,
        highlight_rows={0: C_GREEN_LIGHT, 4: C_RED_LIGHT, 5: C_RED_LIGHT},
        name="TableTrackB",
    )

    # Nota de rodapé técnico
    s12.add_textbox(
        0.55,
        6.63,
        6.95,
        0.42,
        [
            {
                "text": "*Nota: C1 s/ LSTM (pós-hoc) remove H₂, C₂H₄ e C₂H₂ da LSTM sem reotimizar; o contraste justo sem LSTM é C5a / C5b. Não comparar diretamente F1 de Track A (4 classes) e Track B (7 classes).",
                "size": 8.8,
                "italic": True,
                "color": C_MUTED,
            }
        ],
        margins=(0.0, 0.0, 0.0, 0.0),
        name="NoteTable12",
    )

    # Gráfico Trade-off + Quadro Resumo à direita
    s12.add_image_fit(FIG_DIR / "xgb_tradeoff_complexity.png", 7.65, 1.15, 5.12, 3.40, border_color=C_BORDER, name="FigTradeoff")
    s12.add_card(
        7.65,
        4.68,
        5.12,
        2.30,
        "Leitura Diagnóstica dos Resultados",
        [
            {"runs": [r("• Redução Dimensional (Track A): ", 10.8, True, False, C_BLUE), r("IGA corta 17 -> 8 atrib. (−53%) sem perda (ΔF1=−0,0003).", 10.5)], "space_after": 4},
            {"runs": [r("• Equivalência Temporal: ", 10.8, True, False, C_AMBER), r("C5a estático (0,8855) equivale a C2 (0,8911) e C1 (0,8909) sem custo da LSTM.", 10.5)], "space_after": 4},
            {"runs": [r("• Teto Empírico (Track B): ", 10.8, True, False, C_GREEN), r("C3, C4 e M3 convergem a ~0,76–0,80 e superam amplamente as normas mesmo em falhas puras (N=189).", 10.5)]},
        ],
        accent=C_NAVY,
        fill=C_BG_LIGHT,
        name="CardSummary12",
    )
    slides.append(s12)

    # =========================================================================
    # SLIDE 13 — BOOTSTRAP PAREADO E TREESHAP (1min 30s)
    # =========================================================================
    s13 = new_from(
        13,
        notes="""[OBJETIVO] Apresentar em duas áreas a prova estatística via bootstrap pareado e a interpretação físico-química via TreeSHAP.
[TEMPO PREVISTO] 1 minuto e 30 segundos (Acumulado: 18:30).
[MENSAGEM CENTRAL] O bootstrap pareado comprova que o ganho da LSTM é estatisticamente nulo (IC contém 0), enquanto o TreeSHAP revela migração de crédito no Track A e convergência autônoma com o corte de 23% de etileno de Duval no Track B.
[FALA SUGERIDA] "Dividimos este slide nas duas validações que dão sustentação científica às conclusões. Na Área A, à esquerda, apresentamos os contrastes pareados via bootstrap estratificado com 5.000 réplicas no teste. No Track A, comparar C2 com C5a resulta em Delta F1 de +0,0056 com IC 95% de -0,0037 a +0,0150 (p=0,2384); como o intervalo contém zero, não há evidência estatística de ganho com a LSTM. O mesmo ocorre entre C1 e C2 (p=0,9956) e entre os ensembles C3, C4 e Wu et al. no Track B. A superioridade estatística inequívoca ocorre de C3 sobre Duval (+0,3500 em falhas ativas, p=0,0002) e sobre IEC (+0,4699). Na Área B, à direita, o TreeSHAP calculado em log-odds — com correlação de Spearman de 0,9541 entre sementes — explica os dois fenômenos: no Track B, o agrupamento por famílias confirma descargas regidas por acetileno e falhas térmicas por etileno/etano, e o efeito principal univariado de C₂H₄/TH cruza zero entre 20% e 25%, convergindo com o limiar canônico de 23% de Duval. No Track A, o gráfico inferior de migração de crédito prova que H₂_h14 e H₂_h28 disputam peso entre sementes mantendo soma constante (0,4734 ± 0,0131), confirmando redundância colinear — lembrando que SHAP mede atribuição no modelo, não causalidade."
[TRANSIÇÃO] "Com todas as evidências quantitativas e interpretáveis reunidas, fazemos a discussão crítica final."
[PERGUNTAS POSSÍVEIS] "Por que os limiares TreeSHAP de acetileno (~8%) e C2H4/C2H6 (2,97) ficaram abaixo da norma (13% e 4,0)?" -> Porque a base compilada contém casos rotulados como D1/D2 com menor teor de C2H2 e casos rotulados como T3 com mediana C2H4/C2H6 = 0,25; o XGBoost ajustou o limiar aditivo à distribuição real da base.""",
    )
    s13.add_header_and_footer("Rigor Estatístico e Explicabilidade", "Bootstrap Pareado (5.000×) e Interpretabilidade via TreeSHAP", "13 / 15", title_size=21.5)

    # Área A (Esquerda) — Bootstrap Pareado
    s13.add_rect(0.55, 1.12, 5.75, 5.82, fill="F8FAFE", line_color=C_BLUE, line_pt=1.3, rounded=True, radius_adj=2500, name="AreaABox")
    s13.add_textbox(0.70, 1.18, 5.45, 0.30, [{"text": "ÁREA A — SIGNIFICÂNCIA ESTATÍSTICA (Bootstrap Pareado, 5.000×)", "size": 11.0, "bold": True, "color": C_BLUE}], margins=(0, 0, 0, 0))
    s13.add_textbox(
        0.70, 1.48, 5.45, 0.56,
        [
            {
                "runs": [
                    r("Unidade e Regra: ", 10.0, True, False, C_NAVY),
                    r("reamostragem com reposição estratificada por classe nos mesmos índices do teste. Há diferença significativa somente quando o IC 95% exclui ΔF1 = 0.", 10.0),
                ]
            }
        ],
        margins=(0, 0, 0, 0),
    )

    s13.add_table(
        0.70,
        2.12,
        5.45,
        3.50,
        ["Contraste Pareado (A vs. B)", "ΔF1", "IC 95% Bootstrap", "p-valor", "Conclusão"],
        [
            ["Track A: C2 vs. C5a (LSTM fixo)", "+0,0056", "[-0,0037; +0,0150]", "0,2384", "NS (inclui 0)"],
            ["Track A: C1 vs. C5b (LSTM IGA)", "+0,0112", "[-0,0103; +0,0339]", "0,3308", "NS (inclui 0)"],
            ["Track A: C1 vs. C2 (IGA vs. 17)", "-0,0003", "[-0,0137; +0,0124]", "0,9956", "Equivalentes"],
            ["Track B: C3 vs. M3 (Wu et al.)", "+0,0141", "[-0,0054; +0,0368]", "0,1824", "NS (inclui 0)"],
            ["Track B: C3 vs. C4 (IGA vs. Fixo)", "+0,0127", "[-0,0211; +0,0490]", "0,4680", "NS (inclui 0)"],
            ["Track B (Falhas): C3 vs. Duval", "+0,3500", "[+0,2685; +0,4341]", "0,0002", "Sig. (exclui 0)"],
            ["Track B (Falhas): C3 vs. IEC", "+0,4699", "[+0,3920; +0,5553]", "0,0002", "Sig. (exclui 0)"],
        ],
        col_widths_frac=[0.35, 0.13, 0.25, 0.11, 0.16],
        header_fill=C_BLUE,
        font_size=8.6,
        header_size=8.9,
        highlight_rows={0: C_AMBER_LIGHT, 5: C_GREEN_LIGHT, 6: C_GREEN_LIGHT},
        name="TableBootstrap",
    )
    s13.add_textbox(
        0.70,
        5.74,
        5.45,
        1.08,
        [
            {
                "runs": [
                    r("Leitura Rigorosa: ", 10.2, True, False, C_NAVY),
                    r("Como os ICs de C2×C5a, C1×C5b e C3×M3×C4 contêm zero, não se pode afirmar superioridade da LSTM ou do IGA em F1-macro; o ganho comprovado do ML ocorre frente às normas (p = 0,0002).", 10.0),
                ],
                "line_spacing": 115,
            }
        ],
        fill=C_WHITE,
        line_color=C_BORDER,
        rounded=True,
        valign="ctr",
        margins=(0.08, 0.04, 0.08, 0.04),
        name="BoxBootNote",
    )

    # Área B (Direita) — TreeSHAP
    s13.add_rect(6.48, 1.12, 6.29, 5.82, fill="F6FBF9", line_color=C_GREEN, line_pt=1.3, rounded=True, radius_adj=2500, name="AreaBBox")
    s13.add_textbox(6.62, 1.18, 6.00, 0.30, [{"text": "ÁREA B — EXPLICABILIDADE FÍSICA (TreeSHAP em Log-Odds, ρ = 0,9541)", "size": 10.8, "bold": True, "color": C_GREEN}], margins=(0, 0, 0, 0))

    s13.add_image_fit(FIG_F6_DIR / "fig2_grouped_track_b.png", 6.62, 1.52, 3.05, 1.95, border_color=C_BORDER, name="FigShapGrouped")
    s13.add_image_fit(FIG_F6_DIR / "fig4_thresh_ratio_C2H4_TH_D1.png", 9.75, 1.52, 2.88, 1.95, border_color=C_BORDER, name="FigShapThresh")
    s13.add_image_fit(FIG_F6_DIR / "fig7_lstm_credit_migration.png", 6.62, 3.55, 3.25, 2.25, border_color=C_BORDER, name="FigCreditMigration")

    s13.add_textbox(
        9.96,
        3.55,
        2.68,
        2.25,
        [
            {"text": "ACHADOS TREESHAP", "size": 10.2, "bold": True, "color": C_NAVY, "space_after": 4},
            {"runs": [r("• Famílias (Track B): ", 9.5, True, False, C_GREEN), r("D1/D2 guiados por C₂H₂; PD por H₂ e CH₄/H₂; T1–T3 por etileno/etano.", 9.5)], "space_after": 4},
            {"runs": [r("• Limiar φ_ii=0 (D1): ", 9.5, True, False, C_BLUE), r("%C₂H₄ cruza zero em 20–25%, alinhado aos 23% de Duval.", 9.5)], "space_after": 4},
            {"runs": [r("• Migração (Track A): ", 9.5, True, False, C_ORANGE), r("peso oscila entre H₂_h14 e H₂_h28; soma fixa em 0,473 ± 0,013.", 9.5)]},
        ],
        fill=C_WHITE,
        line_color=C_BORDER,
        rounded=True,
        valign="ctr",
        margins=(0.07, 0.04, 0.07, 0.04),
        name="BoxShapInsights",
    )

    s13.add_textbox(
        6.62,
        5.88,
        6.01,
        0.94,
        [
            {
                "runs": [
                    r("Ressalva Científica (Correlação ≠ Causalidade): ", 9.8, True, False, C_RED),
                    r("Valores de Shapley quantificam a contribuição marginal aditiva das variáveis na margem log-odds do modelo XGBoost — não provam relação causal física direta e dividem crédito entre variáveis colineares.", 9.6),
                ],
                "line_spacing": 112,
            }
        ],
        fill=C_RED_LIGHT,
        line_color=C_RED,
        line_pt=1.0,
        rounded=True,
        valign="ctr",
        margins=(0.08, 0.03, 0.08, 0.03),
        name="BoxShapCaution",
    )
    slides.append(s13)

    # =========================================================================
    # SLIDE 14 — DISCUSSÃO CRÍTICA (1min 00s)
    # =========================================================================
    s14 = new_from(
        13,
        notes="""[OBJETIVO] Sintetizar com maturidade científica o que foi demonstrado, o que não foi possível demonstrar, as limitações metodológicas e as implicações para a área.
[TEMPO PREVISTO] 1 minuto (Acumulado: 19:30).
[MENSAGEM CENTRAL] Resultados negativos e limites empíricos bem explicados agregam mais valor à engenharia do que métricas superestimadas sem controle experimental.
[FALA SUGERIDA] "Na discussão crítica sintetizamos quatro pontos fundamentais. Primeiro, o que foi demonstrado: a redução de 53% de variáveis pelo IGA no Track A, a superioridade do XGBoost sobre regras normativas mesmo em falhas puras (0,800 vs. 0,450) e a coerência físico-química via TreeSHAP (convergência em 23% de etileno e estabilidade rho=0,954). Segundo, o que não foi possível demonstrar — e por quê: as projeções da LSTM não superaram o snapshot atual (p=0,238) devido à inércia física das séries (r > 0,997) e ausência de transições de falha nos rótulos; e nenhum modelo de árvores rompeu o teto empírico de 0,76–0,80 no Track B devido à sobreposição intrínseca de 5 gases e divergências de laudos históricos (mediana C₂H₄/C₂H₆ = 0,25 em T3 contra >4,0 da IEC). Terceiro, destacamos as implicações: métricas >99% na literatura frequentemente decorrem de testes minúsculos ou ausência de deduplicação (15,5% nesta base). Quarto, reconhecemos as limitações: fragmentação em dois tracks, previsões in-sample da LSTM no treino do Track A (apesar de o teste ser 100% cego) e CV 5×5 pós-seleção de atributos."
[TRANSIÇÃO] "Encerramos apresentando a conclusão direta e os próximos passos de pesquisa."
[PERGUNTAS POSSÍVEIS] "Se as previsões da LSTM no treino do Track A vieram de séries usadas para treinar a própria LSTM (sem out-of-fold), isso invalida o resultado?" -> Pelo contrário: esse cenário era otimista para a LSTM durante o treino; ainda assim, no teste independente (900 séries inéditas), a LSTM não gerou ganho estatístico sobre o modelo estático C5a, reforçando a conclusão de redundância temporal.""",
    )
    s14.add_header_and_footer("Discussão Crítica", "Achados Comprovados, Limites Empíricos e Limitações", "14 / 15", title_size=21.5)

    s14.add_card(
        0.55,
        1.15,
        5.98,
        2.75,
        "O Que Foi Demonstrado com Evidência",
        [
            {"runs": [r("• Parcimônia dimensional (Track A): ", 11.5, True, False, C_GREEN), r("IGA cortou 17 -> 8 atributos (−53%) preservando F1 ≈ 0,891.", 11.5)], "space_after": 5},
            {"runs": [r("• Superioridade sobre normas (Track B): ", 11.5, True, False, C_GREEN), r("C3 atingiu F1=0,800 em falhas ativas vs. 0,450 (Duval) e 0,330 (IEC) — p=0,0002.", 11.5)], "space_after": 5},
            {"runs": [r("• Coerência físico-química (TreeSHAP): ", 11.5, True, False, C_GREEN), r("ranking estável (ρ=0,954) e recuperação autônoma do corte de 23% de C₂H₄ de Duval.", 11.5)]},
        ],
        accent=C_GREEN,
        fill=C_GREEN_LIGHT,
        name="CardDisc1",
    )
    s14.add_card(
        6.78,
        1.15,
        5.98,
        2.75,
        "O Que NÃO Se Sustentou e Teto dos Dados",
        [
            {"runs": [r("• Hipótese temporal no diagnóstico pontual: ", 11.5, True, False, C_RED), r("LSTM não melhorou o XGBoost (ΔF1=+0,0056, IC inclui 0); r > 0,997 entre obs. e previsto.", 11.5)], "space_after": 5},
            {"runs": [r("• Superioridade do IGA sobre XGBoost fixo: ", 11.5, True, False, C_RED), r("C3, C4 e M3 são estatisticamente equivalentes (IC contém 0).", 11.5)], "space_after": 5},
            {"runs": [r("• Teto empírico (~0,76–0,80) no Track B: ", 11.5, True, False, C_RED), r("sobreposição térmica/elétrica e ruído multi-fonte (ex.: T3 com mediana C₂H₄/C₂H₆=0,25).", 11.5)]},
        ],
        accent=C_RED,
        fill=C_RED_LIGHT,
        name="CardDisc2",
    )

    s14.add_card(
        0.55,
        4.08,
        5.98,
        2.85,
        "Implicações Acadêmicas e Práticas",
        [
            {"runs": [r("• Crítica à literatura (>99%): ", 11.5, True, False, C_NAVY), r("acurácias infladas decorrem de conjuntos minúsculos (ex.: 25 testes) ou falta de deduplicação (15,5% de cópias/conflitos na base pública).", 11.5)], "space_after": 6},
            {"runs": [r("• Custo operacional em subestação: ", 11.5, True, False, C_NAVY), r("para diagnóstico do estado atual sob regime lento, o modelo estático C5a (F1=0,8855) dispensa treinar/manter a LSTM.", 11.5)]},
        ],
        accent=C_NAVY,
        fill=C_BG_BLUE,
        name="CardDisc3",
    )
    s14.add_card(
        6.78,
        4.08,
        5.98,
        2.85,
        "Limitações Metodológicas Reconhecidas",
        [
            {"runs": [r("• Fragmentação dos dados abertos: ", 11.5, True, False, C_AMBER), r("impediu testar LSTM + 7 classes IEC no mesmo equipamento; Track A tem rótulo constante por série.", 11.5)], "space_after": 5},
            {"runs": [r("• Escala e treino do Track A: ", 11.5, True, False, C_AMBER), r("base Katser já vem em ~10^-3 (sem ppm absoluto) e projeções de treino da LSTM não usaram esquema out-of-fold (embora teste seja 100% cego).", 11.5)], "space_after": 5},
            {"runs": [r("• Escopo de busca e CV 5×5: ", 11.5, True, False, C_AMBER), r("CV 5×5 rodou pós-seleção (custo computacional); comparação com Random/Bayesian Search fica para trabalhos futuros.", 11.5)]},
        ],
        accent=C_AMBER,
        fill=C_AMBER_LIGHT,
        name="CardDisc4",
    )
    slides.append(s14)

    # =========================================================================
    # SLIDE 15 — CONCLUSÕES E TRABALHOS FUTUROS (30s)
    # =========================================================================
    s15 = new_from(
        13,
        notes="""[OBJETIVO] Responder diretamente à pergunta de pesquisa, consolidar a principal contribuição e limitação, e apontar os trabalhos futuros sem introduzir novos resultados.
[TEMPO PREVISTO] 30 segundos (Acumulado: 20:00).
[MENSAGEM CENTRAL] O estudo comprova que previsões LSTM são redundantes para classificar o estado pontual sob alta inércia, enquanto IGA-XGBoost + TreeSHAP oferecem diagnóstico multiclasse superior às normas, parcimonioso e fisicamente auditável.
[FALA SUGERIDA] "Respondendo diretamente à pergunta da pesquisa: em séries de DGA com alta inércia temporal, o estado presente dos gases já concentra o sinal diagnóstico, tornando as projeções da LSTM redundantes para classificação pontual. A principal contribuição demonstrada é um protocolo experimental livre de vazamento que une auditoria rigorosa, redução de 53% de variáveis via IGA, superação estatística das normas tradicionais e explicabilidade físico-química validada por TreeSHAP. Como principal limitação, destacamos a ausência de bases públicas que combinem os sete gases e transições reais de falha ao longo do tempo. Como continuidade, o próximo passo natural é redirecionar a alta acurácia preditiva da LSTM (R² > 0,99) para estimar o tempo até o cruzamento de limites normativos (RUL), incorporar a taxa de variação gasosa (Δppm/Δt) e variáveis operacionais como carga e temperatura do óleo. Obrigado e fico à disposição da banca."
[TRANSIÇÃO] Encerramento da exposição oral (20:00) e abertura para arguição com slides de apoio prontos (18 a 23).
[PERGUNTAS POSSÍVEIS] Preparar-se para sabatina usando Slides de Apoio 18 a 23.""",
    )
    s15.add_header_and_footer("Conclusões e Perspectivas", "Conclusões Principais e Direcionamentos para Trabalhos Futuros", "15 / 15")

    s15.add_rect(0.55, 1.15, 12.22, 1.38, fill=C_NAVY, line_color=C_ORANGE, line_pt=2.0, rounded=True, radius_adj=2500, name="ConclusionBanner")
    s15.add_textbox(
        0.75,
        1.24,
        11.82,
        1.20,
        [
            {"text": "RESPOSTA DIRETA AO PROBLEMA DE PESQUISA", "size": 11.0, "bold": True, "color": "FFB899", "space_after": 4},
            {
                "text": "Para o diagnóstico pontual de transformadores sob evolução gradual, as concentrações presentes já retêm o poder discriminativo (projeções LSTM são estatisticamente redundantes: ΔF1=+0,0056, p=0,238). O valor da metodologia reside no protocolo livre de vazamento, na simplificação via IGA, na superação significativa de IEC/Duval e na validação físico-química via TreeSHAP.",
                "size": 13.5,
                "bold": True,
                "color": C_WHITE,
                "line_spacing": 118,
            },
        ],
        valign="ctr",
        margins=(0.04, 0.02, 0.04, 0.02),
        name="ConclusionBannerTxt",
    )

    s15.add_card(
        0.55,
        2.72,
        5.98,
        4.20,
        "Síntese Final (Contribuição e Limitação)",
        [
            {"runs": [r("• Contribuição Metodológica e XAI:\n", 12.5, True, False, C_GREEN), r("  Auditoria deduplicada (1.962 amostras), CustomSMOTE termodinâmico, testes pareados (5.000×) e TreeSHAP provando alinhamento físico (23% de etileno em D1).", 11.8)], "space_after": 10},
            {"runs": [r("• Contribuição Crítica (Resultado Negativo):\n", 12.5, True, False, C_BLUE), r("  Evidência rigorosa de que o acoplamento LSTM -> XGBoost não justifica seu custo extra para classificação do estado atual.", 11.8)], "space_after": 10},
            {"runs": [r("• Limitação Principal:\n", 12.5, True, False, C_RED), r("  Fragmentação das bases abertas (Track A sem CH₄/C₂H₆ e com rótulo constante; Track B sem histórico temporal).", 11.8)]},
        ],
        accent=C_BLUE,
        fill=C_BG_LIGHT,
        name="CardConcLeft",
    )

    s15.add_card(
        6.78,
        2.72,
        5.98,
        4.20,
        "Trabalhos Futuros (Seção 5.4)",
        [
            {"bullet": "1", "bullet_color": C_ORANGE, "runs": [r("Prognóstico de RUL e Alarmes: ", 12.0, True, False, C_NAVY), r("usar a precisão temporal da LSTM para prever quando os gases cruzarão os limites de atenção da IEC/IEEE.", 11.5)], "space_after": 8},
            {"bullet": "2", "bullet_color": C_ORANGE, "runs": [r("Taxa de Geração (Δppm/Δt): ", 12.0, True, False, C_NAVY), r("modelar velocidade diária/mensal de gaseificação como indicador de severidade.", 11.5)], "space_after": 8},
            {"bullet": "3", "bullet_color": C_ORANGE, "runs": [r("Variáveis Operacionais Externas: ", 12.0, True, False, C_NAVY), r("integrar carregamento elétrico, temperatura do topo do óleo e umidade.", 11.5)], "space_after": 8},
            {"bullet": "4", "bullet_color": C_ORANGE, "runs": [r("Bases Temporais Completas: ", 12.0, True, False, C_NAVY), r("validar o fluxo integrado quando houver séries de campo com os 7 gases e transições reais de falha.", 11.5)]},
        ],
        accent=C_ORANGE,
        fill=C_BG_ORANGE,
        name="CardConcFuture",
    )
    slides.append(s15)

    # =========================================================================
    # SLIDE 16 — ENCERRAMENTO INSTITUCIONAL (Base: template slide34.xml)
    # =========================================================================
    s16 = new_from(
        34,
        notes="""[OBJETIVO] Fechamento institucional formal da apresentação antes da arguição da banca.
[TEMPO PREVISTO] Fora dos 20 minutos (transição para arguição).
[MENSAGEM CENTRAL] Agradecimento à banca examinadora, orientadora e UNISINOS.""",
    )
    s16.add_textbox(
        1.50,
        4.95,
        10.33,
        1.45,
        [
            {"text": "OBRIGADO PELA ATENÇÃO", "size": 24.0, "bold": True, "color": C_WHITE, "align": "ctr", "space_after": 8},
            {"text": "Ândrio Gabriel Epping  |  Orientadora: Profa. Me. Janaína Conceição Sutil Lemos", "size": 14.5, "color": "DCE6FF", "align": "ctr", "space_after": 4},
            {"text": "Engenharia da Computação — UNISINOS (2026)  |  À disposição da Banca Examinadora", "size": 13.0, "color": "FFB899", "align": "ctr"},
        ],
        valign="ctr",
        margins=(0.05, 0.05, 0.05, 0.05),
        name="ClosingInfo",
    )
    slides.append(s16)

    # =========================================================================
    # SLIDE 17 — DIVISOR MATERIAL DE APOIO (Base: template slide23.xml)
    # =========================================================================
    s17 = new_from(
        23,
        notes="""[OBJETIVO] Separar visualmente os 15 slides principais da apresentação (20 min) dos slides reservados para arguição da banca (Apoio A1 a A6).""",
    )
    s17.add_textbox(
        1.00,
        2.35,
        11.33,
        2.40,
        [
            {"text": "MATERIAL DE APOIO", "size": 44.0, "bold": True, "color": C_WHITE, "align": "l", "space_after": 10},
            {"text": "Slides Técnicos Reservados para Arguição da Banca Examinadora", "size": 20.0, "bold": False, "color": "DCE6FF", "align": "l"},
        ],
        valign="ctr",
        margins=(0.0, 0.0, 0.0, 0.0),
        name="SupportDivider",
    )
    s17.add_rect(1.00, 4.95, 2.40, 0.07, fill=C_ORANGE, line_color=None, name="DividerOrangeBar")
    slides.append(s17)

    # =========================================================================
    # SLIDE 18 (APOIO A1) — PORTAS LSTM, BPTT, ADAM E CURVAS DE TREINO
    # =========================================================================
    s18 = new_from(
        13,
        notes="""[OBJETIVO - APOIO A1] Responder perguntas sobre o equacionamento interno da célula LSTM, fluxo aditivo do estado da célula C_t contra vanishing gradient, BPTT, Adam e diagnóstico das curvas de treino.
[OBSERVAÇÃO IMPORTANTE] Os gráficos de perda (MSE ~5×10⁻⁴ no treino) e distribuição de erros (MAE ~0,0026–0,0029 por gás) foram gerados na escala normalizada Min-Max [0, 1] (lstm_test_results.json, seed 42), enquanto a Tabela 1 do TCC reporta o erro invertido à escala original Katser (~10⁻⁵).""",
    )
    s18.add_header_and_footer("Material de Apoio — A1", "Portas da Célula LSTM, BPTT, Adam e Curvas de Treinamento", "Apoio 1 / 6")

    s18.add_card(
        0.55,
        1.14,
        5.85,
        3.12,
        "Equações Internas da Célula LSTM (Eqs. 4–9 do TCC)",
        [
            {"runs": [r("• Porta de Esquecimento:  ", 11.2, True, False, C_NAVY), r("f_t = σ(W_f [h_{t-1}, x_t] + b_f)", 11.2, False, True)], "space_after": 4},
            {"runs": [r("• Porta de Entrada e Candidato:  ", 11.2, True, False, C_NAVY), r("i_t = σ(W_i [h_{t-1}, x_t] + b_i);  C~_t = tanh(W_c [h_{t-1}, x_t] + b_c)", 11.0, False, True)], "space_after": 4},
            {"runs": [r("• Atualização Aditiva da Memória:  ", 11.2, True, False, C_TEAL_DARK), r("C_t = f_t * C_{t-1} + i_t * C~_t  (esteira de gradiente)", 11.2, True, True, C_TEAL_DARK)], "space_after": 4},
            {"runs": [r("• Porta de Saída e Estado Oculto:  ", 11.2, True, False, C_NAVY), r("o_t = σ(W_o [h_{t-1}, x_t] + b_o);  h_t = o_t * tanh(C_t)", 11.2, False, True)], "space_after": 6},
            {"runs": [r("• BPTT + Adam: ", 11.0, True, False, C_ORANGE), r("conexão aditiva de C_t evita vanishing gradient no BPTT em w=50 passos; Adam (β₁=0,9, β₂=0,999, lr=10^-3) adapta passos por parâmetro.", 10.5)]},
        ],
        accent=C_TEAL_DARK,
        fill=C_BG_TEAL,
        name="CardA1Eq",
    )

    s18.add_card(
        0.55,
        4.40,
        5.85,
        2.55,
        "Especificação Completa e Diagnóstico de Treino",
        [
            {"runs": [r("• Tensores exatos: ", 11.0, True, False, C_NAVY), r("Treino (660.450, 50, 4) | Val (116.550, 50, 4) | Teste (333.000, 50, 4).", 10.8)], "space_after": 5},
            {"runs": [r("• Por que val_loss < train_loss? ", 11.0, True, False, C_ORANGE), r("No treino, Dropout (0,20) desativa 20% das unidades elevando a perda média da época; na validação o Dropout é desligado (inferência determinística).", 10.5)], "space_after": 5},
            {"runs": [r("• Nota de escala nos gráficos à direita: ", 11.0, True, False, C_MUTED), r("gerados na escala Min-Max [0, 1] (seed 42: MAE_H2=0,00287, CO=0,00242); após inversão para a escala original Katser, MAE médio = 5,59×10^-5.", 10.2)]},
        ],
        accent=C_NAVY,
        fill=C_BG_LIGHT,
        name="CardA1Diag",
    )

    s18.add_image_fit(FIG_DIR / "lstm_loss_w50_seed42_norm.png", 6.58, 1.14, 6.18, 2.95, border_color=C_BORDER, name="FigLossCurve")
    s18.add_image_fit(FIG_DIR / "lstm_error_distribution_norm.png", 6.58, 4.24, 6.18, 2.70, border_color=C_BORDER, name="FigErrDist")
    slides.append(s18)

    # =========================================================================
    # SLIDE 19 (APOIO A2) — CROMOSSOMO DO IGA, CUSTOMSMOTE E CONVERGÊNCIA
    # =========================================================================
    s19 = new_from(
        13,
        notes="""[OBJETIVO - APOIO A2] Responder perguntas sobre os operadores genéticos do IGA (cruzamento BLX-alpha, mutação, viés inicial), preservação termodinâmica no CustomSMOTE e histórico de convergência.
[OBSERVAÇÃO DE RASTREABILIDADE] O JSON oficial `iga_evolution_track_b.json` atinge fitness CV = 0,7956 na geração 26 (46 gerações rodadas até parada antecipada), enquanto a figura `iga_convergence.png` salva em `tcc-unisinos/figuras/` registra o histórico de uma corrida anterior (~0,791).""",
    )
    s19.add_header_and_footer("Material de Apoio — A2", "Operadores Genéticos (IGA), CustomSMOTE e Convergência", "Apoio 2 / 6", title_size=21.5)

    s19.add_card(
        0.55,
        1.14,
        6.10,
        2.75,
        "Decodificação, Cruzamento BLX-α e Mutação (iga.py)",
        [
            {"runs": [r("• Decodificação afim: ", 11.0, True, False, C_NAVY), r("θ_k = lo_k + g_k(hi_k − lo_k)  (arredondado para int em max_depth, n_estimators, min_child_weight).", 10.5)], "space_after": 5},
            {"runs": [r("• Crossover (P_c = 0,90): ", 11.0, True, False, C_BLUE), r("Uniforme (p=0,5/bit) na máscara + BLX-α (α=0,50) nos 9 contínuos: sorteia em [min−0,5Δ, max+0,5Δ] com clip [0,1].", 10.5)], "space_after": 5},
            {"runs": [r("• Mutação: ", 11.0, True, False, C_ORANGE), r("Bit-flip (P_m=0,10/bit) nos binários + ruído Gaussiano N(0, 0,10²) com clip [0,1] nos contínuos.", 10.5)], "space_after": 5},
            {"runs": [r("• Fitness puro (Treino CV 5-fold): ", 11.0, True, False, C_GREEN), r("média de F1-macro em StratifiedKFold(5); cromossomos com < 2 bits ativos recebem 0,0.", 10.5)]},
        ],
        accent=C_NAVY,
        fill=C_BG_LIGHT,
        name="CardA2IGA",
    )

    s19.add_card(
        0.55,
        4.04,
        6.10,
        2.92,
        "Pipeline CustomSMOTE Termodinâmico (Track B)",
        [
            {"runs": [r("• Problema do SMOTE ingênuo: ", 11.0, True, False, C_RED), r("interpolar linearmente razões (ex.: C₂H₂/C₂H₄) viola a relação não linear entre gases: (a₁+a₂)/(b₁+b₂) ≠ ½(a₁/b₁ + a₂/b₂).", 10.5)], "space_after": 5},
            {"runs": [r("• Solução em 4 etapas (train_classification.py):\n", 11.0, True, False, C_GREEN), r("  1) Isola apenas os 5 gases brutos e aplica SMOTE (k=5);\n  2) Inverte Min-Max e exp(x)−1 para ppm físico;\n  3) Recalcula analiticamente todas as 14 razões, TH, TH₁ e %;\n  4) Reaplica scaler e seleciona colunas no ImbPipeline dentro da CV.", 10.5)]},
        ],
        accent=C_GREEN,
        fill=C_GREEN_LIGHT,
        name="CardA2SMOTE",
    )

    s19.add_image_fit(FIG_DIR / "iga_convergence.png", 6.82, 1.14, 5.95, 3.05, border_color=C_BORDER, name="FigIGAConv")

    s19.add_card(
        6.82,
        4.32,
        5.95,
        2.64,
        "Atributos Efetivamente Selecionados pelo IGA",
        [
            {"runs": [r("• Track A — C1 (8 de 17): ", 10.8, True, False, C_BLUE), r("obs_CO, ratio_H2_C2H2, ratio_C2H2_TH_k, ratio_C2H4_TH_k, lstm_h14_H2, lstm_h28_H2, lstm_h28_C2H4, lstm_h28_C2H2.", 10.2)], "space_after": 6},
            {"runs": [r("• Track B — C3 (16 de 19): ", 10.8, True, False, C_GREEN), r("descartou apenas 3 variáveis (C₂H₂ bruto, ratio_C2H2_CH4 e ratio_CH4_TH1), retendo todos os demais indicadores físicos.", 10.2)], "space_after": 6},
            {"runs": [r("• Fitness CV final (JSON): ", 10.5, True, False, C_MUTED), r("Track A = 0,8827 (39 ger.) | Track B = 0,7956 (46 ger.).", 10.2)]},
        ],
        accent=C_BLUE,
        fill=C_BG_BLUE,
        name="CardA2Feats",
    )
    slides.append(s19)

    # =========================================================================
    # SLIDE 20 (APOIO A3) — FUNÇÃO OBJETIVO DO XGBOOST E REGULARIZAÇÃO
    # =========================================================================
    s20 = new_from(
        13,
        notes="""[OBJETIVO - APOIO A3] Responder perguntas sobre a matemática do XGBoost (expansão de Taylor de 2ª ordem, gradiente/hessiana, peso das folhas, Gain) e explicar por que max_depth=12 não causou overfitting severo.
[MENSAGEM CENTRAL] A combinação de colsample_bytree=0,3433, reg_lambda=3,9425, gamma=1,4356 e min_child_weight=3 controla efetivamente a complexidade das árvores.""",
    )
    s20.add_header_and_footer("Material de Apoio — A3", "Objetivo do XGBoost, Ganho de Split e Controle de Overfitting", "Apoio 3 / 6", title_size=21.5)

    s20.add_card(
        0.55,
        1.14,
        6.00,
        3.20,
        "Formulação Matemática de 2ª Ordem (Eqs. 1–3 do TCC)",
        [
            {"runs": [r("• Objetivo regularizado na iteração t:\n", 11.2, True, False, C_NAVY), r("  L~^(t) ≈ Σ [g_i f_t(x_i) + ½ h_i f_t²(x_i)] + Ω(f_t)", 11.2, True, True, C_BLUE)], "space_after": 5},
            {"runs": [r("• Termo de penalização estrutural:\n", 11.2, True, False, C_NAVY), r("  Ω(f) = γ T_L + ½ λ Σ w_j² + α Σ |w_j|", 11.2, False, True)], "space_after": 5},
            {"runs": [r("• Derivadas Softmax (mlogloss):\n", 11.0, True, False, C_NAVY), r("  g_{i,c} = p_{i,c} − y_{i,c}   |   h_{i,c} = p_{i,c}(1 − p_{i,c}) em (0; 0,25]", 11.0, False, True)], "space_after": 5},
            {"runs": [r("• Peso ótimo foliar e Ganho de Split:\n", 11.0, True, False, C_NAVY), r("  w_j* = −sign(G_j) max(0, |G_j|−α) / (H_j + λ)\n  Gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] − γ", 10.5, False, True)]},
        ],
        accent=C_NAVY,
        fill=C_BG_LIGHT,
        name="CardA3Math",
    )

    s20.add_card(
        6.75,
        1.14,
        6.02,
        3.20,
        "Por Que max_depth=12 Não Gerou Overfitting em C3?",
        [
            {"runs": [r("1. Subamostragem de Colunas (colsample_bytree = 0,3433):\n", 11.2, True, False, C_GREEN), r("   Cada árvore enxerga apenas ~5 dos 16 atributos, descorrelacionando as árvores (efeito Random Forest).", 10.8)], "space_after": 6},
            {"runs": [r("2. Amortecimento Ridge L₂ Elevado (reg_lambda = 3,9425):\n", 11.2, True, False, C_BLUE), r("   λ elevado no denominador reduz drasticamente o peso w_j* de folhas com poucos exemplos.", 10.8)], "space_after": 6},
            {"runs": [r("3. Poda por Ganho e Suporte Mínimo (γ=1,4356; mcw=3):\n", 11.2, True, False, C_ORANGE), r("   Como h_{i,c} ≤ 0,25, min_child_weight=3 exige no mínimo ≥ 12 amostras incertas por folha; splits fracos são podados.", 10.8)]},
        ],
        accent=C_GREEN,
        fill=C_GREEN_LIGHT,
        name="CardA3Reg",
    )

    # Diagrama simples de exemplo de árvore XGBoost em DGA
    s20.add_rect(0.55, 4.50, 12.22, 2.45, fill=C_BG_BLUE, line_color=C_BLUE, line_pt=1.2, rounded=True, radius_adj=2500, name="TreeExampleBg")
    s20.add_textbox(0.72, 4.56, 11.80, 0.28, [{"text": "EXEMPLO CONCEITUAL DE DIVISÃO DE NÓ EM DGA (Log-Odds para Classe D1)", "size": 10.8, "bold": True, "color": C_NAVY}], margins=(0, 0, 0, 0))

    s20.add_textbox(
        4.60, 4.92, 3.80, 0.54,
        [{"text": "Nó Raiz: ratio_C2H2_TH < 0,13 ?", "size": 11.0, "bold": True, "color": C_WHITE, "align": "ctr"}],
        fill=C_NAVY, line_color=C_NAVY, rounded=True, valign="ctr",
    )
    s20.add_line_arrow(5.60, 5.46, 3.10, 5.82, color=C_RED, width_pt=1.8)
    s20.add_line_arrow(7.40, 5.46, 9.90, 5.82, color=C_GREEN, width_pt=1.8)
    s20.add_textbox(2.75, 5.44, 1.45, 0.26, [{"text": "Sim (< 13%)", "size": 9.5, "bold": True, "color": C_RED, "align": "ctr"}], margins=(0, 0, 0, 0))
    s20.add_textbox(8.85, 5.44, 1.45, 0.26, [{"text": "Não (≥ 13%)", "size": 9.5, "bold": True, "color": C_GREEN, "align": "ctr"}], margins=(0, 0, 0, 0))

    s20.add_textbox(
        1.20, 5.82, 3.80, 0.92,
        [{"text": "Folha: w* = -G_L / (H_L + λ) < 0\n(Reduz escore log-odds de D1)", "size": 10.5, "bold": True, "color": C_RED, "align": "ctr"}],
        fill=C_WHITE, line_color=C_RED, line_pt=1.3, rounded=True, valign="ctr",
    )
    s20.add_textbox(
        7.80, 5.82, 4.20, 0.92,
        [{"text": "Subnó: ratio_C2H4_TH < 0,23 ?\nSim -> w* > 0 (favorece D1) | Não -> favorece D2", "size": 10.5, "bold": True, "color": C_GREEN, "align": "ctr"}],
        fill=C_WHITE, line_color=C_GREEN, line_pt=1.3, rounded=True, valign="ctr",
    )
    slides.append(s20)

    # =========================================================================
    # SLIDE 21 (APOIO A4) — MATRIZES DE CONFUSÃO E MÉTRICAS POR CLASSE
    # =========================================================================
    s21 = new_from(
        13,
        notes="""[OBJETIVO - APOIO A4] Mostrar as matrizes de confusão completas e o desempenho detalhado por classe (Precision, Recall, F1, Suporte) nos Tracks A e B.
[NOTA DE RASTREABILIDADE] Na matriz `xgb_confusion_track_b.png` (que ilustra a Seção 4.3 do TCC), observam-se 6 D1→D2, 4 D2→D1 e 8 T2→T2 (F1_macro=0,779), correspondente à execução gráfica salva antes da atualização do JSON `iga_track_b_results.json` (seed 42: F1_macro=0,7883; 7 D1→D2, 4 D2→D1, 9 T2→T2). Documentamos ambos explicitamente.""",
    )
    s21.add_header_and_footer("Material de Apoio — A4", "Matrizes de Confusão Completas e Métricas por Classe", "Apoio 4 / 6")

    s21.add_image_fit(FIG_DIR / "xgb_confusion_track_a.png", 0.55, 1.12, 4.85, 3.30, border_color=C_BORDER, name="FigCMA")
    s21.add_image_fit(FIG_DIR / "xgb_confusion_track_b.png", 5.65, 1.12, 4.85, 3.30, border_color=C_BORDER, name="FigCMB")

    # Tabela detalhada por classe (Track A C1 + Track B C3 — seed 42 oficial JSON)
    s21.add_table(
        0.55,
        4.52,
        4.85,
        2.10,
        ["Track A (C1, s42)", "Prec.", "Rec.", "F1", "Sup."],
        [
            ["Normal", "0,989", "0,970", "0,979", "731"],
            ["PD", "0,895", "0,895", "0,895", "38"],
            ["D1", "0,855", "0,959", "0,904", "49"],
            ["T1", "0,744", "0,817", "0,779", "82"],
            ["Macro Avg (N=900)", "0,871", "0,910", "0,889", "900"],
        ],
        col_widths_frac=[0.38, 0.16, 0.16, 0.16, 0.14],
        header_fill=C_BLUE,
        font_size=9.0,
        header_size=9.2,
        highlight_rows={3: C_AMBER_LIGHT, 4: C_BG_BLUE},
        name="TableClsA",
    )

    s21.add_table(
        5.65,
        4.52,
        4.85,
        2.45,
        ["Track B (C3, s42)", "Prec.", "Rec.", "F1", "Sup."],
        [
            ["Normal", "0,979", "0,896", "0,936", "106"],
            ["PD", "0,813", "0,765", "0,788", "17"],
            ["D1", "0,810", "0,829", "0,819", "41"],
            ["D2", "0,737", "0,848", "0,789", "33"],
            ["T1", "0,538", "0,875", "0,667", "16"],
            ["T2", "0,692", "0,563", "0,621", "16"],
            ["T3", "0,921", "0,879", "0,899", "66"],
        ],
        col_widths_frac=[0.36, 0.16, 0.16, 0.16, 0.16],
        header_fill=C_GREEN,
        font_size=8.4,
        header_size=8.8,
        highlight_rows={4: C_AMBER_LIGHT, 5: C_AMBER_LIGHT, 6: C_GREEN_LIGHT},
        name="TableClsB",
    )

    s21.add_card(
        10.65,
        1.12,
        2.12,
        5.85,
        "Leitura Física",
        [
            {"runs": [r("• Track A (Normal <-> T1):\n", 9.8, True, False, C_BLUE), r("17 Normal->T1 e 8 T1->Normal. Aquecimentos leves (<300°C) geram pouco gás, próximo ao basal.", 9.4)], "space_after": 8},
            {"runs": [r("• Track B (D1 <-> D2):\n", 9.8, True, False, C_GREEN), r("Principal troca elétrica (10–11 casos): transição contínua entre centelhamento e arco.", 9.4)], "space_after": 8},
            {"runs": [r("• T1 e T2 (N=16):\n", 9.8, True, False, C_AMBER), r("Suporte baixo + fronteira térmica estreita explicam F1 moderado.", 9.4)], "space_after": 8},
            {"runs": [r("• Nota:\n", 9.4, True, False, C_MUTED), r("Tabela B mostra seed 42 de iga_track_b_results.json (F1=0,7883; 5-seed=0,7970).", 8.8)]},
        ],
        accent=C_NAVY,
        fill=C_BG_LIGHT,
        title_size=11.5,
        name="CardA4Obs",
    )
    slides.append(s21)

    # =========================================================================
    # SLIDE 22 (APOIO A5) — AUDITORIA NORMATIVA E PROJEÇÃO NO TRIÂNGULO DE DUVAL
    # =========================================================================
    s22 = new_from(
        13,
        notes="""[OBJETIVO - APOIO A5] Responder por que os métodos normativos tradicionais obtiveram F1 baixo no Track B, com prova quantitativa na base completa (1.962 amostras) e projeção visual dos erros sobre o Triângulo de Duval.
[MENSAGEM CENTRAL] Não houve erro de implementação normativa (validada por testes unitários canônicos); há duas causas reais: (1) normas não definem classe Normal (89% dos normais caem em T1 no Duval e 83,6% na IEC) e (2) os rótulos T3 da base possuem mediana C2H4/C2H6 = 0,25 contra >4,0 da IEC.""",
    )
    s22.add_header_and_footer("Material de Apoio — A5", "Auditoria Normativa (N=1.962) e Mapeamento no Triângulo de Duval", "Apoio 5 / 6")

    s22.add_table(
        0.55,
        1.14,
        6.80,
        3.55,
        ["Classe (N)", "IEC Concord.", "IEC ND (%)", "Duval Concord.", "Duval DT (%)", "Principal Desvio (Duval)"],
        [
            ["Normal (707)", "0,0%", "14,0%", "0,0%", "0,7%", "89,0% -> T1 (sem classe Normal)"],
            ["PD (111)", "14,4%", "80,2%", "14,4%", "4,5%", "52,3% -> T1 (%CH₄ med ≈ 97%)"],
            ["D1 (273)", "24,2%", "71,4%", "76,2%", "4,8%", "14,7% -> D2"],
            ["D2 (221)", "15,8%", "40,7%", "58,4%", "10,0%", "24,4% -> D1"],
            ["T1 (107)", "48,6%", "36,4%", "31,8%", "2,8%", "41,1% -> T2"],
            ["T2 (108)", "64,8%", "17,6%", "38,9%", "10,2%", "39,8% -> T3"],
            ["T3 (435)", "31,3%", "14,0%", "38,4%", "8,5%", "39,1% -> T2 (med. C₂H₄/C₂H₆=0,25)"],
        ],
        col_widths_frac=[0.18, 0.13, 0.12, 0.14, 0.12, 0.31],
        header_fill=C_NAVY,
        font_size=9.0,
        header_size=9.2,
        highlight_rows={0: C_RED_LIGHT, 1: C_AMBER_LIGHT, 6: C_AMBER_LIGHT},
        aligns=["l", "ctr", "ctr", "ctr", "ctr", "l"],
        name="TableNormAudit",
    )

    s22.add_card(
        0.55,
        4.84,
        6.80,
        2.10,
        "Conclusões da Auditoria Normativa (Seção 4.4)",
        [
            {"runs": [r("1. Escopo das Normas: ", 11.0, True, False, C_RED), r("Duval e IEC exigem falha prévia; aplicar sobre 707 amostras normais zera a classe Normal. No subconjunto só-falhas (N=189), Duval sobe para 0,4499 (90,5% cobertura, 53,8% acurácia) e IEC para 0,3301 (60,3% cobertura).", 10.4)], "space_after": 5},
            {"runs": [r("2. Anomalia de Rotulagem em T3: ", 11.0, True, False, C_AMBER), r("na base compilada, a mediana de C₂H₄/C₂H₆ em T3 é 0,25 (IEC exige >4,0). O ML alcança ~0,80 aprendendo a convenção real dos dados, não por falha física das normas.", 10.4)]},
        ],
        accent=C_ORANGE,
        fill=C_BG_ORANGE,
        name="CardA5Audit",
    )

    s22.add_image_fit(FIG_F6_DIR / "fig5_duval_triangle.png", 7.50, 1.14, 5.28, 5.80, border_color=C_BORDER, name="FigDuvalErrors")
    slides.append(s22)

    # =========================================================================
    # SLIDE 23 (APOIO A6) — SHAP COMPLEMENTAR E REFERÊNCIAS BIBLIOGRÁFICAS
    # =========================================================================
    s23 = new_from(
        13,
        notes="""[OBJETIVO - APOIO A6] Disponibilizar evidências complementares do TreeSHAP (Heatmap por classe, Beeswarm D1, Waterfall da amostra 11 onde o modelo corrige o laudo de campo D1 para D2 conforme a geometria canônica de Duval) e as referências bibliográficas essenciais.""",
    )
    s23.add_header_and_footer("Material de Apoio — A6", "Análises Complementares de TreeSHAP e Referências Bibliográficas", "Apoio 6 / 6")

    s23.add_image_fit(FIG_F6_DIR / "fig1_heatmap_track_b.png", 0.55, 1.12, 4.15, 3.15, border_color=C_BORDER, name="FigHeatmap")
    s23.add_image_fit(FIG_F6_DIR / "fig3_beeswarm_D1.png", 4.82, 1.12, 3.85, 3.15, border_color=C_BORDER, name="FigBeeswarmD1")
    s23.add_image_fit(FIG_F6_DIR / "fig6_waterfall_error_D1_to_D2_11.png", 8.80, 1.12, 3.96, 3.15, border_color=C_BORDER, name="FigWaterfall11")

    s23.add_textbox(
        0.55,
        4.34,
        12.22,
        0.58,
        [
            {
                "runs": [
                    r("Estudo de Caso Local (Amostra 11 — Waterfall à direita): ", 10.5, True, False, C_NAVY),
                    r("Rótulo original = D1, mas XGBoost previu D2 (margem f(x)=2,214 vs. E[f(X)]=0,153) devido a %C₂H₄ = 37,7% (+0,23, acima de 23%) e %C₂H₂ ≈ 42% (+0,31). Pela própria geometria canônica do Triângulo de Duval, essa amostra pertence à zona D2 -> o modelo identificou uma inconsistência no laudo de campo.", 10.2),
                ]
            }
        ],
        fill=C_BG_BLUE,
        line_color=C_BLUE,
        rounded=True,
        valign="ctr",
        margins=(0.08, 0.03, 0.08, 0.03),
        name="BoxWaterfallObs",
    )

    s23.add_card(
        0.55,
        5.02,
        12.22,
        1.95,
        "Referências Bibliográficas Fundamentais",
        [
            {"runs": [r("• Normas e DGA: ", 10.0, True, False, C_NAVY), r("IEC 60599 (2022); Duval (2002, IEEE Electr. Insul. Mag.); Ali et al. (2023, EPSR); Al-Sakini et al. (2025, Appl. Sci.).", 9.8)], "space_after": 3},
            {"runs": [r("• Bases de Dados: ", 10.0, True, False, C_NAVY), r("Katser et al. (2022, arXiv:2211.11013 — Track A); Li, Wang & Song (2019, IEEE Access — Track B); Ibrahim et al. (2018, DGALab).", 9.8)], "space_after": 3},
            {"runs": [r("• Algoritmos e XAI: ", 10.0, True, False, C_NAVY), r("Hochreiter & Schmidhuber (1997, LSTM); Chen & Guestrin (2016, XGBoost); Wu et al. (2021, IGA-XGBoost, Front. Energy Res.); Wang et al. (2023, IEEE TDEI); Lundberg & Lee (2017, SHAP); Lundberg et al. (2020, TreeSHAP, Nat. Mach. Intell.).", 9.8)]},
        ],
        accent=C_NAVY,
        fill=C_BG_LIGHT,
        title_size=11.5,
        header_h=0.30,
        name="CardRefs",
    )
    slides.append(s23)

    return slides


# ---------------------------------------------------------------------------
# Empacotamento OpenXML completo preservando slideMaster, temas e SVG nativos
# ---------------------------------------------------------------------------
def build_notes_xml(notes_text: str) -> bytes:
    """Gera ppt/notesSlides/notesSlideN.xml válido para PowerPoint."""
    notes = etree.Element(
        qn(NS_P, "notes"),
        nsmap={"a": NS_A, "r": NS_R, "p": NS_P},
    )
    cSld = etree.SubElement(notes, qn(NS_P, "cSld"))
    spTree = etree.SubElement(cSld, qn(NS_P, "spTree"))

    nvGrp = etree.SubElement(spTree, qn(NS_P, "nvGrpSpPr"))
    etree.SubElement(nvGrp, qn(NS_P, "cNvPr"), id="1", name="")
    etree.SubElement(nvGrp, qn(NS_P, "cNvGrpSpPr"))
    etree.SubElement(nvGrp, qn(NS_P, "nvPr"))
    grpPr = etree.SubElement(spTree, qn(NS_P, "grpSpPr"))
    xf = etree.SubElement(grpPr, qn(NS_A, "xfrm"))
    etree.SubElement(xf, qn(NS_A, "off"), x="0", y="0")
    etree.SubElement(xf, qn(NS_A, "ext"), cx="0", cy="0")
    etree.SubElement(xf, qn(NS_A, "chOff"), x="0", y="0")
    etree.SubElement(xf, qn(NS_A, "chExt"), cx="0", cy="0")

    # Placeholder da miniatura do slide
    sp_img = etree.SubElement(spTree, qn(NS_P, "sp"))
    nv1 = etree.SubElement(sp_img, qn(NS_P, "nvSpPr"))
    etree.SubElement(nv1, qn(NS_P, "cNvPr"), id="2", name="Slide Image Placeholder 1")
    cNv1 = etree.SubElement(nv1, qn(NS_P, "cNvSpPr"))
    etree.SubElement(cNv1, qn(NS_A, "spLocks"), noGrp="1", noRot="1", noChangeAspect="1")
    nvPr1 = etree.SubElement(nv1, qn(NS_P, "nvPr"))
    etree.SubElement(nvPr1, qn(NS_P, "ph"), type="sldImg", idx="2")
    etree.SubElement(sp_img, qn(NS_P, "spPr"))

    # Placeholder do texto das notas
    sp_body = etree.SubElement(spTree, qn(NS_P, "sp"))
    nv2 = etree.SubElement(sp_body, qn(NS_P, "nvSpPr"))
    etree.SubElement(nv2, qn(NS_P, "cNvPr"), id="3", name="Notes Placeholder 2")
    cNv2 = etree.SubElement(nv2, qn(NS_P, "cNvSpPr"))
    etree.SubElement(cNv2, qn(NS_A, "spLocks"), noGrp="1")
    nvPr2 = etree.SubElement(nv2, qn(NS_P, "nvPr"))
    etree.SubElement(nvPr2, qn(NS_P, "ph"), type="body", sz="quarter", idx="3")
    etree.SubElement(sp_body, qn(NS_P, "spPr"))

    txBody = etree.SubElement(sp_body, qn(NS_P, "txBody"))
    etree.SubElement(txBody, qn(NS_A, "bodyPr"))
    etree.SubElement(txBody, qn(NS_A, "lstStyle"))

    for line in notes_text.strip().splitlines():
        p = etree.SubElement(txBody, qn(NS_A, "p"))
        r_el = etree.SubElement(p, qn(NS_A, "r"))
        rPr = etree.SubElement(r_el, qn(NS_A, "rPr"), lang="pt-BR", dirty="0")
        t = etree.SubElement(r_el, qn(NS_A, "t"))
        t.text = line
        etree.SubElement(p, qn(NS_A, "endParaRPr"), lang="pt-BR", dirty="0")

    clrMap = etree.SubElement(notes, qn(NS_P, "clrMapOvr"))
    etree.SubElement(clrMap, qn(NS_A, "masterClrMapping"))
    return etree.tostring(notes, xml_declaration=True, encoding="UTF-8", standalone="yes")


def build_notes_rels_xml(slide_num: int) -> bytes:
    rels = etree.Element(qn(NS_REL, "Relationships"), nsmap={None: NS_REL})
    rel1 = etree.SubElement(rels, qn(NS_REL, "Relationship"))
    rel1.set("Id", "rId1")
    rel1.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster")
    rel1.set("Target", "../notesMasters/notesMaster1.xml")
    rel2 = etree.SubElement(rels, qn(NS_REL, "Relationship"))
    rel2.set("Id", "rId2")
    rel2.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide")
    rel2.set("Target", f"../slides/slide{slide_num}.xml")
    return etree.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone="yes")


def save_presentation(tpl_path: Path, out_path: Path, slides: list[OpenXMLSlide]) -> None:
    tmp_out = out_path.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(tpl_path, "r") as zin, zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as zout:
        # 1. Descobrir apenas as mídias realmente referenciadas pelos layouts/masters mantidos e pelos 23 slides
        referenced_media: set[str] = set()
        for item in zin.infolist():
            rname = item.filename
            if not rname.endswith(".rels"):
                continue
            if rname.startswith("ppt/slides/") or rname.startswith("ppt/notesSlides/") or rname == "ppt/_rels/presentation.xml.rels":
                continue
            base_dir = "" if rname == "_rels/.rels" else posixpath.dirname(posixpath.dirname(rname))
            rel_root = etree.fromstring(zin.read(rname))
            for rel in rel_root:
                tgt = rel.get("Target", "")
                if rel.get("TargetMode", "") == "External" or tgt.startswith("http"):
                    continue
                resolved = posixpath.normpath(posixpath.join(base_dir, tgt)).lstrip("/")
                if resolved.startswith("ppt/media/"):
                    referenced_media.add(resolved)

        for sl in slides:
            for rel in sl.rels_root:
                tgt = rel.get("Target", "")
                if rel.get("TargetMode", "") == "External" or tgt.startswith("http"):
                    continue
                resolved = posixpath.normpath(posixpath.join("ppt/slides", tgt)).lstrip("/")
                if resolved.startswith("ppt/media/"):
                    referenced_media.add(resolved)

        # Copiar partes do template exceto slides/notes originais, presentation.xml(.rels), [Content_Types].xml e mídias não usadas
        skip_prefixes = ("ppt/slides/", "ppt/notesSlides/")
        skip_exact = {
            "[Content_Types].xml",
            "ppt/presentation.xml",
            "ppt/_rels/presentation.xml.rels",
        }
        for item in zin.infolist():
            name = item.filename
            if name in skip_exact or any(name.startswith(p) for p in skip_prefixes):
                continue
            if name.startswith("ppt/media/") and name not in referenced_media:
                continue
            zout.writestr(item.filename, zin.read(item.filename))

        # 2. Atualizar presentation.xml e presentation.xml.rels
        pres_root = etree.fromstring(zin.read("ppt/presentation.xml"))
        pres_rels = etree.fromstring(zin.read("ppt/_rels/presentation.xml.rels"))

        # Remover relacionamentos antigos de slides
        for rel in list(pres_rels):
            if rel.get("Type", "").endswith("/slide"):
                pres_rels.remove(rel)

        # Calcular próximo rId livre em presentation.xml.rels
        used_rids = set()
        for rel in pres_rels:
            rid = rel.get("Id", "")
            if rid.startswith("rId"):
                try:
                    used_rids.add(int(rid[3:]))
                except ValueError:
                    pass
        next_rid = max(used_rids or {0}) + 1

        sldIdLst = pres_root.find("p:sldIdLst", NSMAP_SLD)
        for child in list(sldIdLst):
            sldIdLst.remove(child)

        # 3. Atualizar [Content_Types].xml
        ct_root = etree.fromstring(zin.read("[Content_Types].xml"))
        for ov in list(ct_root):
            pn = ov.get("PartName", "")
            if pn.startswith("/ppt/slides/slide") or pn.startswith("/ppt/notesSlides/notesSlide"):
                ct_root.remove(ov)
            elif pn.startswith("/ppt/media/") and pn.lstrip("/") not in referenced_media:
                ct_root.remove(ov)

        # Garantir Default extensions para jpg/jpeg/png/svg caso precisem
        existing_exts = {el.get("Extension", "").lower() for el in ct_root if etree.QName(el).localname == "Default"}
        for ext, ctype in [("png", "image/png"), ("jpeg", "image/jpeg"), ("jpg", "image/jpeg"), ("svg", "image/svg+xml")]:
            if ext not in existing_exts:
                etree.SubElement(ct_root, qn(NS_CT, "Default"), Extension=ext, ContentType=ctype)

        # 4. Escrever os 23 slides, rels, mídias extras e notesSlides
        base_sld_id = 256
        for sl in slides:
            s_num = sl.slide_num
            # Link slide -> notesSlide
            sl.add_rel(
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide",
                f"../notesSlides/notesSlide{s_num}.xml",
            )

            # Slide XML & Rels
            zout.writestr(
                f"ppt/slides/slide{s_num}.xml",
                etree.tostring(sl.root, xml_declaration=True, encoding="UTF-8", standalone="yes"),
            )
            zout.writestr(
                f"ppt/slides/_rels/slide{s_num}.xml.rels",
                etree.tostring(sl.rels_root, xml_declaration=True, encoding="UTF-8", standalone="yes"),
            )

            # Notes XML & Rels
            zout.writestr(f"ppt/notesSlides/notesSlide{s_num}.xml", build_notes_xml(sl.notes_text))
            zout.writestr(f"ppt/notesSlides/_rels/notesSlide{s_num}.xml.rels", build_notes_rels_xml(s_num))

            # Extra media
            for m_name, m_bytes in sl.extra_media:
                zout.writestr(f"ppt/media/{m_name}", m_bytes)

            # Registrar slide em presentation.xml & rels
            rid = f"rId{next_rid}"
            next_rid += 1
            rel_el = etree.SubElement(pres_rels, qn(NS_REL, "Relationship"))
            rel_el.set("Id", rid)
            rel_el.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide")
            rel_el.set("Target", f"slides/slide{s_num}.xml")

            sldId = etree.SubElement(sldIdLst, qn(NS_P, "sldId"))
            sldId.set("id", str(base_sld_id + s_num))
            sldId.set(qn(NS_R, "id"), rid)

            # Registrar Content Types
            etree.SubElement(
                ct_root,
                qn(NS_CT, "Override"),
                PartName=f"/ppt/slides/slide{s_num}.xml",
                ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
            )
            etree.SubElement(
                ct_root,
                qn(NS_CT, "Override"),
                PartName=f"/ppt/notesSlides/notesSlide{s_num}.xml",
                ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml",
            )

        # Atualizar propriedades docProps/core.xml se existente (metadados do TCC)
        zout.writestr(
            "ppt/presentation.xml",
            etree.tostring(pres_root, xml_declaration=True, encoding="UTF-8", standalone="yes"),
        )
        zout.writestr(
            "ppt/_rels/presentation.xml.rels",
            etree.tostring(pres_rels, xml_declaration=True, encoding="UTF-8", standalone="yes"),
        )
        zout.writestr(
            "[Content_Types].xml",
            etree.tostring(ct_root, xml_declaration=True, encoding="UTF-8", standalone="yes"),
        )

    shutil.move(str(tmp_out), str(out_path))


def main() -> None:
    if not TEMPLATE_PPTX.exists():
        raise FileNotFoundError(f"Modelo institucional não encontrado: {TEMPLATE_PPTX}")

    cadeia_png = ensure_cadeia_causal_png()
    with zipfile.ZipFile(TEMPLATE_PPTX, "r") as tpl_zip:
        slides = build_slides(tpl_zip, cadeia_png)

    save_presentation(TEMPLATE_PPTX, OUTPUT_PPTX, slides)
    print(f"Apresentação gerada com sucesso: {OUTPUT_PPTX} ({len(slides)} slides)")


if __name__ == "__main__":
    main()
