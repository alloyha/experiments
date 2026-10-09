"""
Plantei.com.br — Plant Scraper
================================
CSS classes confirmadas via inspeção ao vivo:
  - Lista: ul.list-product  /  .categoria-offer__products
  - Item:  li.list-product__items
  - Preço: .current-price
  - Nome:  .item__description  ou  <a> dentro do card

Estrutura de preço observada no HTML:
  R$ 78,90  → preço "de" (original / riscado)
  R$ 69,43  → preço "por" (preço atual de venda)
  R$ 62,49  → à vista com desconto (pix)
  2x de R$ 34,72 → parcelamento

Install:
    pip install requests playwright pandas beautifulsoup4 lxml
    playwright install chromium

Uso:
    python plantei_scraper.py                    # todas as páginas
    python plantei_scraper.py --pages 3         # teste rápido
    python plantei_scraper.py --headed          # ver o browser
    python plantei_scraper.py --debug           # salvar HTML renderizado
    python plantei_scraper.py --output ./data   # pasta de saída
    python plantei_scraper.py --only-available  # só plantas disponíveis
    python plantei_scraper.py --enrich          # busca preço das esgotadas
"""

import re
import json
import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("plantei")

BASE_URL     = "https://www.plantei.com.br"
CATEGORY_URL = f"{BASE_URL}/plantas-naturais"

SOLDOUT_RE = re.compile(
    r"esgotado|sold[\s_-]?out|indispon[íi]vel|sem\s*estoque|avise.me", re.I
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9",
}

# ──────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────

@dataclass
class PriceOption:
    label: str
    price: float          # preço "por" (de venda)
    old_price: float | None = None   # preço "de" (riscado)
    pix_price: float | None = None   # preço à vista / pix


@dataclass
class Plant:
    name: str
    url: str
    status: str           # "available" | "soldout"
    price_options: list[PriceOption] = field(default_factory=list)
    image_url: str = ""

    @property
    def min_price(self):
        return min((o.price for o in self.price_options), default=None)


# ──────────────────────────────────────────────
# Price helpers
# ──────────────────────────────────────────────

def _parse_brl(text: str) -> float | None:
    """'R$ 29,90' → 29.90"""
    clean = re.sub(r"[^\d,]", "", (text or "")).replace(",", ".").strip(".")
    try:
        v = float(clean)
        return v if v > 0 else None
    except ValueError:
        return None


def _extract_prices(text: str) -> list[float]:
    """Extrai todos os valores R$ de um bloco de texto, na ordem em que aparecem."""
    return [
        v for m in re.findall(r"R\$\s*[\d.]+,\d{2}", text)
        if (v := _parse_brl(m))
    ]


def _parse_price_block(text: str) -> PriceOption | None:
    """
    Interpreta o bloco de texto de preço de um card.

    Padrão Plantei/Tray:
      prices[0] = preço "de" (original, riscado)   — pode não existir
      prices[1] = preço "por" (preço atual)
      prices[2] = preço à vista / pix

    Se só houver 1 preço: é o preço atual sem desconto.
    """
    prices = _extract_prices(text)
    if not prices:
        return None

    if len(prices) == 1:
        return PriceOption(label="Unidade", price=prices[0])

    if len(prices) == 2:
        # Pode ser [de, por] ou [por, pix] — heurística: maior = original
        if prices[0] > prices[1]:
            return PriceOption(label="Unidade", price=prices[1], old_price=prices[0])
        else:
            return PriceOption(label="Unidade", price=prices[0], pix_price=prices[1])

    # 3+ preços: [de, por, pix, ...]
    old_p = prices[0]
    por_p = prices[1]
    pix_p = prices[2] if len(prices) >= 3 else None
    return PriceOption(label="Unidade", price=por_p, old_price=old_p, pix_price=pix_p)


# ══════════════════════════════════════════════
# Parser HTML — usa seletores CSS confirmados
# ══════════════════════════════════════════════

# Seletores de card em ordem de prioridade (confirmados via inspeção)
CARD_SELECTORS = [
    "li.list-product__items",
    "ul.list-product > li",
    ".categoria-offer__products li",
    "li.item",
]

# Seletores de preço dentro do card
PRICE_SELECTORS = [
    ".current-price",
    "[class*='current-price']",
    "[class*='preco-por']",
    "[class*='price-current']",
    "[itemprop='price']",
]

# Seletores de preço antigo (riscado)
OLD_PRICE_SELECTORS = [
    "[class*='preco-de']",
    "[class*='old-price']",
    "[class*='price-old']",
    ".de", "s", "del",
]

# Seletores de nome
NAME_SELECTORS = [
    ".item__description",
    "[class*='item__description']",
    "[class*='product-name']",
    "[class*='nome']",
    "h2", "h3",
]


def parse_card(card) -> Plant | None:
    """Parseia um único card de produto (tag BeautifulSoup)."""

    # ── Nome ─────────────────────────────────
    name = ""
    for sel in NAME_SELECTORS:
        el = card.select_one(sel)
        if el:
            name = el.get_text(strip=True)
            if name:
                break
    if not name:
        # Fallback: link com título razoável
        for a in card.find_all("a", href=True):
            t = a.get_text(" ", strip=True)
            if len(t) > 5 and not re.search(
                r"comprar|whats|avise|ver tudo|carrinho|adicionar", t, re.I
            ):
                name = t
                break

    if not name or len(name) < 3:
        return None

    # Limpa ruído de preço que pode ter vazado para o nome
    name = re.split(r"R\$", name)[0].strip()
    name = re.sub(r"\s{2,}", " ", name).strip()
    if not name or len(name) < 3:
        return None

    # ── URL ──────────────────────────────────
    url = ""
    link = card.find("a", href=re.compile(r"^(?!javascript|#|mailto|tel|whatsapp)"))
    if link:
        url = link["href"]
        if url and not url.startswith("http"):
            url = BASE_URL + url.lstrip("/")

    # ── Imagem ───────────────────────────────
    image_url = ""
    img = card.find("img")
    if img:
        image_url = (
            img.get("data-src") or img.get("data-lazy") or
            img.get("data-original") or img.get("src") or ""
        )
        if "empty.png" in image_url:
            image_url = ""

    # ── Status ───────────────────────────────
    card_text = card.get_text(" ", strip=True)
    status = "soldout" if SOLDOUT_RE.search(card_text) else "available"

    # ── Preços ───────────────────────────────
    price_options: list[PriceOption] = []

    # Variantes via <select>
    select = card.find("select")
    if select:
        for opt in select.find_all("option"):
            if not opt.get("value") or opt["value"] == "0":
                continue
            label_text = opt.get_text(strip=True)
            p = _parse_brl(opt.get("data-price") or opt.get("data-valor") or "")
            if not p:
                vals = _extract_prices(label_text)
                p = vals[0] if vals else None
            lbl = re.sub(r"R\$.*", "", label_text).strip(" —-")
            if lbl and p:
                price_options.append(PriceOption(label=lbl, price=p))

    # Preço único (sem variantes)
    if not price_options:
        # Tenta seletor específico de preço atual
        price_el = None
        old_el   = None
        for sel in PRICE_SELECTORS:
            price_el = card.select_one(sel)
            if price_el:
                break
        for sel in OLD_PRICE_SELECTORS:
            old_el = card.select_one(sel)
            if old_el:
                break

        if price_el:
            por_p = _parse_brl(price_el.get_text())
            old_p = _parse_brl(old_el.get_text()) if old_el else None
            if por_p:
                # Pega preço à vista do texto restante
                remaining_text = card_text
                all_prices = _extract_prices(remaining_text)
                pix_p = None
                if len(all_prices) >= 3:
                    pix_p = all_prices[2]
                price_options.append(PriceOption(
                    label="Unidade", price=por_p, old_price=old_p, pix_price=pix_p
                ))
        else:
            # Fallback: inferir do texto completo do card
            opt = _parse_price_block(card_text)
            if opt:
                price_options.append(opt)

    return Plant(
        name=name,
        url=url,
        status=status,
        price_options=price_options,
        image_url=image_url,
    )


def parse_html(html: str) -> list[Plant]:
    """Parseia o HTML renderizado e retorna lista de plantas."""
    soup = BeautifulSoup(html, "lxml")
    plants: list[Plant] = []
    seen: set[str] = set()

    cards = []
    for sel in CARD_SELECTORS:
        cards = soup.select(sel)
        if cards:
            log.info("  Seletor '%s' → %d cards", sel, len(cards))
            break

    if not cards:
        log.warning("  Nenhum seletor CSS funcionou. Tentando heurística...")
        # Heurística: qualquer <li> com preço + link + imagem
        for el in soup.find_all("li"):
            text = el.get_text()
            if (
                re.search(r"R\$\s*[\d.]+,\d{2}", text) and
                el.find("a", href=True) and
                el.find("img")
            ):
                cards.append(el)
        log.info("  Heurística encontrou %d candidatos", len(cards))

    for card in cards:
        plant = parse_card(card)
        if not plant or plant.url in seen:
            continue
        seen.add(plant.url)
        plants.append(plant)

    return plants


# ══════════════════════════════════════════════
# Playwright — browser headless
# ══════════════════════════════════════════════

JS_PAGES = r"""() => {
    const nums = [];
    document.querySelectorAll('ul.pagination a, .paginacao a, [class*="paginat"] a').forEach(a => {
        const m = (a.href || '').match(/pg=(\d+)/);
        if (m) nums.push(parseInt(m[1]));
        const t = a.textContent.trim();
        if (/^\d+$/.test(t) && parseInt(t) < 500) nums.push(parseInt(t));
    });
    return nums;
}"""

# Scroll feito em Python para evitar "context destroyed" ao navegar com JS async
SCROLL_STEPS = 4
SCROLL_STEP_MS = 300


async def _scroll_page(page: Page) -> None:
    """Scroll em passos via Python — evita 'context destroyed' do JS async."""
    try:
        height = await page.evaluate("document.body.scrollHeight")
        for i in range(1, SCROLL_STEPS + 1):
            await page.evaluate(f"window.scrollTo(0, {int(height * i / SCROLL_STEPS)})")
            await page.wait_for_timeout(SCROLL_STEP_MS)
        await page.evaluate("window.scrollTo(0, 0)")
    except Exception:
        pass  # página pode ter navegado — ignora e continua


async def scrape_page(page: Page, url: str, debug: bool = False) -> list[Plant]:
    await page.goto(url, wait_until="networkidle", timeout=35000)

    # Scroll por etapas em Python (~1.5s) para ativar lazy-load
    await _scroll_page(page)
    await page.wait_for_timeout(500)

    html = await page.content()

    if debug:
        pg_num = re.search(r"pg=(\d+)", url)
        n = pg_num.group(1) if pg_num else "1"
        fname = f"debug_page_{n}.html"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(html)
        log.info("  [debug] HTML salvo → %s", fname)

    return parse_html(html)


async def scrape_all(max_pages=0, headed=False, debug=False,
                     concurrency: int = 4) -> list[Plant]:
    """
    Scrapes all listing pages using a worker-queue:
    N workers continuously pull page numbers from a queue and process them
    until the queue is empty — no waiting for slow siblings in a batch.
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headed)
        context = await browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="pt-BR",
            viewport={"width": 1440, "height": 900},
        )

        # ── Probe page 1 to discover total pages ─────────────────────────────
        log.info("Detectando páginas...")
        probe = await context.new_page()
        page1_plants = await scrape_page(probe, CATEGORY_URL, debug=debug)
        nums = await probe.evaluate(JS_PAGES)
        await probe.close()

        total_pages = max(nums, default=1) if nums else 1
        if max_pages and max_pages < total_pages:
            total_pages = max_pages
        log.info("Total: %d páginas | %d workers", total_pages, concurrency)

        # ── Build queue with remaining page numbers ───────────────────────────
        results: dict[int, list[Plant]] = {1: page1_plants}
        queue: asyncio.Queue[int] = asyncio.Queue()
        for pn in range(2, total_pages + 1):
            await queue.put(pn)

        done = asyncio.Event()
        completed = 0

        # ── Worker: pulls page numbers until queue empty ──────────────────────
        async def worker(worker_id: int) -> None:
            nonlocal completed
            while True:
                try:
                    pn = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                url = f"{CATEGORY_URL}?pg={pn}"
                page = await context.new_page()
                try:
                    plants = await scrape_page(page, url, debug=debug)
                    results[pn] = plants
                    completed += 1
                    log.info("  [worker %d] pág %d/%d → %d cards (%d feitas)",
                             worker_id, pn, total_pages, len(plants), completed)
                except Exception as exc:
                    results[pn] = []
                    log.warning("  [worker %d] pág %d falhou: %s", worker_id, pn, exc)
                finally:
                    await page.close()
                    queue.task_done()

        # ── Launch N workers concurrently ─────────────────────────────────────
        await asyncio.gather(*[worker(i + 1) for i in range(concurrency)])
        await browser.close()

    # ── Merge results in page order, deduplicating by URL ────────────────────
    all_plants: list[Plant] = []
    seen_urls: set[str] = set()
    for pn in sorted(results):
        new = 0
        for p in results[pn]:
            if p.url not in seen_urls:
                seen_urls.add(p.url)
                all_plants.append(p)
                new += 1
        if new:
            log.info("  Pág %d → %d novas (total: %d)", pn, new, len(all_plants))

    return all_plants



# ══════════════════════════════════════════════
# Enriquecimento — busca preço das esgotadas
# ══════════════════════════════════════════════

# Seletores do container do produto na página de detalhe (Tray Commerce)
# Importante: limitar ao container evita capturar preços do banner de frete
PRODUCT_CONTAINER_SELECTORS = [
    ".product-info",
    ".product-detail",
    ".product-main",
    "[class*='product-detail']",
    "[class*='product-info']",
    "#product",
    "main",        # fallback largo mas ainda melhor que body inteiro
]

# JS que extrai preço diretamente do DOM renderizado, dentro do container do produto
JS_ENRICH_PRICE = r"""() => {
    // Encontra o container do produto (evita banners e nav)
    const containerSels = [
        '.product-info', '.product-detail', '.product-main',
        '[class*="product-detail"]', '[class*="product-info"]',
        '#product', 'main'
    ];
    let container = null;
    for (const s of containerSels) {
        container = document.querySelector(s);
        if (container) break;
    }
    if (!container) container = document.body;

    const result = { price: null, old_price: null, pix_price: null, variants: [] };

    // 1. meta itemprop (mais confiável, não sofre com layout)
    const metaPrice = document.querySelector('meta[itemprop="price"]');
    if (metaPrice) {
        result.price = parseFloat(metaPrice.getAttribute('content'));
    }

    // 2. JSON-LD schema.org
    if (!result.price) {
        document.querySelectorAll('script[type="application/ld+json"]').forEach(s => {
            try {
                const d = JSON.parse(s.textContent);
                const offers = d.offers || (d['@graph'] || []).flatMap(x => x.offers || []);
                const offer = Array.isArray(offers) ? offers[0] : offers;
                if (offer && offer.price) result.price = parseFloat(offer.price);
            } catch(e) {}
        });
    }

    // 3. Seletores visuais DENTRO do container
    if (!result.price) {
        const priceSels = [
            '.current-price', '[class*="current-price"]', '[class*="preco-por"]',
            '[class*="price-current"]', '[itemprop="price"]'
        ];
        for (const s of priceSels) {
            const el = container.querySelector(s);
            if (el) {
                const txt = el.textContent.replace(/[^\d,]/g, '').replace(',', '.');
                const v = parseFloat(txt);
                if (v > 0) { result.price = v; break; }
            }
        }
    }

    // 4. Preço antigo (riscado) dentro do container
    const oldSels = [
        '[class*="preco-de"]', '[class*="old-price"]', '[class*="price-old"]',
        '.de', 's', 'del'
    ];
    for (const s of oldSels) {
        const el = container.querySelector(s);
        if (el) {
            const txt = el.textContent.replace(/[^\d,]/g, '').replace(',', '.');
            const v = parseFloat(txt);
            if (v > 0) { result.old_price = v; break; }
        }
    }

    // 5. Preço à vista / pix — busca texto "à vista" dentro do container
    const avistaSels = [
        '[class*="pix"]', '[class*="avista"]', '[class*="vista"]',
        '[class*="cash"]', '[class*="desconto"]'
    ];
    for (const s of avistaSels) {
        const el = container.querySelector(s);
        if (el) {
            const txt = el.textContent.replace(/[^\d,]/g, '').replace(',', '.');
            const v = parseFloat(txt);
            if (v > 0 && v !== result.price) { result.pix_price = v; break; }
        }
    }

    // 6. Variantes via <select> dentro do container
    const sel = container.querySelector('select');
    if (sel) {
        sel.querySelectorAll('option').forEach(opt => {
            if (!opt.value || opt.value === '0') return;
            const p = parseFloat(opt.dataset.price || opt.dataset.valor || '');
            const label = opt.textContent.trim().replace(/R\$.*/, '').trim();
            if (label && p > 0) result.variants.push({ label, price: p });
        });
    }

    return result;
}"""


async def _parse_enrich_data(data: dict) -> list[PriceOption]:
    """Converts raw JS price data into PriceOption list."""
    price_options: list[PriceOption] = []
    if data.get("variants"):
        for v in data["variants"]:
            if v.get("price") and v.get("label"):
                price_options.append(PriceOption(
                    label=v["label"], price=float(v["price"])
                ))
    if not price_options and data.get("price"):
        price_options.append(PriceOption(
            label="Unidade",
            price=float(data["price"]),
            old_price=float(data["old_price"]) if data.get("old_price") else None,
            pix_price=float(data["pix_price"]) if data.get("pix_price") else None,
        ))
    return price_options


async def enrich_soldout(plants: list[Plant], delay: float = 0.3,
                         headed: bool = False, concurrency: int = 5) -> list[Plant]:
    """
    Fetches prices for sold-out plants using a worker-queue:
    N workers pull plants from a queue and visit their pages until empty.
    Workers never wait for each other — a fast page immediately unblocks
    the next item, unlike batch parallelism where the slowest in a batch
    delays all subsequent work.
    """
    targets = [p for p in plants if p.status == "soldout" and not p.price_options]
    if not targets:
        log.info("Nenhuma esgotada sem preço para enriquecer.")
        return plants

    total = len(targets)
    log.info("Enriquecendo %d esgotadas — %d workers", total, concurrency)

    # ── Populate queue ────────────────────────────────────────────────────────
    queue: asyncio.Queue[tuple[int, Plant]] = asyncio.Queue()
    for i, plant in enumerate(targets, 1):
        await queue.put((i, plant))

    completed = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headed)
        context = await browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="pt-BR",
            viewport={"width": 1280, "height": 900},
        )

        # ── Worker: pulls (idx, plant) until queue empty ──────────────────────
        async def fetch_price(plant: Plant, attempt: int) -> list[PriceOption]:
            """Open a tab, grab price, close tab. Returns empty list on failure."""
            page = await context.new_page()
            try:
                await page.goto(plant.url, wait_until="domcontentloaded", timeout=20000)
                try:
                    await page.wait_for_selector(
                        '.current-price, [itemprop="price"], meta[itemprop="price"], '
                        '[class*="preco-por"], [class*="current-price"]',
                        timeout=6000,
                    )
                except Exception:
                    pass
                data = await page.evaluate(JS_ENRICH_PRICE)
                return await _parse_enrich_data(data)
            finally:
                await page.close()

        async def worker(worker_id: int) -> None:
            nonlocal completed
            while True:
                try:
                    idx, plant = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                price_options: list[PriceOption] = []
                last_exc: Exception | None = None

                # Up to 2 attempts per item (1 retry on failure)
                for attempt in range(1, 3):
                    try:
                        price_options = await fetch_price(plant, attempt)
                        break   # success — stop retrying
                    except Exception as exc:
                        last_exc = exc
                        if attempt < 2:
                            log.warning("  [w%d] %d/%d tentativa %d falhou (%s), tentando novamente…",
                                        worker_id, idx, total, attempt, type(exc).__name__)
                            await asyncio.sleep(1.5)  # brief pause before retry
                        else:
                            log.warning("  [w%d] %d/%d ERRO após %d tentativas: %s — %s",
                                        worker_id, idx, total, attempt,
                                        plant.name[:30], exc)

                if price_options:
                    plant.price_options = price_options
                    completed += 1
                    log.info("  [w%d] %d/%d ✓ R$ %.2f  %s",
                             worker_id, idx, total,
                             price_options[0].price, plant.name[:40])
                elif last_exc is None:
                    # Loaded fine but genuinely no price found on the page
                    log.info("  [w%d] %d/%d ✗ sem preço  %s",
                             worker_id, idx, total, plant.name[:40])

                queue.task_done()
                if delay > 0:
                    await asyncio.sleep(delay)

        # ── Launch N workers ──────────────────────────────────────────────────
        await asyncio.gather(*[worker(i + 1) for i in range(concurrency)])
        await browser.close()

    enriched = sum(1 for p in targets if p.price_options)
    log.info("Enriquecimento concluído: %d/%d com preço.", enriched, total)
    return plants

# ══════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════

COLUMNS = ["name", "url", "status", "variant", "price_brl", "old_price_brl", "pix_price_brl", "image_url"]


def _build_rows(plants: list[Plant]) -> list[dict]:
    rows = []
    for p in plants:
        if p.price_options:
            for o in p.price_options:
                rows.append({
                    "name": p.name, "url": p.url, "status": p.status,
                    "variant": o.label, "price_brl": o.price,
                    "old_price_brl": o.old_price, "pix_price_brl": o.pix_price,
                    "image_url": p.image_url,
                })
        else:
            rows.append({
                "name": p.name, "url": p.url, "status": p.status,
                "variant": "", "price_brl": None, "old_price_brl": None,
                "pix_price_brl": None, "image_url": p.image_url,
            })
    return rows


def _col_widths(df: pd.DataFrame) -> dict:
    """Calcula largura ideal para cada coluna do Excel."""
    widths = {}
    for col in df.columns:
        max_data = df[col].astype(str).map(len).max()
        widths[col] = min(max(max_data, len(col)) + 2, 80)
    return widths


def save_outputs(plants: list[Plant], output_dir: str = ".") -> tuple[str, str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path  = out / f"plantei_plants_{ts}.csv"
    json_path = out / f"plantei_plants_{ts}.json"
    xlsx_path = out / f"plantei_plants_{ts}.xlsx"

    rows = _build_rows(plants)
    df_all  = pd.DataFrame(rows, columns=COLUMNS)
    df_avail = df_all[df_all["status"] == "available"].reset_index(drop=True)
    df_sold  = df_all[df_all["status"] == "soldout"].reset_index(drop=True)

    # ── CSV ──────────────────────────────────────────────────────────────────
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("CSV  → %s  (%d linhas)", csv_path, len(df_all))

    # ── JSON ─────────────────────────────────────────────────────────────────
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "scraped_at":   datetime.now().isoformat(),
            "source_url":   CATEGORY_URL,
            "total_plants": len(plants),
            "available":    int((df_all["status"] == "available").sum()),
            "soldout":      int((df_all["status"] == "soldout").sum()),
            "plants": [{
                "name": p.name, "url": p.url, "status": p.status,
                "image_url": p.image_url, "min_price_brl": p.min_price,
                "price_options": [asdict(o) for o in p.price_options],
            } for p in plants],
        }, f, ensure_ascii=False, indent=2)
    log.info("JSON → %s", json_path)

    # ── Excel (3 abas) ───────────────────────────────────────────────────────
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in [
                ("Todas", df_all),
                ("Disponíveis", df_avail),
                ("Esgotadas", df_sold),
            ]:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]

                # Largura das colunas
                for col, width in _col_widths(df).items():
                    col_idx = df.columns.get_loc(col) + 1
                    ws.column_dimensions[ws.cell(1, col_idx).column_letter].width = width

                # Cabeçalho em negrito
                from openpyxl.styles import Font, PatternFill, Alignment
                header_fill = PatternFill("solid", fgColor="2D6A4F")
                for cell in ws[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = header_fill
                    cell.alignment = Alignment(horizontal="center")

                # Linhas alternadas
                from openpyxl.styles import PatternFill as PF
                fill_even = PF("solid", fgColor="F0F7F4")
                for row in ws.iter_rows(min_row=2):
                    if row[0].row % 2 == 0:
                        for cell in row:
                            cell.fill = fill_even

                # Congelar cabeçalho
                ws.freeze_panes = "A2"

        log.info("XLSX → %s  (3 abas: Todas / Disponíveis / Esgotadas)", xlsx_path)
    except ImportError:
        log.warning("openpyxl não instalado — Excel ignorado. Instale: pip install openpyxl")
        xlsx_path = None

    return str(csv_path), str(json_path), str(xlsx_path) if xlsx_path else ""


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Scraper de plantas — plantei.com.br")
    parser.add_argument("--pages",           type=int,   default=0,   help="Máx páginas (0=todas)")
    parser.add_argument("--concurrency",     type=int,   default=4,   help="Tabs paralelas no scraping principal (padrão: 4)")
    parser.add_argument("--enrich-delay",    type=float, default=0.3, help="Pausa entre lotes de enrich (padrão: 0.3s)")
    parser.add_argument("--enrich-workers",  type=int,   default=5,   help="Tabs paralelas no enrich (padrão: 5)")
    parser.add_argument("--headed",         action="store_true",      help="Mostrar o browser")
    parser.add_argument("--debug",          action="store_true",      help="Salvar HTML renderizado")
    parser.add_argument("--only-available", action="store_true",      help="Exportar só plantas disponíveis")
    parser.add_argument("--enrich",         action="store_true",      help="Buscar preço das esgotadas sem preço")
    parser.add_argument("--output",         type=str,   default=".",  help="Pasta de saída")
    args = parser.parse_args()

    # ── Scraping principal ───────────────────────────────────────────────────
    plants = asyncio.run(scrape_all(
        max_pages=args.pages,
        headed=args.headed,
        debug=args.debug,
        concurrency=args.concurrency,
    ))

    if not plants:
        print("\n⚠  Nenhuma planta encontrada.")
        print("   Tente: python plantei_scraper.py --debug --pages 1\n")
        return

    # ── Enriquecimento das esgotadas ─────────────────────────────────────────
    if args.enrich:
        plants = asyncio.run(enrich_soldout(
            plants,
            delay=args.enrich_delay,
            headed=args.headed,
            concurrency=args.enrich_workers,
        ))

    # ── Filtro --only-available ──────────────────────────────────────────────
    export_plants = [p for p in plants if p.status == "available"] if args.only_available else plants
    if args.only_available:
        log.info("--only-available: exportando %d plantas disponíveis", len(export_plants))

    # ── Exportar ─────────────────────────────────────────────────────────────
    csv_path, json_path, xlsx_path = save_outputs(export_plants, args.output)

    available = sum(1 for p in plants if p.status == "available")
    soldout   = sum(1 for p in plants if p.status == "soldout")
    no_price  = sum(1 for p in plants if not p.price_options)
    enriched  = sum(1 for p in plants if p.status == "soldout" and p.price_options)

    sep = "=" * 54
    print(f"""
{sep}
  Total scraped      : {len(plants)}
  Disponíveis        : {available}
  Esgotadas          : {soldout}
    com preço        : {enriched}
    sem preço        : {soldout - enriched}
  Exportadas         : {len(export_plants)}{"  (só disponíveis)" if args.only_available else ""}
  CSV                : {csv_path}
  JSON               : {json_path}
  Excel              : {xlsx_path or "(openpyxl não instalado)"}
{sep}
""")


if __name__ == "__main__":
    main()

