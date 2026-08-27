import json, os, re
from collections import defaultdict
from pathlib import Path

# ── Entity registry: grain text → entity_id ───────────────────────────────────────────
GRAIN_TO_ENTITY: dict[str, str] = {
    "fatura":           "invoice",
    "assinatura":       "subscription",
    "cliente":          "customer",
    "oportunidade":     "opportunity",
    "usuário":          "user",
    "usuario":          "user",
    "lead":             "lead",
    "pedido":           "order",
    "ticket":           "ticket",
    "colaborador":      "employee",
    "deploy":           "deployment",
    "incidente":        "incident",
    "campanha":         "campaign",
    "sku":              "sku",
    "contrato":         "contract",
    "mês":              "month",
    "mes":              "month",
    "período":          "period",
    "periodo":          "period",
    "dia":              "day",
    "semana":           "week",
    "cohort":           "cohort",
    "resposta":         "survey_response",
    "sessão":           "session",
    "sessao":           "session",
    "release":          "release",
    "execução":         "pipeline_run",
    "execucao":         "pipeline_run",
    "dataset":          "dataset",
    "auditoria":        "audit",
    "vulnerabilidade":  "vulnerability",
    "controle":         "control",
    "processo":         "process",
    "conta":            "account",
    "movimento":        "movement",
    "vaga":             "job",
    "contratação":      "hire",
    "contratacao":      "hire",
    "entrega":          "delivery",
    "pedido de compra": "purchase_order",
    "meta":             "target",
    "kpi":              "kpi",
    "feature":          "feature",
    "oferta":           "offer",
    "trial":            "trial",
    "ação":             "action",
    "acao":             "action",
    "item":             "item",
    "carrinho":         "cart",
    "checkout":         "checkout",
    "recurso":          "resource",
    "operação":         "operation",
    "operacao":         "operation",
    "linha de venda":   "sale_line",
    "agente":           "agent",
    "produto":          "product",
    "requisição":       "request",
    "requisicao":       "request",
    "serviço":          "service",
    "servico":          "service",
}

ENTITY_PK: dict[str, str] = {
    "invoice":         "invoice_id",
    "subscription":    "subscription_id",
    "customer":        "customer_id",
    "opportunity":     "opportunity_id",
    "user":            "user_id",
    "lead":            "lead_id",
    "order":           "order_id",
    "ticket":          "ticket_id",
    "employee":        "employee_id",
    "deployment":      "deployment_id",
    "incident":        "incident_id",
    "campaign":        "campaign_id",
    "sku":             "sku_id",
    "contract":        "contract_id",
    "month":           "month_date",
    "period":          "period_date",
    "day":             "date_day",
    "week":            "date_week",
    "cohort":          "cohort_id",
    "survey_response": "response_id",
    "session":         "session_id",
    "release":         "release_id",
    "pipeline_run":    "run_id",
    "dataset":         "dataset_id",
    "audit":           "audit_id",
    "vulnerability":   "vulnerability_id",
    "control":         "control_id",
    "process":         "process_id",
    "account":         "account_id",
    "movement":        "movement_id",
    "job":             "job_id",
    "hire":            "hire_id",
    "delivery":        "delivery_id",
    "purchase_order":  "po_id",
    "target":          "target_id",
    "kpi":             "kpi_id",
    "feature":         "feature_id",
    "offer":           "offer_id",
    "trial":           "trial_id",
    "action":          "action_id",
    "item":            "item_id",
    "cart":            "cart_id",
    "checkout":        "checkout_id",
    "resource":        "resource_id",
    "operation":       "operation_id",
    "sale_line":       "sale_line_id",
    "agent":           "agent_id",
    "product":         "product_id",
    "request":         "request_id",
    "service":         "service_id",
}


def _infer_entity_id(grain: str) -> str | None:
    return GRAIN_TO_ENTITY.get(grain.lower().strip())


def _infer_derivation_type(aggregation: str, deps: list) -> str:
    """base = computable from one source; derived = depends on other metrics or complex expr."""
    if deps or aggregation == "custom":
        return "derived"
    return "base"


def _infer_metric_type(aggregation: str) -> str:
    """Semantic category, orthogonal to derivation_type."""
    if aggregation == "ratio":
        return "ratio"
    return "scalar"


def _infer_metric_kind(aggregation: str, deps: list) -> str:
    # kept for backward compatibility; prefer derivation_type + metric_type
    if deps or aggregation == "custom":
        return "derived"
    if aggregation == "ratio":
        return "ratio"
    return "base"


def _infer_additivity(aggregation: str, unit: str) -> str:
    if aggregation in ("ratio", "custom"):
        return "non_additive"
    if aggregation in ("avg", "max", "min", "count_distinct"):
        return "semi_additive"
    _non_additive_units = {"%", "x", "score", "meses", "dias", "horas", "minutos",
                           "BRL/mês", "unidades/tempo", "unidades/recurso", "BRL/pessoa",
                           "deploys/período", "BRL/dia", "eventos/usuário",
                           "sessões/usuário", "pedidos/cliente", "tickets/cliente",
                           "incidentes/unidade", "reclamações/unidade", "defeitos/unidade"}
    if unit in _non_additive_units:
        return "non_additive"
    return "additive"


def _infer_time_grain(supported_periods: list) -> str:
    for p in supported_periods:
        if p in ("dia", "day"):
            return "day"
        if p in ("semana", "week"):
            return "week"
    return "month"


catalog = []

# ── Computational dependency edges ────────────────────────────────────────────
DEPS: dict = {
    "finance.arr":                   [("finance.mrr", "computational")],
    "finance.arpu":                  [("finance.net_revenue", "denominator")],
    "finance.arpa":                  [("finance.net_revenue", "denominator")],
    "finance.gross_margin":          [("finance.net_revenue", "denominator")],
    "finance.gross_profit":          [("finance.net_revenue", "computational")],
    "finance.ebitda_margin":         [("finance.ebitda", "computational"), ("finance.net_revenue", "denominator")],
    "finance.operating_margin":      [("finance.net_revenue", "denominator")],
    "finance.net_margin":            [("finance.net_revenue", "denominator")],
    "finance.free_cash_flow":        [("finance.operating_cash_flow", "computational")],
    "finance.runway":                [("finance.cash_balance", "computational"), ("finance.burn_rate", "denominator")],
    "finance.ltv_cac_ratio":         [("finance.ltv", "computational"), ("finance.cac", "denominator")],
    "finance.cac_payback_months":    [("finance.cac", "computational")],
    "finance.mrr_growth_rate":       [("finance.mrr", "computational")],
    "finance.revenue_growth_rate":   [("finance.revenue", "computational")],
    "finance.net_revenue_retention": [("finance.mrr", "computational"),
                                      ("customer.expansion_mrr", "computational"),
                                      ("customer.churned_mrr", "computational"),
                                      ("customer.contraction_mrr", "computational")],
    "marketing.mql_to_sql_rate":     [("marketing.mql_volume", "computational"), ("marketing.sql_volume", "denominator")],
    "marketing.cost_per_mql":        [("marketing.mql_volume", "denominator")],
    "marketing.cost_per_sql":        [("marketing.sql_volume", "denominator")],
    "marketing.cost_per_lead":       [("marketing.leads_generated", "denominator")],
    "product.dau_mau_ratio":         [("product.dau", "computational"), ("product.mau", "denominator")],
    "sales.pipeline_coverage":       [("sales.pipeline_value", "computational"), ("sales.sales_target", "denominator")],
    "sales.sales_velocity":          [("sales.win_rate", "computational"),
                                      ("sales.average_deal_size", "computational"),
                                      ("sales.sales_cycle_length", "denominator")],
    "customer.revenue_churn_rate":   [("customer.churned_mrr", "computational"),
                                      ("customer.contraction_mrr", "computational")],
    "customer.gross_revenue_retention": [("customer.churned_mrr", "computational"),
                                         ("customer.contraction_mrr", "computational")],
}

_TABLE_COL  = re.compile(r'\b([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)\b')
_SQL_TOKENS = frozenset({"nullif", "count", "sum", "avg", "median", "min", "max",
                          "distinct", "where", "between", "not", "and", "or", "in"})


def _extract_lineage(expr: str, source_table) -> dict:
    """Parse table.column references from a formula expression into a lineage block."""
    seen: dict = {}
    cols = []
    for tbl, col in _TABLE_COL.findall(expr.lower()):
        if tbl in _SQL_TOKENS:
            continue
        if tbl not in seen:
            seen[tbl] = tbl
        role = "date_key" if col.endswith(("_at", "_date")) else (
               "filter"   if col in ("status", "type", "category", "severity") else "numerator")
        cols.append({"source": tbl, "table": tbl, "column": col, "role": role})
    if not cols:
        return {}
    sources = list(seen)
    joins = [
        {"left": sources[i], "right": sources[i + 1], "type": "INNER",
         "on": f"{sources[i]}.id = {sources[i + 1]}.{sources[i]}_id"}
        for i in range(len(sources) - 1)
    ]
    return {"columns": cols, "joins": joins}


def _make_quality(aggregation: str, unit: str, data_quality: str, refresh: str) -> list:
    """Build a structured quality contract for a metric."""
    age = "26" if "daily" in refresh or "diário" in refresh else "168"
    rules: list = [{"dimension": "freshness", "rule": "max_age_hours",
                    "threshold": age, "severity": "error"}]
    if aggregation in ("sum", "count", "count_distinct"):
        rules.append({"dimension": "completeness", "rule": "null_rate",
                      "threshold": "0.01", "severity": "warning"})
    if aggregation == "ratio":
        rules.append({"dimension": "completeness", "rule": "denominator_not_zero_rate",
                      "threshold": "0.99", "severity": "warning"})
        if unit == "%":
            rules.append({"dimension": "accuracy", "rule": "value_in_range_0_to_2",
                          "threshold": None, "severity": "error"})
    if data_quality == "audited":
        rules.append({"dimension": "consistency", "rule": "cross_system_reconciliation",
                      "threshold": "0.001", "severity": "error"})
    return rules

def add(id, name, aliases, department, tags, description, expression, aggregation, grain, unit,
        dimensions, default_period, supported_periods, owner_team, owner_contact,
        status="active", refresh="daily", quality="audited", version="1.0",
        when=None, related=None, questions=None, benchmarks=None,
        source_table=None, cost="low", cacheable=True):
    formula = {"expression": expression, "language": "pseudocode"}
    if source_table:
        formula["source_table"] = source_table
    catalog.append({
        "id": id, "name": name, "aliases": aliases, "department": department,
        "tags": tags, "description": description, "formula": formula,
        "aggregation": aggregation, "grain": grain, "unit": unit,
        "dimensions": [{"name": n, "join_path": p} for n,p in dimensions],
        "default_period": default_period, "supported_periods": supported_periods,
        "owner": {"team": owner_team, "contact": owner_contact},
        "status": status, "refresh_frequency": refresh,
        "data_quality": quality, "version": version,
        "change_log": [{"date": "2024-01-15", "change": "Métrica adicionada ao catálogo canônico v1.0."}],
        "dependencies": [{"depends_on": dep, "type": t} for dep, t in DEPS.get(id, [])],
        "quality": _make_quality(aggregation, unit, quality, refresh),
        "lineage": _extract_lineage(expression, source_table),
        "entity_id": _infer_entity_id(grain),
        "display_grain": grain,
        "metric_kind": _infer_metric_kind(aggregation, DEPS.get(id, [])),
        "derivation_type": _infer_derivation_type(aggregation, DEPS.get(id, [])),
        "metric_type": _infer_metric_type(aggregation),
        "additivity": _infer_additivity(aggregation, unit),
        "time_grain": _infer_time_grain(supported_periods),
        "superseded_by": None,
        "deprecated_at": None,
        "deprecation_reason": None,
        "usage_context": {
            "when_to_use": when or description,
            "related_metrics": related or [],
            "example_questions": questions or [f"Qual é {name.lower()}?"],
            "benchmarks": benchmarks or {}
        },
        "access": {
            "endpoint": f"/api/metrics/{id.replace('.', '/')}",
            "requires_permission": f"{id.split('.')[0]}.read",
            "execution_cost": cost, "cacheable": cacheable
        }
    })

# Helper for standard dimensions by domain
dims = {
    "finance": [("segmento_cliente","customer.segment"),("produto","product.id"),("plano","subscription.plan_id")],
    "sales": [("vendedor","opportunity.owner_id"),("produto","opportunity.product_id"),("canal","opportunity.source")],
    "marketing": [("canal","campaign.channel"),("campanha","campaign.id"),("segmento","customer.segment")],
    "product": [("plataforma","event.platform"),("plano","account.plan_id"),("cohort","user.signup_cohort")],
    "customer": [("segmento","customer.segment"),("plano","account.plan_id"),("cohort","customer.cohort")],
    "support": [("canal","ticket.channel"),("categoria","ticket.category"),("prioridade","ticket.priority")],
    "engineering": [("time","deployment.team"),("serviço","deployment.service"),("repositório","deployment.repository")],
    "hr": [("departamento","employee.department"),("cargo","employee.role"),("localidade","employee.location")],
    "operations": [("unidade","operation.site"),("processo","operation.process"),("categoria","operation.category")],
    "supply_chain": [("produto","product.id"),("fornecedor","supplier.id"),("localidade","warehouse.location")],
    "data": [("pipeline","pipeline.id"),("domínio","dataset.domain"),("criticidade","dataset.criticality")],
    "ecommerce": [("canal","order.channel"),("categoria","product.category"),("produto","product.id")],
    "security": [("serviço","security.service"),("severidade","incident.severity"),("origem","event.source")],
    "strategy": [("unidade_negócio","business_unit.id"),("região","business_unit.region"),("segmento","customer.segment")],
    "quality": [("processo","quality.process"),("produto","quality.product"),("fornecedor","quality.supplier")]
}

def metric(domain, key, name, desc, expr, agg, grain, unit, aliases=None, tags=None, **kw):
    add(f"{domain}.{key}", name, aliases or [name.lower()], kw.pop("department", domain.title()),
        tags or [domain], desc, expr, agg, grain, unit, dims.get(domain, []),
        kw.pop("default_period","mês corrente"), kw.pop("supported_periods",["dia","mês","trimestre","ano","customizado"]),
        kw.pop("owner_team",domain.title()), kw.pop("owner_contact",f"{domain}@empresa.com"), **kw)

# FINANCE (25)
finance_metrics = [
("revenue","Receita Total","Soma da receita faturada no período.","SUM(invoice.net_amount)","sum","fatura","BRL"),
("gross_revenue","Receita Bruta","Receita antes de descontos, devoluções e impostos dedutíveis.","SUM(invoice.gross_amount)","sum","fatura","BRL"),
("net_revenue","Receita Líquida","Receita após descontos, devoluções e abatimentos aplicáveis.","SUM(invoice.net_amount)","sum","fatura","BRL"),
("mrr","Receita Recorrente Mensal (MRR)","Receita recorrente ativa normalizada para base mensal.","SUM(active_subscription.monthly_equivalent)","sum","assinatura","BRL"),
("arr","Receita Recorrente Anualizada (ARR)","MRR anualizado para representar receita recorrente em base anual.","MRR * 12","custom","mês","BRL"),
("arpu","Receita Média por Usuário (ARPU)","Receita média atribuída por usuário ativo ou pagante no período.","net_revenue / NULLIF(distinct_paying_users,0)","ratio","mês","BRL"),
("arpa","Receita Média por Conta (ARPA)","Receita média por conta pagante no período.","net_revenue / NULLIF(distinct_paying_accounts,0)","ratio","mês","BRL"),
("gross_margin","Margem Bruta","Percentual da receita líquida que permanece após custos diretamente atribuíveis.","(net_revenue - cogs) / NULLIF(net_revenue,0)","ratio","mês","%"),
("gross_profit","Lucro Bruto","Receita líquida menos custo dos produtos ou serviços vendidos.","net_revenue - cogs","custom","mês","BRL"),
("ebitda","EBITDA","Resultado operacional antes de juros, impostos, depreciação e amortização.","operating_profit + depreciation + amortization","custom","mês","BRL"),
("ebitda_margin","Margem EBITDA","EBITDA dividido pela receita líquida.","EBITDA / NULLIF(net_revenue,0)","ratio","mês","%"),
("operating_margin","Margem Operacional","Resultado operacional dividido pela receita líquida.","operating_profit / NULLIF(net_revenue,0)","ratio","mês","%"),
("net_margin","Margem Líquida","Lucro líquido dividido pela receita líquida.","net_income / NULLIF(net_revenue,0)","ratio","mês","%"),
("cash_balance","Saldo de Caixa","Saldo de caixa e equivalentes na data de referência.","SUM(cash_accounts.balance)","sum","conta","BRL"),
("operating_cash_flow","Fluxo de Caixa Operacional","Caixa gerado ou consumido pelas operações.","SUM(operating_cash_movements)","sum","movimento","BRL"),
("free_cash_flow","Fluxo de Caixa Livre","Caixa operacional após investimentos de capital.","operating_cash_flow - capex","custom","mês","BRL"),
("burn_rate","Burn Rate","Velocidade média de consumo líquido de caixa.","net_cash_outflow / months","ratio","mês","BRL/mês"),
("runway","Runway","Meses estimados até o caixa se esgotar sob o burn atual.","cash_balance / NULLIF(monthly_net_burn,0)","ratio","mês","meses"),
("cac","Custo de Aquisição de Cliente (CAC)","Investimento de marketing e vendas dividido pelos novos clientes pagantes adquiridos.","(marketing_spend + sales_spend) / NULLIF(new_customers,0)","ratio","mês","BRL"),
("ltv","Lifetime Value (LTV)","Valor econômico esperado de um cliente ao longo de sua vida, segundo a metodologia definida.","expected_gross_profit_per_period * expected_lifetime_periods","custom","cliente","BRL"),
("ltv_cac_ratio","Razão LTV:CAC","Valor de vida do cliente dividido pelo custo de aquisição.","LTV / NULLIF(CAC,0)","ratio","mês","x"),
("cac_payback_months","Payback do CAC","Número de meses necessários para recuperar o CAC via margem bruta.","CAC / NULLIF(monthly_gross_profit_per_customer,0)","ratio","cliente","meses"),
("revenue_growth_rate","Crescimento de Receita","Variação percentual da receita em relação ao período comparável anterior.","(revenue - prior_revenue) / NULLIF(prior_revenue,0)","ratio","período","%"),
("mrr_growth_rate","Crescimento de MRR","Variação percentual do MRR em relação ao período anterior.","(MRR - prior_MRR) / NULLIF(prior_MRR,0)","ratio","mês","%"),
("net_revenue_retention","Net Revenue Retention (NRR)","Receita recorrente inicial retida após expansão, contração e churn, sem incluir novos clientes.","(starting_MRR + expansion_MRR - contraction_MRR - churned_MRR) / NULLIF(starting_MRR,0)","ratio","cohort","%")
]
for x in finance_metrics: metric("finance",*x)

# SALES (20)
sales_metrics = [
("pipeline_value","Valor do Pipeline","Valor total das oportunidades abertas elegíveis.","SUM(open_opportunity.amount)","sum","oportunidade","BRL"),
("weighted_pipeline","Pipeline Ponderado","Pipeline ponderado pelas probabilidades de fechamento.","SUM(opportunity.amount * opportunity.probability)","sum","oportunidade","BRL"),
("pipeline_coverage","Cobertura de Pipeline","Pipeline necessário em relação ao restante da meta de vendas.","open_pipeline / NULLIF(target - closed_won,0)","ratio","período","x"),
("win_rate","Taxa de Conversão de Vendas (Win Rate)","Percentual de oportunidades encerradas que foram ganhas.","closed_won / NULLIF(closed_won + closed_lost,0)","ratio","oportunidade","%"),
("sales_cycle_length","Duração do Ciclo de Vendas","Tempo médio entre criação e fechamento de uma oportunidade.","AVG(days_between(created_at,closed_at))","avg","oportunidade","dias"),
("lead_to_opportunity_rate","Conversão Lead → Oportunidade","Percentual de leads que se tornam oportunidades qualificadas.","opportunities / NULLIF(leads,0)","ratio","lead","%"),
("opportunity_to_customer_rate","Conversão Oportunidade → Cliente","Percentual de oportunidades encerradas que viram clientes.","won_opportunities / NULLIF(closed_opportunities,0)","ratio","oportunidade","%"),
("average_deal_size","Ticket Médio de Venda","Valor médio das oportunidades ganhas.","AVG(closed_won.amount)","avg","oportunidade","BRL"),
("bookings","Bookings","Valor contratado em novos negócios no período.","SUM(closed_won.contract_value)","sum","contrato","BRL"),
("quota_attainment","Atingimento de Quota","Vendas realizadas como percentual da quota atribuída.","closed_won_amount / NULLIF(quota,0)","ratio","vendedor","%"),
("new_logo_revenue","Receita de Novos Clientes","Receita contratada proveniente de novos clientes.","SUM(new_customer.contract_value)","sum","contrato","BRL"),
("expansion_revenue","Receita de Expansão","Receita incremental de clientes existentes por upsell, cross-sell ou expansão.","SUM(expansion.contract_value)","sum","contrato","BRL"),
("sales_velocity","Velocidade de Vendas","Valor esperado gerado pelo funil por unidade de tempo.","qualified_opportunities * avg_deal_size * win_rate / sales_cycle_length","custom","período","BRL/dia"),
("forecast_accuracy","Acurácia de Forecast","Proximidade entre previsão comercial e resultado realizado.","1 - ABS(forecast - actual) / NULLIF(ABS(actual),0)","custom","período","%"),
("average_days_to_close_won","Dias Médios até Ganho","Tempo médio entre criação e fechamento ganho.","AVG(days_between(created_at,won_at))","avg","oportunidade","dias"),
("lost_rate","Taxa de Perda","Percentual de oportunidades encerradas que foram perdidas.","closed_lost / NULLIF(closed_won + closed_lost,0)","ratio","oportunidade","%"),
("discount_rate","Taxa Média de Desconto","Desconto médio aplicado sobre o preço de tabela.","SUM(list_price - sold_price) / NULLIF(SUM(list_price),0)","ratio","linha de venda","%"),
("average_revenue_per_sales_rep","Receita Média por Vendedor","Receita ganha média atribuída por vendedor.","closed_won_amount / NULLIF(active_sales_reps,0)","ratio","período","BRL"),
("opportunities_created","Oportunidades Criadas","Número de novas oportunidades criadas no período.","COUNT(opportunity.id)","count","oportunidade","unidades"),
("sales_target","Meta de Vendas","Valor-alvo de vendas para o período e escopo definidos.","SUM(target.amount)","sum","meta","BRL")
]
for x in sales_metrics: metric("sales",*x)

# MARKETING (18)
marketing_metrics = [
("leads_generated","Leads Gerados","Número de leads capturados no período.","COUNT(lead.id)","count","lead","unidades"),
("mql_volume","Volume de MQLs","Número de leads qualificados por marketing.","COUNT(lead.id WHERE status='mql')","count","lead","unidades"),
("sql_volume","Volume de SQLs","Número de leads aceitos e qualificados por vendas.","COUNT(lead.id WHERE status='sql')","count","lead","unidades"),
("mql_to_sql_rate","Taxa MQL → SQL","Percentual de MQLs que se tornam SQLs.","sql / NULLIF(mql,0)","ratio","lead","%"),
("cost_per_lead","Custo por Lead","Investimento de marketing dividido pelo volume de leads.","marketing_spend / NULLIF(leads,0)","ratio","lead","BRL"),
("cost_per_mql","Custo por MQL","Investimento de marketing dividido pelo número de MQLs.","marketing_spend / NULLIF(mql,0)","ratio","MQL","BRL"),
("cost_per_sql","Custo por SQL","Investimento de marketing dividido pelo número de SQLs.","marketing_spend / NULLIF(sql,0)","ratio","SQL","BRL"),
("campaign_roi","ROI de Campanha","Retorno incremental atribuído à campanha em relação ao custo.","(attributed_margin - campaign_cost) / NULLIF(campaign_cost,0)","ratio","campanha","%"),
("marketing_sourced_revenue","Receita Sourced por Marketing","Receita de negócios atribuídos à origem de marketing.","SUM(attributed_closed_won.amount)","sum","oportunidade","BRL"),
("marketing_influenced_revenue","Receita Influenciada por Marketing","Receita de negócios que tiveram interação atribuível com marketing.","SUM(influenced_closed_won.amount)","sum","oportunidade","BRL"),
("website_sessions","Sessões do Website","Número de sessões do website no período.","COUNT(session.id)","count","sessão","unidades"),
("website_conversion_rate","Conversão do Website","Percentual de sessões que atingem o evento de conversão definido.","conversions / NULLIF(sessions,0)","ratio","sessão","%"),
("email_open_rate","Taxa de Abertura de Email","Percentual de emails entregues que foram abertos.","opened / NULLIF(delivered,0)","ratio","email","%"),
("email_click_rate","Taxa de Clique de Email","Percentual de emails entregues que geraram clique.","clicked / NULLIF(delivered,0)","ratio","email","%"),
("unsubscribe_rate","Taxa de Descadastro","Percentual de emails entregues que resultaram em descadastro.","unsubscribed / NULLIF(delivered,0)","ratio","email","%"),
("paid_media_roas","ROAS de Mídia Paga","Receita atribuída dividida pelo gasto em mídia.","attributed_revenue / NULLIF(media_spend,0)","ratio","campanha","x"),
("brand_search_share","Participação de Busca de Marca","Participação estimada da demanda de busca associada à marca.","brand_search_volume / NULLIF(total_category_search_volume,0)","ratio","período","%"),
("customer_acquisition_growth","Crescimento de Aquisição","Variação do número de novos clientes adquiridos.","(new_customers - prior_new_customers) / NULLIF(prior_new_customers,0)","ratio","período","%")
]
for x in marketing_metrics: metric("marketing",*x)

# PRODUCT (20)
product_metrics = [
("dau","Usuários Ativos Diários (DAU)","Número de usuários únicos que executaram uma ação ativa no dia.","COUNT(DISTINCT active_user_id)","count_distinct","dia","usuários"),
("wau","Usuários Ativos Semanais (WAU)","Número de usuários únicos ativos na semana.","COUNT(DISTINCT active_user_id)","count_distinct","semana","usuários"),
("mau","Usuários Ativos Mensais (MAU)","Número de usuários únicos ativos no mês.","COUNT(DISTINCT active_user_id)","count_distinct","mês","usuários"),
("dau_mau_ratio","Razão DAU/MAU (Stickiness)","Razão entre usuários ativos diários e mensais.","AVG(DAU) / NULLIF(MAU,0)","ratio","mês","%"),
("signup_conversion_rate","Conversão de Cadastro","Percentual de visitantes elegíveis que concluem cadastro.","signups / NULLIF(eligible_visitors,0)","ratio","usuário","%"),
("activation_rate","Taxa de Ativação","Percentual de novos usuários que atingem o evento de valor definido.","activated_users / NULLIF(new_users,0)","ratio","cohort","%"),
("time_to_activation","Tempo até Ativação","Tempo mediano entre cadastro e primeiro evento de ativação.","MEDIAN(days_between(signup_at,activation_at))","custom","usuário","dias"),
("onboarding_completion_rate","Conclusão de Onboarding","Percentual de novos usuários que concluem o onboarding.","completed_onboarding / NULLIF(started_onboarding,0)","ratio","usuário","%"),
("feature_adoption_rate","Adoção de Feature","Percentual de usuários ativos que utilizaram uma feature.","feature_users / NULLIF(active_users,0)","ratio","feature","%"),
("feature_usage_frequency","Frequência de Uso de Feature","Número médio de usos da feature por usuário adotante.","feature_events / NULLIF(feature_users,0)","ratio","feature","eventos/usuário"),
("session_frequency","Frequência de Sessões","Número médio de sessões por usuário ativo.","sessions / NULLIF(active_users,0)","ratio","usuário","sessões/usuário"),
("avg_session_duration","Duração Média da Sessão","Duração média das sessões de usuários.","AVG(session.duration_seconds)","avg","sessão","segundos"),
("retention_d1","Retenção D1","Percentual da cohort que retorna no dia seguinte ao evento de aquisição.","retained_d1 / NULLIF(cohort_users,0)","ratio","cohort","%"),
("retention_d7","Retenção D7","Percentual da cohort que retorna sete dias após aquisição.","retained_d7 / NULLIF(cohort_users,0)","ratio","cohort","%"),
("retention_d30","Retenção D30","Percentual da cohort que retorna trinta dias após aquisição.","retained_d30 / NULLIF(cohort_users,0)","ratio","cohort","%"),
("trial_to_paid_rate","Conversão Trial → Pago","Percentual de trials que se tornam clientes pagantes.","paid_trials / NULLIF(completed_trials,0)","ratio","trial","%"),
("free_to_paid_rate","Conversão Free → Pago","Percentual de usuários gratuitos que se tornam pagantes.","new_payers / NULLIF(eligible_free_users,0)","ratio","usuário","%"),
("product_revenue_per_user","Receita por Usuário","Receita atribuída por usuário ativo ou pagante.","product_revenue / NULLIF(paying_users,0)","ratio","usuário","BRL"),
("north_star_value","Valor da North Star Metric","Valor agregado do evento ou resultado escolhido como principal indicador de valor entregue.","SUM(north_star_event.value)","sum","período","unidades"),
("user_growth_rate","Crescimento de Usuários","Variação percentual da base ativa em relação ao período comparável.","(active_users - prior_active_users) / NULLIF(prior_active_users,0)","ratio","período","%")
]
for x in product_metrics: metric("product",*x)

# CUSTOMER SUCCESS (16)
customer_metrics = [
("churn_rate","Taxa de Churn de Clientes","Percentual de clientes elegíveis no início do período que cancelam.","churned_customers / NULLIF(starting_eligible_customers,0)","ratio","cliente","%"),
("logo_retention_rate","Retenção de Clientes","Percentual de clientes iniciais que permanecem ativos ao fim do período.","ending_starting_cohort_customers / NULLIF(starting_customers,0)","ratio","cohort","%"),
("revenue_churn_rate","Taxa de Churn de Receita","Percentual da receita recorrente inicial perdida por churn e contração.","(churned_MRR + contraction_MRR) / NULLIF(starting_MRR,0)","ratio","cohort","%"),
("gross_revenue_retention","Gross Revenue Retention (GRR)","Receita inicial retida após churn e contração, excluindo expansão.","(starting_MRR - churned_MRR - contraction_MRR) / NULLIF(starting_MRR,0)","ratio","cohort","%"),
("expansion_mrr","MRR de Expansão","MRR incremental obtido de clientes existentes por expansão.","SUM(expansion.mrr_delta)","sum","cliente","BRL"),
("contraction_mrr","MRR de Contração","MRR perdido por redução de contrato de clientes existentes.","SUM(contraction.mrr_delta)","sum","cliente","BRL"),
("churned_mrr","MRR Perdido por Churn","MRR perdido por cancelamento de clientes.","SUM(churned_subscription.mrr)","sum","cliente","BRL"),
("reactivation_mrr","MRR de Reativação","MRR recuperado de clientes anteriormente cancelados.","SUM(reactivation.mrr)","sum","cliente","BRL"),
("customer_health_score","Customer Health Score","Pontuação composta de saúde do cliente baseada nos sinais definidos pela organização.","WEIGHTED_SUM(health_signals)","custom","cliente","score"),
("nps","Net Promoter Score (NPS)","Percentual de promotores menos percentual de detratores.","pct_promoters - pct_detractors","custom","resposta","score"),
("csat","Customer Satisfaction Score (CSAT)","Percentual ou média de respostas positivas à pesquisa de satisfação.","positive_responses / NULLIF(valid_responses,0)","ratio","resposta","%"),
("customer_effort_score","Customer Effort Score (CES)","Avaliação do esforço percebido pelo cliente para realizar uma tarefa.","AVG(response.score)","avg","resposta","score"),
("time_to_value","Tempo até Valor","Tempo entre início do relacionamento e primeira realização do valor definido.","MEDIAN(days_between(start_at,value_at))","custom","cliente","dias"),
("customer_expansion_rate","Taxa de Expansão de Clientes","Percentual de clientes existentes que expandiram seu relacionamento.","expanded_customers / NULLIF(starting_customers,0)","ratio","cliente","%"),
("customer_concentration","Concentração de Receita em Clientes","Participação da receita gerada pelos maiores clientes no total.","SUM(top_n_customer_revenue) / NULLIF(total_revenue,0)","ratio","período","%"),
("renewal_rate","Taxa de Renovação","Percentual de contratos elegíveis que são renovados.","renewed_contracts / NULLIF(eligible_contracts,0)","ratio","contrato","%")
]
for x in customer_metrics: metric("customer",*x)

# ECOMMERCE (15)
ecom_metrics = [
("orders","Pedidos","Número de pedidos realizados no período.","COUNT(order.id)","count","pedido","pedidos"),
("purchasers","Compradores","Número de clientes únicos que realizaram ao menos uma compra.","COUNT(DISTINCT order.customer_id)","count_distinct","cliente","clientes"),
("conversion_rate","Taxa de Conversão de Compra","Percentual de sessões ou visitantes que realizam compra.","purchasers / NULLIF(eligible_visitors,0)","ratio","sessão","%"),
("average_order_value","Ticket Médio (AOV)","Receita líquida média por pedido.","net_sales / NULLIF(orders,0)","ratio","pedido","BRL"),
("items_per_order","Itens por Pedido","Quantidade média de itens por pedido.","items_sold / NULLIF(orders,0)","ratio","pedido","itens"),
("cart_add_rate","Taxa de Adição ao Carrinho","Percentual de sessões elegíveis que adicionam itens ao carrinho.","sessions_with_add_to_cart / NULLIF(eligible_sessions,0)","ratio","sessão","%"),
("cart_abandonment_rate","Abandono de Carrinho","Percentual de carrinhos iniciados que não resultam em compra.","abandoned_carts / NULLIF(carts,0)","ratio","carrinho","%"),
("checkout_completion_rate","Conclusão de Checkout","Percentual de checkouts iniciados que resultam em compra.","orders_from_checkout / NULLIF(checkouts,0)","ratio","checkout","%"),
("refund_rate","Taxa de Reembolso","Percentual do valor ou pedidos vendidos que foram reembolsados.","refunded_orders / NULLIF(orders,0)","ratio","pedido","%"),
("return_rate","Taxa de Devolução","Percentual de pedidos ou itens devolvidos.","returned_items / NULLIF(sold_items,0)","ratio","item","%"),
("repeat_purchase_rate","Taxa de Recompra","Percentual de compradores que realizam uma compra adicional no período definido.","repeat_buyers / NULLIF(purchasers,0)","ratio","cliente","%"),
("customer_lifetime_orders","Pedidos por Cliente ao Longo da Vida","Número médio de pedidos por cliente desde a primeira compra.","orders / NULLIF(unique_customers,0)","ratio","cliente","pedidos/cliente"),
("discount_rate","Taxa de Desconto","Percentual do valor de tabela concedido em descontos.","discount_amount / NULLIF(gross_merchandise_value,0)","ratio","pedido","%"),
("gross_merchandise_value","GMV","Valor bruto de mercadorias transacionadas antes de ajustes definidos.","SUM(order.gross_value)","sum","pedido","BRL"),
("net_merchandise_value","NMV","Valor líquido de mercadorias após descontos, devoluções e ajustes aplicáveis.","SUM(order.net_value)","sum","pedido","BRL")
]
for x in ecom_metrics: metric("ecommerce",*x)

# SUPPORT (12)
support_metrics = [
("ticket_volume","Volume de Tickets","Número de tickets criados no período.","COUNT(ticket.id)","count","ticket","tickets"),
("first_response_time","Tempo até Primeira Resposta","Tempo médio ou mediano entre criação e primeira resposta humana.","MEDIAN(minutes_between(created_at,first_response_at))","custom","ticket","minutos"),
("resolution_time","Tempo de Resolução","Tempo entre criação e resolução do ticket.","MEDIAN(hours_between(created_at,resolved_at))","custom","ticket","horas"),
("first_contact_resolution_rate","Resolução no Primeiro Contato","Percentual de tickets resolvidos sem interação adicional significativa.","fcr_tickets / NULLIF(resolved_tickets,0)","ratio","ticket","%"),
("ticket_reopen_rate","Taxa de Reabertura","Percentual de tickets resolvidos que são reabertos.","reopened_tickets / NULLIF(resolved_tickets,0)","ratio","ticket","%"),
("backlog","Backlog de Suporte","Número de tickets abertos e não resolvidos na referência.","COUNT(ticket.id WHERE status NOT IN ('resolved','closed'))","count","ticket","tickets"),
("sla_breach_rate","Taxa de Violação de SLA","Percentual de tickets que ultrapassam o SLA aplicável.","sla_breached / NULLIF(sla_eligible,0)","ratio","ticket","%"),
("csat","CSAT de Suporte","Satisfação média ou percentual positivo após atendimento.","AVG(survey.score)","avg","resposta","score"),
("tickets_per_customer","Tickets por Cliente","Número médio de tickets por cliente atendido.","tickets / NULLIF(customers,0)","ratio","cliente","tickets/cliente"),
("agent_utilization","Utilização de Agente","Proporção do tempo disponível do agente dedicado a trabalho produtivo.","productive_minutes / NULLIF(available_minutes,0)","ratio","agente","%"),
("cost_per_ticket","Custo por Ticket","Custo operacional do suporte dividido pelo volume de tickets.","support_cost / NULLIF(tickets,0)","ratio","ticket","BRL"),
("escalation_rate","Taxa de Escalação","Percentual de tickets que exigem transferência para outro nível.","escalated_tickets / NULLIF(tickets,0)","ratio","ticket","%")
]
for x in support_metrics: metric("support",*x)

# ENGINEERING (12)
engineering_metrics = [
("deployment_frequency","Frequência de Deploy","Número de deployments bem-sucedidos por unidade de tempo.","successful_deployments / period_duration","ratio","período","deploys/período"),
("change_lead_time","Lead Time de Mudança","Tempo entre commit e implantação bem-sucedida em produção.","MEDIAN(hours_between(commit_at,production_at))","custom","deploy","horas"),
("failed_deployment_recovery_time","Tempo de Recuperação de Deploy Falho","Tempo para recuperar de uma implantação que falha e exige intervenção.","MEDIAN(minutes_between(failure_at,recovery_at))","custom","deploy","minutos"),
("change_fail_rate","Taxa de Falha de Mudança","Percentual de deployments que causam falha em produção e exigem intervenção.","failed_deployments / NULLIF(deployments,0)","ratio","deploy","%"),
("deployment_rework_rate","Taxa de Retrabalho de Deploy","Percentual de deployments que são trabalho não planejado para corrigir incidentes.","rework_deployments / NULLIF(deployments,0)","ratio","deploy","%"),
("release_success_rate","Taxa de Sucesso de Release","Percentual de releases sem rollback, hotfix crítico ou falha definida.","successful_releases / NULLIF(releases,0)","ratio","release","%"),
("incident_count","Número de Incidentes","Quantidade de incidentes de produção no período.","COUNT(incident.id)","count","incidente","incidentes"),
("incident_rate","Taxa de Incidentes","Incidentes normalizados por volume de releases, tráfego ou período definido.","incidents / normalization_base","ratio","período","incidentes/unidade"),
("service_availability","Disponibilidade do Serviço","Percentual de tempo em que o serviço atende o critério de disponibilidade.","available_time / NULLIF(total_time,0)","ratio","serviço","%"),
("error_rate","Taxa de Erro","Proporção de requisições que resultam em erro segundo a definição operacional.","error_requests / NULLIF(total_requests,0)","ratio","requisição","%"),
("change_volume","Volume de Mudanças","Número de mudanças de software implantadas no período.","COUNT(deployment.id)","count","deploy","deployments"),
("mean_time_to_recover","MTTR Operacional","Tempo médio para restaurar o serviço após um incidente operacional.","AVG(hours_between(incident_start,recovery_at))","avg","incidente","horas")
]
for x in engineering_metrics: metric("engineering",*x)

# HR (12)
hr_metrics = [
("headcount","Headcount","Número de colaboradores ativos na data de referência.","COUNT(employee.id WHERE active=true)","count","colaborador","pessoas"),
("headcount_growth_rate","Crescimento de Headcount","Variação percentual do headcount em relação ao período comparável.","(headcount - prior_headcount) / NULLIF(prior_headcount,0)","ratio","período","%"),
("voluntary_turnover_rate","Turnover Voluntário","Percentual médio de colaboradores que saem voluntariamente no período.","voluntary_exits / AVG(headcount)","ratio","período","%"),
("involuntary_turnover_rate","Turnover Involuntário","Percentual médio de colaboradores desligados involuntariamente.","involuntary_exits / AVG(headcount)","ratio","período","%"),
("retention_rate","Taxa de Retenção de Pessoas","Percentual de colaboradores iniciais que permanecem ao fim do período.","retained_employees / NULLIF(starting_headcount,0)","ratio","cohort","%"),
("time_to_hire","Tempo para Contratação","Tempo entre abertura da vaga e aceite da proposta.","MEDIAN(days_between(job_opened_at,offer_accepted_at))","custom","contratação","dias"),
("time_to_fill","Tempo para Preenchimento","Tempo entre abertura da vaga e ocupação efetiva.","MEDIAN(days_between(job_opened_at,start_at))","custom","vaga","dias"),
("offer_acceptance_rate","Taxa de Aceite de Oferta","Percentual de ofertas de emprego aceitas.","accepted_offers / NULLIF(offers,0)","ratio","oferta","%"),
("cost_per_hire","Custo por Contratação","Custo total de recrutamento dividido pelas contratações realizadas.","recruiting_cost / NULLIF(new_hires,0)","ratio","contratação","BRL"),
("absenteeism_rate","Taxa de Absenteísmo","Percentual de horas planejadas perdidas por ausência.","absence_hours / NULLIF(scheduled_hours,0)","ratio","colaborador","%"),
("revenue_per_employee","Receita por Colaborador","Receita atribuída dividida pelo headcount médio.","revenue / NULLIF(average_headcount,0)","ratio","período","BRL/pessoa"),
("employee_engagement_score","Índice de Engajamento","Pontuação agregada de engajamento obtida por pesquisa definida pela organização.","AVG(engagement_survey.score)","avg","resposta","score")
]
for x in hr_metrics: metric("hr",*x)

# OPERATIONS (12)
ops_metrics = [
("throughput","Throughput Operacional","Volume de unidades processadas por unidade de tempo.","processed_units / period_duration","ratio","período","unidades/tempo"),
("cycle_time","Tempo de Ciclo Operacional","Tempo entre início e conclusão de uma operação.","MEDIAN(hours_between(start_at,completed_at))","custom","operação","horas"),
("on_time_rate","Taxa de Entrega no Prazo","Percentual de operações concluídas dentro do prazo acordado.","on_time / NULLIF(completed,0)","ratio","operação","%"),
("capacity_utilization","Utilização de Capacidade","Uso efetivo da capacidade disponível.","used_capacity / NULLIF(available_capacity,0)","ratio","período","%"),
("productivity","Produtividade","Output operacional por unidade de recurso consumido.","output / NULLIF(resource_input,0)","ratio","período","unidades/recurso"),
("cost_per_unit","Custo por Unidade","Custo operacional total dividido pelo output produzido.","operating_cost / NULLIF(output_units,0)","ratio","unidade","BRL/unidade"),
("defect_rate","Taxa de Defeito","Proporção de unidades que apresentam defeito segundo o critério definido.","defective_units / NULLIF(produced_units,0)","ratio","unidade","%"),
("first_pass_yield","First Pass Yield","Percentual de unidades aprovadas sem retrabalho na primeira passagem.","first_pass_good / NULLIF(total_units,0)","ratio","unidade","%"),
("rework_rate","Taxa de Retrabalho","Percentual de unidades que exigem processamento adicional.","reworked_units / NULLIF(processed_units,0)","ratio","unidade","%"),
("downtime_rate","Taxa de Indisponibilidade","Percentual de tempo em que o recurso operacional está indisponível.","downtime / NULLIF(scheduled_time,0)","ratio","recurso","%"),
("sla_attainment","Atingimento de SLA Operacional","Percentual de operações que atendem ao SLA definido.","sla_met / NULLIF(sla_eligible,0)","ratio","operação","%"),
("forecast_accuracy","Acurácia de Previsão Operacional","Acurácia entre previsão operacional e realização observada.","1 - ABS(forecast-actual)/NULLIF(ABS(actual),0)","custom","período","%")
]
for x in ops_metrics: metric("operations",*x)

# SUPPLY CHAIN (12)
sc_metrics = [
("inventory_value","Valor de Estoque","Valor contábil ou gerencial do estoque na referência.","SUM(inventory.stock_value)","sum","SKU","BRL"),
("inventory_units","Unidades em Estoque","Quantidade física disponível ou registrada em estoque.","SUM(inventory.quantity)","sum","SKU","unidades"),
("inventory_turnover","Giro de Estoque","Custo das mercadorias vendidas dividido pelo estoque médio.","cogs / NULLIF(average_inventory_value,0)","ratio","período","x"),
("days_inventory_outstanding","Dias de Estoque (DIO)","Número estimado de dias de estoque com base no consumo.","average_inventory / NULLIF(cogs,0) * days","custom","período","dias"),
("stockout_rate","Taxa de Ruptura","Proporção de demanda ou SKUs indisponíveis quando solicitados.","stockout_events / NULLIF(demand_events,0)","ratio","SKU","%"),
("fill_rate","Taxa de Atendimento de Pedido","Percentual da demanda atendida imediatamente pelo estoque disponível.","fulfilled_demand / NULLIF(total_demand,0)","ratio","pedido","%"),
("supplier_on_time_rate","Entrega no Prazo do Fornecedor","Percentual de entregas de fornecedores recebidas no prazo.","on_time_deliveries / NULLIF(deliveries,0)","ratio","entrega","%"),
("purchase_order_cycle_time","Tempo de Ciclo de Compra","Tempo entre criação do pedido de compra e recebimento.","MEDIAN(days_between(po_created_at,received_at))","custom","pedido de compra","dias"),
("procurement_savings","Economia de Compras","Economia obtida em relação ao baseline de preço definido.","baseline_cost - actual_cost","custom","pedido de compra","BRL"),
("forecast_bias","Viés de Forecast","Desvio sistemático entre demanda prevista e observada.","(forecast - actual) / NULLIF(actual,0)","ratio","período","%"),
("forecast_accuracy","Acurácia de Forecast de Demanda","Proximidade entre demanda prevista e realizada.","1 - ABS(forecast-actual)/NULLIF(ABS(actual),0)","custom","SKU/período","%"),
("order_fulfillment_cycle_time","Tempo de Fulfillment","Tempo entre pedido confirmado e pedido entregue.","MEDIAN(days_between(order_confirmed_at,delivered_at))","custom","pedido","dias")
]
for x in sc_metrics: metric("supply_chain",*x)

# DATA (10)
data_metrics = [
("freshness","Freshness de Dados","Atraso entre a última atualização observada e o momento esperado de disponibilidade.","NOW() - MAX(dataset.updated_at)","custom","dataset","tempo"),
("completeness","Completude","Percentual de registros ou campos esperados que estão presentes.","present_values / NULLIF(expected_values,0)","ratio","dataset","%"),
("null_rate","Taxa de Nulos","Percentual de valores nulos em um atributo monitorado.","null_values / NULLIF(total_values,0)","ratio","coluna","%"),
("duplicate_rate","Taxa de Duplicidade","Percentual de registros que violam a chave ou unicidade esperada.","duplicate_rows / NULLIF(total_rows,0)","ratio","registro","%"),
("validity_rate","Taxa de Validade","Percentual de valores que satisfazem regras de domínio e formato.","valid_values / NULLIF(total_values,0)","ratio","coluna","%"),
("pipeline_success_rate","Taxa de Sucesso de Pipeline","Percentual de execuções de pipeline concluídas com sucesso.","successful_runs / NULLIF(total_runs,0)","ratio","execução","%"),
("pipeline_failure_rate","Taxa de Falha de Pipeline","Percentual de execuções de pipeline que falham.","failed_runs / NULLIF(total_runs,0)","ratio","execução","%"),
("pipeline_duration","Duração de Pipeline","Tempo de execução de um pipeline entre início e conclusão.","MEDIAN(minutes_between(start_at,end_at))","custom","execução","minutos"),
("data_incident_rate","Taxa de Incidentes de Dados","Incidentes de qualidade ou disponibilidade por período ou dataset.","data_incidents / normalization_base","ratio","período","incidentes/unidade"),
("data_contract_compliance","Conformidade de Data Contract","Percentual de datasets que atendem aos contratos de esquema, qualidade e SLA.","compliant_checks / NULLIF(total_checks,0)","ratio","dataset","%")
]
for x in data_metrics: metric("data",*x)

# SECURITY (8)
security_metrics = [
("security_incident_count","Incidentes de Segurança","Número de incidentes de segurança registrados.","COUNT(security_incident.id)","count","incidente","incidentes"),
("mean_time_to_detect","Tempo Médio de Detecção (MTTD)","Tempo entre ocorrência de um incidente e sua detecção.","AVG(minutes_between(occurred_at,detected_at))","avg","incidente","minutos"),
("mean_time_to_contain","Tempo Médio de Contenção (MTTC)","Tempo entre detecção e contenção de um incidente.","AVG(minutes_between(detected_at,contained_at))","avg","incidente","minutos"),
("mean_time_to_remediate","Tempo Médio de Remediação (MTTR)","Tempo entre detecção e remediação completa.","AVG(hours_between(detected_at,remediated_at))","avg","incidente","horas"),
("critical_vulnerability_count","Vulnerabilidades Críticas","Número de vulnerabilidades críticas abertas na referência.","COUNT(vulnerability.id WHERE severity='critical' AND status='open')","count","vulnerabilidade","vulnerabilidades"),
("vulnerability_remediation_sla_rate","Atendimento de SLA de Vulnerabilidade","Percentual de vulnerabilidades remediadas dentro do prazo.","remediated_within_sla / NULLIF(sla_eligible,0)","ratio","vulnerabilidade","%"),
("security_control_compliance","Conformidade de Controles","Percentual de controles de segurança avaliados como conformes.","compliant_controls / NULLIF(assessed_controls,0)","ratio","controle","%"),
("phishing_failure_rate","Taxa de Falha em Simulação de Phishing","Percentual de participantes que executam uma ação considerada falha na simulação.","failed_users / NULLIF(simulated_users,0)","ratio","usuário","%")
]
for x in security_metrics: metric("security",*x)

# STRATEGY / EXECUTIVE (8)
strategy_metrics = [
("customer_count","Clientes Ativos","Número de clientes ativos na referência.","COUNT(DISTINCT customer.id WHERE status='active')","count_distinct","cliente","clientes"),
("new_customer_count","Novos Clientes","Número de clientes adquiridos no período.","COUNT(DISTINCT customer.id WHERE first_purchase BETWEEN period)","count_distinct","cliente","clientes"),
("customer_growth_rate","Crescimento de Clientes","Variação percentual da base de clientes em relação ao período comparável.","(customers-prior_customers)/NULLIF(prior_customers,0)","ratio","período","%"),
("revenue_per_customer","Receita por Cliente","Receita média por cliente ativo ou pagante.","revenue / NULLIF(customers,0)","ratio","cliente","BRL"),
("profitability_per_customer","Lucro por Cliente","Lucro ou margem atribuível dividido pela base de clientes.","contribution_profit / NULLIF(customers,0)","ratio","cliente","BRL"),
("market_share","Participação de Mercado","Receita ou volume da empresa dividido pelo mercado relevante definido.","company_value / NULLIF(total_market_value,0)","ratio","período","%"),
("share_of_wallet","Share of Wallet","Participação estimada da empresa no gasto total do cliente na categoria relevante.","company_customer_spend / NULLIF(customer_category_spend,0)","ratio","cliente","%"),
("strategic_kpi_attainment","Atingimento de KPI Estratégico","Resultado realizado como percentual da meta estratégica.","actual / NULLIF(target,0)","ratio","KPI","%")
]
for x in strategy_metrics: metric("strategy",*x)

# QUALITY / MANUFACTURING (8)
quality_metrics = [
("defect_density","Densidade de Defeitos","Número de defeitos normalizado por unidade produzida, código ou volume definido.","defects / NULLIF(units,0)","ratio","unidade","defeitos/unidade"),
("customer_complaint_rate","Taxa de Reclamação de Cliente","Reclamações normalizadas por volume de clientes ou unidades vendidas.","complaints / NULLIF(units_or_customers,0)","ratio","período","reclamações/unidade"),
("scrap_rate","Taxa de Refugo","Percentual de produção descartada por não conformidade.","scrap_units / NULLIF(produced_units,0)","ratio","unidade","%"),
("cost_of_poor_quality","Custo da Má Qualidade","Custo associado a falhas internas, externas, retrabalho e refugo.","SUM(internal_failure_cost + external_failure_cost + rework_cost)","sum","período","BRL"),
("audit_nonconformity_rate","Taxa de Não Conformidade em Auditoria","Não conformidades encontradas por item ou requisito auditado.","nonconformities / NULLIF(audited_items,0)","ratio","auditoria","%"),
("corrective_action_on_time_rate","Ações Corretivas no Prazo","Percentual de ações corretivas concluídas dentro do prazo.","on_time_actions / NULLIF(due_actions,0)","ratio","ação","%"),
("process_capability_index","Índice de Capacidade de Processo","Índice estatístico de capacidade do processo em relação aos limites de especificação.","MIN((USL-mean)/(3*stddev),(mean-LSL)/(3*stddev))","custom","processo","índice"),
("first_time_quality_rate","Qualidade na Primeira Passagem","Percentual de unidades que atendem aos requisitos sem correção.","first_pass_good / NULLIF(total_units,0)","ratio","unidade","%")
]
for x in quality_metrics: metric("quality",*x)

# Normalize benchmark field and add a catalog-level summary.
catalog.sort(key=lambda x: x["id"])

# Build entity table from all unique entity_ids referenced by metrics
_entity_grains: dict = defaultdict(set)
for _m in catalog:
    _eid = _m.get("entity_id")
    if _eid:
        _entity_grains[_eid].add(_m["grain"].lower())

entities = sorted([
    {
        "entity_id":    eid,
        "name":         eid.replace("_", " ").title(),
        "pk_column":    ENTITY_PK.get(eid, eid + "_id"),
        "grain_aliases": sorted(_entity_grains[eid]),
    }
    for eid in _entity_grains
], key=lambda x: x["entity_id"])

output = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "catalog_version": "2.0",
    "title": "Comprehensive Business Metric Catalog",
    "description": "Research-informed canonical metric catalog spanning finance, sales, marketing, product, customer success, ecommerce, support, engineering, HR, operations, supply chain, data, security, strategy and quality.",
    "entity_count": len(entities),
    "metric_count": len(catalog),
    "entities": entities,
    "metrics": catalog
}

path = Path("data/metric_catalog_v1.json")
path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

# Also create a compact index for easy browsing.
index = []
for m in catalog:
    index.append({
        "id": m["id"],
        "name": m["name"],
        "department": m["department"],
        "aggregation": m["aggregation"],
        "grain": m["grain"],
        "unit": m["unit"],
        "status": m["status"]
    })
index_path = Path("data/metric_catalog_index_v1.json")
index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Created {path} with {len(catalog)} canonical metrics.")
print(f"Created {index_path} with compact index.")
print("Domains:", sorted({m["id"].split('.')[0] for m in catalog}))

