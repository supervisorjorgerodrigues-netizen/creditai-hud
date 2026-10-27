"""Contratos Pydantic v2 (entrada/saída) da API CreditAI."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TipoMoradia(str, Enum):
    PROPRIA = "PROPRIA"
    ALUGADA = "ALUGADA"
    FINANCIADA = "FINANCIADA"
    OUTROS = "OUTROS"


class Finalidade(str, Enum):
    EDUCACAO = "EDUCACAO"
    SAUDE = "SAUDE"
    NEGOCIOS = "NEGOCIOS"
    PESSOAL = "PESSOAL"
    QUITACAO_DIVIDAS = "QUITACAO_DIVIDAS"
    IMOVEL = "IMOVEL"


class DadosClienteInput(BaseModel):
    """Apenas `idade` e `pontuacao_credito` são obrigatórios (retrocompatível com o projeto base)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "nome_cliente": "Carlos Eduardo Santos",
                "idade": 35,
                "pontuacao_credito": 720,
                "renda_mensal": 7500.0,
                "valor_emprestimo": 18000.0,
                "prazo_meses": 24,
                "tempo_emprego_anos": 5.0,
                "inadimplencia_previa": 0,
                "consultas_6m": 1,
                "utilizacao_cartao": 0.35,
                "tipo_moradia": "PROPRIA",
                "finalidade_emprestimo": "NEGOCIOS",
            }
        },
    )

    nome_cliente: str = Field("Cliente Anônimo", min_length=2, max_length=120)
    idade: int = Field(..., ge=18, le=100)
    pontuacao_credito: int = Field(..., ge=300, le=850, description="Score de bureau (300–850)")
    renda_mensal: float = Field(5000.0, gt=0, le=1_000_000)
    valor_emprestimo: float = Field(15000.0, ge=500, le=500_000)
    prazo_meses: int = Field(24, ge=3, le=120)
    tempo_emprego_anos: float = Field(3.0, ge=0, le=60)
    inadimplencia_previa: int = Field(0, ge=0, le=1)
    consultas_6m: int = Field(1, ge=0, le=30, description="Consultas ao CPF nos últimos 6 meses")
    utilizacao_cartao: float = Field(0.35, ge=0, le=1, description="Uso do limite do cartão (0–1)")
    tipo_moradia: TipoMoradia = TipoMoradia.ALUGADA
    finalidade_emprestimo: Finalidade = Finalidade.PESSOAL


class FatorDecisao(BaseModel):
    atributo: str
    valor: Any
    impacto: float = Field(..., description="Δ probabilidade vs. perfil de referência")


class PredicaoCreditoOutput(BaseModel):
    nome_cliente: str
    decisao: str
    codigo_decisao: int
    score_creditai: int = Field(..., ge=0, le=1000)
    rating: str
    faixa_risco: str
    probabilidade_aprovacao: float
    probabilidade_risco_inadimplencia: float
    limiar_politica: float
    parcela_estimada: float
    comprometimento_renda: float
    fatores: list[FatorDecisao]
    latencia_ms: float
    versao_modelo: str


class HealthResponse(BaseModel):
    status_servico: str
    modelo_carregado_em_ram: bool
    modelo: Optional[str] = None
    algoritmo: Optional[str] = None
    versao_modelo: Optional[str] = None
    treinado_em: Optional[str] = None
    status_treino: str
    metricas_referencia_teste: dict[str, Any] = {}
    latencia_p50_ms: Optional[float] = None
    total_predicoes: int = 0


class TreinoRequest(BaseModel):
    n_amostras: Optional[int] = Field(None, ge=1000, le=200_000)
    max_iter: Optional[int] = Field(None, ge=20, le=3000)
    passo_iter: Optional[int] = Field(None, ge=5, le=200)
    learning_rate: Optional[float] = Field(None, gt=0, le=1)
    max_depth: Optional[int] = Field(None, ge=2, le=16)
    paciencia: Optional[int] = Field(None, ge=1, le=50)
    limiar_decisao: Optional[float] = Field(None, gt=0, lt=1)
    regerar_dataset: Optional[bool] = None
    pausa_visual_s: Optional[float] = Field(None, ge=0, le=2)
