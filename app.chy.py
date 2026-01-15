import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(page_title="Plateforme d'Analyse Financière", layout="wide", initial_sidebar_state="expanded")

st.title("📈 Plateforme d'Analyse Financière")
st.markdown("### Projet de Mathématiques Appliquées à la Finance")

# ============================================
# SECTION 1: ACQUISITION DES DONNÉES
# ============================================
st.sidebar.header("⚙️ Configuration")

# Sélection de l'actif
actif = st.sidebar.selectbox(
    "Actif financier",
    ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "GOOGL", "MSFT"]
)

# Sélection de la période
col1, col2 = st.sidebar.columns(2)
date_debut = col1.date_input("Date début", datetime(2023, 1, 1))
date_fin = col2.date_input("Date fin", datetime(2023, 12, 31))

# Fréquence
frequence = st.sidebar.selectbox("Fréquence", ["1d", "1h", "5m"])

# Bouton de chargement
charger = st.sidebar.button("🔄 Charger les données")

@st.cache_data
def charger_donnees(ticker, debut, fin, interval="1d"):
    """Charge les données financières via yfinance"""
    try:
        data = yf.download(ticker, start=debut, end=fin, interval=interval, progress=False)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None

# ============================================
# SECTION 2: TRAITEMENT MATHÉMATIQUE
# ============================================

def calculer_rendements(df):
    """
    Calcul des rendements arithmétiques et logarithmiques
    Rendement arithmétique: Rt = (Pt - Pt-1) / Pt-1
    Rendement logarithmique: rt = ln(Pt / Pt-1)
    """
    df['Rendement_Arith'] = df['Close'].pct_change()
    df['Rendement_Log'] = np.log(df['Close'] / df['Close'].shift(1))
    return df

def calculer_statistiques(rendements):
    """
    Calcul des statistiques descriptives
    Moyenne: μ = (1/n) * Σ(Ri)
    Variance: σ² = (1/(n-1)) * Σ(Ri - μ)²
    Volatilité annualisée: σ_annuel = σ_quotidien * √252
    Skewness: E[(R - μ)³] / σ³
    Kurtosis: E[(R - μ)⁴] / σ⁴
    """
    r = rendements.dropna()
    
    stats_dict = {
        'Moyenne (%)': r.mean() * 100,
        'Médiane (%)': r.median() * 100,
        'Écart-type (%)': r.std() * 100,
        'Volatilité annualisée (%)': r.std() * np.sqrt(252) * 100,
        'Skewness': stats.skew(r),
        'Kurtosis': stats.kurtosis(r),
        'Minimum (%)': r.min() * 100,
        'Maximum (%)': r.max() * 100,
        'Percentile 5% (%)': r.quantile(0.05) * 100,
        'Percentile 25% (%)': r.quantile(0.25) * 100,
        'Percentile 75% (%)': r.quantile(0.75) * 100,
        'Percentile 95% (%)': r.quantile(0.95) * 100,
    }
    
    return stats_dict

# ============================================
# SECTION 3: INDICATEURS TECHNIQUES
# ============================================

def calculer_sma(df, periode):
    """
    Moyenne Mobile Simple
    SMAn(t) = (1/n) * Σ(Pt-i) pour i=0 à n-1
    """
    return df['Close'].rolling(window=periode).mean()

def calculer_ema(df, periode):
    """
    Moyenne Mobile Exponentielle
    EMAn(t) = α * Pt + (1-α) * EMAn(t-1)
    où α = 2/(n+1)
    """
    return df['Close'].ewm(span=periode, adjust=False).mean()

def calculer_bollinger(df, periode=20, num_std=2):
    """
    Bandes de Bollinger
    Bande supérieure = SMAn(t) + k * σn(t)
    Bande inférieure = SMAn(t) - k * σn(t)
    """
    sma = df['Close'].rolling(window=periode).mean()
    std = df['Close'].rolling(window=periode).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    return sma, upper, lower

def calculer_rsi(df, periode=14):
    """
    Relative Strength Index
    RSI = 100 - (100 / (1 + RS))
    où RS = Moyenne des gains / Moyenne des pertes
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periode).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periode).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculer_macd(df, rapide=12, lent=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence)
    MACD = EMA12(t) - EMA26(t)
    Signal = EMA9(MACD)
    Histogramme = MACD - Signal
    """
    ema_rapide = df['Close'].ewm(span=rapide, adjust=False).mean()
    ema_lent = df['Close'].ewm(span=lent, adjust=False).mean()
    macd = ema_rapide - ema_lent
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# ============================================
# SECTION 4: BACKTESTING
# ============================================

def backtest_strategie_sma(df, periode_courte=20, periode_longue=50, capital_initial=1000, frais=0.001):
    """
    Backtesting d'une stratégie de croisement de moyennes mobiles
    Signal d'achat: SMA_courte > SMA_longue
    Signal de vente: SMA_courte < SMA_longue
    """
    df = df.copy()
    
    # Calcul des SMA
    df['SMA_court'] = calculer_sma(df, periode_courte)
    df['SMA_long'] = calculer_sma(df, periode_longue)
    
    # Génération des signaux
    df['Signal'] = 0
    df.loc[df['SMA_court'] > df['SMA_long'], 'Signal'] = 1
    
    # Positions (décalage d'un jour pour éviter le look-ahead bias)
    df['Position'] = df['Signal'].shift(1)
    
    # Calcul des rendements
    df['Rendement_Actif'] = df['Close'].pct_change()
    
    # Rendement de la stratégie: R_strat_t = Position_(t-1) * R_actif_t
    df['Rendement_Strat'] = df['Position'] * df['Rendement_Actif']
    
    # Identification des transactions (changement de position)
    df['Transaction'] = df['Position'].diff().abs()
    
    # Application des frais de transaction
    df['Frais'] = df['Transaction'] * frais
    df['Rendement_Net'] = df['Rendement_Strat'] - df['Frais']
    
    # Évolution du capital: Ct = Ct-1 * (1 + R_strat_t)
    df['Capital'] = capital_initial * (1 + df['Rendement_Net']).cumprod()
    
    # Capital Buy & Hold (référence)
    df['Capital_BH'] = capital_initial * (1 + df['Rendement_Actif']).cumprod()
    
    # Calcul des métriques
    rendements_strat = df['Rendement_Net'].dropna()
    
    # Rendement total
    rendement_total = (df['Capital'].iloc[-1] - capital_initial) / capital_initial * 100
    rendement_bh = (df['Capital_BH'].iloc[-1] - capital_initial) / capital_initial * 100
    
    # Volatilité annualisée
    volatilite = rendements_strat.std() * np.sqrt(252) * 100
    
    # Ratio de Sharpe: (moyenne(R_strat) / σ_strat) * √252
    if rendements_strat.std() > 0:
        sharpe = (rendements_strat.mean() / rendements_strat.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Maximum Drawdown: MDD = max((Ct - max(Cs)) / max(Cs))
    capital_cummax = df['Capital'].cummax()
    drawdown = (df['Capital'] - capital_cummax) / capital_cummax
    max_drawdown = drawdown.min() * 100
    
    # Nombre de trades
    nb_trades = int(df['Transaction'].sum())
    
    # Calcul du win rate et profit factor
    df['PnL'] = df['Capital'].diff()
    trades_winning = len(df[df['PnL'] > 0])
    trades_total = len(df[df['PnL'] != 0])
    win_rate = (trades_winning / trades_total * 100) if trades_total > 0 else 0
    
    total_gains = df[df['PnL'] > 0]['PnL'].sum()
    total_pertes = abs(df[df['PnL'] < 0]['PnL'].sum())
    profit_factor = (total_gains / total_pertes) if total_pertes > 0 else 0
    
    metriques = {
        'Rendement Total (%)': rendement_total,
        'Rendement Buy & Hold (%)': rendement_bh,
        'Volatilité Annualisée (%)': volatilite,
        'Ratio de Sharpe': sharpe,
        'Maximum Drawdown (%)': max_drawdown,
        'Nombre de Trades': nb_trades,
        'Win Rate (%)': win_rate,
        'Profit Factor': profit_factor
    }
    
    return df, metriques

# ============================================
# INTERFACE PRINCIPALE
# ============================================

if charger or 'df' not in st.session_state:
    with st.spinner("Chargement des données..."):
        df = charger_donnees(actif, date_debut, date_fin, frequence)
        if df is not None and len(df) > 0:
            st.session_state.df = df
            st.success(f"✅ {len(df)} lignes chargées pour {actif}")
        else:
            st.error("Impossible de charger les données")
            st.stop()

if 'df' in st.session_state:
    df = st.session_state.df.copy()
    
    # Calcul des rendements
    df = calculer_rendements(df)
    
    # ============================================
    # GRAPHIQUE PRINCIPAL
    # ============================================
    
    st.header("📊 Graphique Principal")
    
    # Sélection des indicateurs
    col1, col2, col3, col4 = st.columns(4)
    show_sma20 = col1.checkbox("SMA(20)", value=True)
    show_sma50 = col2.checkbox("SMA(50)", value=True)
    show_bollinger = col3.checkbox("Bandes de Bollinger", value=False)
    show_rsi = col4.checkbox("RSI", value=False)
    
    # Création du graphique
    fig = make_subplots(
        rows=2 if show_rsi else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3] if show_rsi else [1],
        subplot_titles=("Prix et Indicateurs", "RSI") if show_rsi else ("Prix et Indicateurs",)
    )
    
    # Prix
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        name='Prix',
        line=dict(color='#3B82F6', width=2)
    ), row=1, col=1)
    
    # Indicateurs
    if show_sma20:
        sma20 = calculer_sma(df, 20)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=sma20,
            name='SMA(20)',
            line=dict(color='#10B981', width=1.5)
        ), row=1, col=1)
    
    if show_sma50:
        sma50 = calculer_sma(df, 50)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=sma50,
            name='SMA(50)',
            line=dict(color='#F59E0B', width=1.5)
        ), row=1, col=1)
    
    if show_bollinger:
        sma, upper, lower = calculer_bollinger(df)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=upper,
            name='BB Supérieure',
            line=dict(color='#EF4444', width=1, dash='dash')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=lower,
            name='BB Inférieure',
            line=dict(color='#EF4444', width=1, dash='dash'),
            fill='tonexty'
        ), row=1, col=1)
    
    if show_rsi:
        rsi = calculer_rsi(df)
        fig.add_trace(go.Scatter(
            x=df['Date'], y=rsi,
            name='RSI',
            line=dict(color='#8B5CF6', width=2)
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(
        height=600 if show_rsi else 500,
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # STATISTIQUES
    # ============================================
    
    st.header("📈 Statistiques des Rendements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistiques Descriptives")
        stats_dict = calculer_statistiques(df['Rendement_Arith'])
        stats_df = pd.DataFrame.from_dict(stats_dict, orient='index', columns=['Valeur'])
        st.dataframe(stats_df, use_container_width=True)
        
        # Test de normalité
        rendements_clean = df['Rendement_Arith'].dropna()
        stat, p_value = stats.shapiro(rendements_clean[:5000])  # Limite pour Shapiro
        st.metric("Test de Normalité (Shapiro-Wilk)", 
                  f"p-value = {p_value:.4f}",
                  "Non normal" if p_value < 0.05 else "Normal")
    
    with col2:
        st.subheader("Distribution des Rendements")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df['Rendement_Arith'].dropna() * 100,
            nbinsx=50,
            name='Rendements',
            marker_color='#3B82F6'
        ))
        fig_hist.update_layout(
            template='plotly_dark',
            xaxis_title='Rendement (%)',
            yaxis_title='Fréquence',
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # ============================================
    # BACKTESTING
    # ============================================
    
    st.header("🎯 Backtesting de Stratégie")
    st.markdown("**Stratégie de croisement de moyennes mobiles (SMA)**")
    
    col1, col2, col3 = st.columns(3)
    sma_court = col1.number_input("SMA Court", value=20, min_value=5, max_value=100)
    sma_long = col2.number_input("SMA Long", value=50, min_value=10, max_value=200)
    capital = col3.number_input("Capital Initial (€)", value=1000, min_value=100)
    
    if st.button("🚀 Lancer le Backtest"):
        with st.spinner("Calcul en cours..."):
            df_backtest, metriques = backtest_strategie_sma(df, sma_court, sma_long, capital)
            
            # Affichage des métriques
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rendement Total", f"{metriques['Rendement Total (%)']:.2f}%")
            col2.metric("Ratio de Sharpe", f"{metriques['Ratio de Sharpe']:.2f}")
            col3.metric("Max Drawdown", f"{metriques['Maximum Drawdown (%)']:.2f}%")
            col4.metric("Win Rate", f"{metriques['Win Rate (%)']:.2f}%")
            
            # Tableau des métriques
            st.subheader("Métriques Détaillées")
            metriques_df = pd.DataFrame.from_dict(metriques, orient='index', columns=['Valeur'])
            st.dataframe(metriques_df, use_container_width=True)
            
            # Graphique de l'évolution du capital
            st.subheader("Évolution du Capital")
            fig_capital = go.Figure()
            fig_capital.add_trace(go.Scatter(
                x=df_backtest['Date'],
                y=df_backtest['Capital'],
                name='Stratégie',
                line=dict(color='#10B981', width=2)
            ))
            fig_capital.add_trace(go.Scatter(
                x=df_backtest['Date'],
                y=df_backtest['Capital_BH'],
                name='Buy & Hold',
                line=dict(color='#3B82F6', width=2, dash='dash')
            ))
            fig_capital.update_layout(
                template='plotly_dark',
                xaxis_title='Date',
                yaxis_title='Capital (€)',
                height=400
            )
            st.plotly_chart(fig_capital, use_container_width=True)
            
            # Analyse critique
            st.subheader("💡 Analyse Critique")
            if metriques['Rendement Total (%)'] > metriques['Rendement Buy & Hold (%)']:
                st.success(f"✅ La stratégie surperforme le Buy & Hold de {metriques['Rendement Total (%)'] - metriques['Rendement Buy & Hold (%)']:.2f}%")
            else:
                st.warning(f"⚠️ La stratégie sous-performe le Buy & Hold de {metriques['Rendement Buy & Hold (%)'] - metriques['Rendement Total (%)']:.2f}%")
            
            st.info(f"""
            **Interprétation:**
            - Le ratio de Sharpe de {metriques['Ratio de Sharpe']:.2f} indique {'un bon' if metriques['Ratio de Sharpe'] > 1 else 'un faible'} rendement ajusté au risque
            - Le Maximum Drawdown de {metriques['Maximum Drawdown (%)']:.2f}% représente la perte maximale depuis un pic
            - {metriques['Nombre de Trades']} trades ont été exécutés avec un win rate de {metriques['Win Rate (%)']:.2f}%
            """)

    # ============================================
    # DONNÉES BRUTES
    # ============================================
    
    with st.expander("📋 Voir les données brutes"):
        st.dataframe(df.tail(50), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Projet de Mathématiques Appliquées à la Finance** | Encadrant: M. Hamza Saber")