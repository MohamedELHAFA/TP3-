#!/usr/bin/env python3
"""
Smart City Waste Management Dashboard - Version Interactive
Dashboard Streamlit avancé avec fonctionnalités de prise de décision
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import boto3
import io
from datetime import datetime, timedelta
import time
from botocore.client import Config
import math
from geopy.distance import geodesic
import openrouteservice

# Configuration de la page
st.set_page_config(
    page_title="Smart Waste Management Paris",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjNkODlhODg2ZjI4MDQwNDZhMTg4NzM1MDcxNjQ4ZWQyIiwiaCI6Im11cm11cjY0In0="  # 🔒 Remplace par ta clé

ors_client = openrouteservice.Client(key=ORS_API_KEY)

# Configuration S3/MinIO
@st.cache_resource
def init_s3_client():
    return boto3.client(
        's3',
        endpoint_url='http://localhost:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123',
        config=Config(signature_version='s3v4')
    )

s3 = init_s3_client()

# Variables de session pour l'interactivité
if 'selected_sensors' not in st.session_state:
    st.session_state.selected_sensors = []
if 'planned_route' not in st.session_state:
    st.session_state.planned_route = []
if 'route_geometries' not in st.session_state:
    st.session_state.route_geometries = []
if 'alert_threshold' not in st.session_state:
    st.session_state.alert_threshold = 80
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'all'
if 'last_action' not in st.session_state:
    st.session_state.last_action = None
if 'awaiting_confirmation' not in st.session_state:
    st.session_state.awaiting_confirmation = False
if 'confirmation_details' not in st.session_state:
    st.session_state.confirmation_details = None
if 'route_calculated' not in st.session_state:
    st.session_state.route_calculated = False

# Fonctions de chargement des données
@st.cache_data(ttl=30)
def load_sensor_current_data():
    """Charge les niveaux actuels des capteurs depuis Gold"""
    try:
        obj = s3.get_object(Bucket='gold', Key='sensor/sensor_data.txt')
        fill_levels = [int(x.strip()) for x in obj['Body'].read().decode('utf-8').splitlines()]
        return fill_levels
    except Exception as e:
        st.error(f"Erreur chargement niveaux actuels: {e}")
        return []

@st.cache_data(ttl=60)
def load_sensor_positions():
    """Charge les positions des capteurs depuis Silver"""
    try:
        obj = s3.get_object(Bucket='silver', Key='sensors/positions.parquet')
        df_positions = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        return df_positions
    except Exception as e:
        st.error(f"Erreur chargement positions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_historic_data():
    """Charge l'historique des niveaux depuis Silver"""
    try:
        obj = s3.get_object(Bucket='silver', Key='sensors/historic_fill.parquet')
        df_historic = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        if 'timestamp' in df_historic.columns:
            df_historic['ts'] = pd.to_datetime(df_historic['timestamp'], unit='ms')
        return df_historic
    except Exception as e:
        st.error(f"Erreur chargement historique: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_tonnage_data():
    """Charge les données de tonnage depuis Silver"""
    try:
        obj = s3.get_object(Bucket='silver', Key='sensor_tonnage/proxy.parquet')
        df_tonnage = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        return df_tonnage
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_model_metrics():
    """Charge les métriques du modèle ML depuis Gold"""
    try:
        paginator = s3.get_paginator('list_objects_v2')
        objects = []
        for page in paginator.paginate(Bucket='gold', Prefix='metrics/'):
            objects.extend(page.get('Contents', []))
        
        if not objects:
            return None
            
        latest = max(objects, key=lambda x: x['LastModified'])
        obj = s3.get_object(Bucket='gold', Key=latest['Key'])
        metrics = json.loads(obj['Body'].read().decode('utf-8'))
        return metrics
    except Exception as e:
        return None

# Fonctions d'optimisation et de calcul
import requests

def get_real_route_distance(start_coords, end_coords):
    """Calcule distance et durée de conduite réelle entre deux points via OpenRouteService"""
    try:
        route = ors_client.directions(
            coordinates=[(start_coords[1], start_coords[0]), (end_coords[1], end_coords[0])],
            profile='driving-car',
            format='geojson'
        )
        
        geometry = route['features'][0]['geometry']['coordinates']
        distance_km = route['features'][0]['properties']['segments'][0]['distance'] / 1000  # m → km
        duration_min = route['features'][0]['properties']['segments'][0]['duration'] / 60  # sec → min
        
        return {
            'distance_km': distance_km,
            'duration_min': duration_min,
            'geometry': geometry,
            'success': True
        }
    except Exception as e:
        print(f"Erreur ORS: {e}")
        # Fallback géodésique
        distance_km = geodesic(start_coords, end_coords).kilometers * 1.3
        duration_min = (distance_km / 20) * 60
        return {
            'distance_km': distance_km,
            'duration_min': duration_min,
            'geometry': [],
            'success': False
        }

def calculate_route_optimization(priority_sensors, df_positions, depot_lat=48.8566, depot_lon=2.3522):
    """Calcule un parcours optimisé avec distances routières améliorées"""
    if not priority_sensors:
        return [], 0, 0, []
    
    # Récupérer les positions des capteurs prioritaires
    route_points = []
    for sensor_id in priority_sensors:
        sensor_data = df_positions[df_positions['sensor_id'] == sensor_id]
        if not sensor_data.empty:
            route_points.append({
                'sensor_id': sensor_id,
                'lat': sensor_data.iloc[0]['lat'],
                'lon': sensor_data.iloc[0]['lon']
            })
    
    if not route_points:
        return [], 0, 0, []
    
    # Algorithme glouton avec vraies distances routières
    unvisited = route_points.copy()
    route = []
    current_pos = (depot_lat, depot_lon)
    total_distance = 0
    total_time = 0
    route_geometries = []
    
    # Progress bar dans un container pour éviter qu'elle disparaisse
    progress_container = st.empty()
    progress_bar = progress_container.progress(0)
    
    step = 0
    total_steps = len(route_points) + 1  # +1 pour le retour au dépôt
    
    while unvisited:
        # Calculer les vraies distances routières vers tous les points non visités
        real_distances = []
        for point in unvisited:
            route_info = get_real_route_distance(current_pos, (point['lat'], point['lon']))
            real_distances.append({
                'distance': route_info['distance_km'],
                'duration': route_info['duration_min'],
                'geometry': route_info['geometry'],
                'api_success': route_info['success'],
                'point': point
            })
        
        # Trouver le point le plus proche selon la vraie distance
        closest = min(real_distances, key=lambda x: x['distance'])
        closest_point = closest['point']
        
        # Ajouter à la route
        route.append({
            **closest_point,
            'real_distance': closest['distance'],
        'real_duration': closest['duration'],
        'api_used': closest['api_success'],
        'geometry': closest['geometry']
        })
        
        # Mettre à jour les totaux
        total_distance += closest['distance']
        total_time += closest['duration']
        
        # Retirer le point visité et mettre à jour la position
        unvisited.remove(closest_point)
        current_pos = (closest_point['lat'], closest_point['lon'])
        
        # Mettre à jour la barre de progression
        step += 1
        progress_bar.progress(step / total_steps)
    
    # Calculer le retour au dépôt
    return_info = get_real_route_distance(current_pos, (depot_lat, depot_lon))
    total_distance += return_info['distance_km']
    total_time += return_info['duration_min']
    
    # Finaliser la barre de progression
    progress_bar.progress(1.0)
    
    # Nettoyer la barre de progression après un délai
    import time
    time.sleep(1)
    progress_container.empty()
    
    # Ajouter le temps de collecte (5 min par capteur)
    collection_time = len(route) * 5
    total_time += collection_time
    
    return route, total_distance, total_time, route_geometries

def predict_next_collection_time(sensor_id, df_historic, current_level):
    """Prédit quand un capteur sera plein"""
    if df_historic.empty:
        return "Données insuffisantes"
    
    # Filtrer les données du capteur
    sensor_data = df_historic[df_historic['sensor_id'] == sensor_id].sort_values('ts')
    
    if len(sensor_data) < 5:
        return "Données insuffisantes"
    
    # Calculer le taux de remplissage moyen (sur les 24 dernières heures)
    recent_data = sensor_data.tail(12)  # 12 dernières mesures
    if len(recent_data) < 2:
        return "Données insuffisantes"
    
    fill_rates = []
    for i in range(1, len(recent_data)):
        time_diff = (recent_data.iloc[i]['ts'] - recent_data.iloc[i-1]['ts']).total_seconds() / 3600
        level_diff = recent_data.iloc[i]['fill_level'] - recent_data.iloc[i-1]['fill_level']
        if time_diff > 0 and level_diff > 0:  # Seulement si le niveau augmente
            fill_rates.append(level_diff / time_diff)
    
    if not fill_rates:
        return "Tendance stable"
    
    avg_fill_rate = np.mean(fill_rates)  # % par heure
    if avg_fill_rate <= 0:
        return "Tendance stable"
    
    # Calculer le temps pour atteindre 95%
    remaining_capacity = 95 - (current_level * 100)
    if remaining_capacity <= 0:
        return "URGENT - Déjà plein!"
    
    hours_to_full = remaining_capacity / (avg_fill_rate * 100)
    
    if hours_to_full < 6:
        return f"🔴 {hours_to_full:.1f}h"
    elif hours_to_full < 24:
        return f"🟡 {hours_to_full:.1f}h"
    else:
        return f"🟢 {hours_to_full/24:.1f}j"

def generate_operational_recommendations(fill_levels, df_positions, df_tonnage):
    """Génère des recommandations opérationnelles intelligentes"""
    recommendations = []
    
    urgent_count = sum(1 for x in fill_levels if x >= 85)
    warning_count = sum(1 for x in fill_levels if 70 <= x < 85)
    
    # Recommandations basées sur les niveaux
    if urgent_count > 0:
        recommendations.append({
            'type': 'URGENT',
            'title': f'Collecte immédiate requise',
            'description': f'{urgent_count} capteur(s) ≥85%. Risque de débordement.',
            'action': 'Déployer équipe de collecte prioritaire',
            'priority': 1
        })
    
    if warning_count >= 5:
        recommendations.append({
            'type': 'WARNING',
            'title': 'Planifier collecte préventive',
            'description': f'{warning_count} capteurs approchent de la saturation',
            'action': 'Programmer tournée dans les 4-6h',
            'priority': 2
        })
    
    # Recommandations basées sur l'efficacité
    if urgent_count + warning_count >= 8:
        recommendations.append({
            'type': 'EFFICIENCY',
            'title': 'Optimiser avec camion supplémentaire',
            'description': 'Volume élevé justifie une 2ème équipe',
            'action': 'Activer protocole multi-équipes',
            'priority': 2
        })
    
    # Recommandations basées sur les patterns
    current_hour = datetime.now().hour
    if current_hour >= 6 and current_hour <= 10 and warning_count > 2:
        recommendations.append({
            'type': 'TIMING',
            'title': 'Fenêtre optimale de collecte',
            'description': 'Période matinale idéale (trafic réduit)',
            'action': 'Prioriser collecte maintenant',
            'priority': 2
        })
    
    return sorted(recommendations, key=lambda x: x['priority'])

# Fonctions de visualisation avancées
def create_interactive_map(fill_levels, df_positions, selected_sensors=None, planned_route=None, route_geometries=None, view_mode='all', alert_threshold=80):
    """Crée la carte interactive avec sélection et routes simplifiées"""
    fig = go.Figure()
    
    if df_positions.empty:
        return fig
    
    # Filtrage selon le mode de vue
    if view_mode == 'urgent_only':
        visible_indices = [i for i, level in enumerate(fill_levels) if level >= alert_threshold]
    elif view_mode == 'planned_route':
        selected_ids = selected_sensors or []
        visible_indices = [i for i, row in df_positions.iterrows() if row['sensor_id'] in selected_ids]
    else:  # 'all'
        visible_indices = list(range(len(fill_levels)))
    
    # Préparer les données des capteurs visibles
    sensor_data = []
    for i in visible_indices:
        if i < len(df_positions):
            row = df_positions.iloc[i]
            fill_level = fill_levels[i] if i < len(fill_levels) else 50
            status = "🔴 URGENT" if fill_level >= 85 else "🟡 ATTENTION" if fill_level >= alert_threshold else "🟢 OK"
            
            sensor_data.append({
                'lat': row['lat'],
                'lon': row['lon'],
                'sensor_id': row['sensor_id'],
                'fill_level': fill_level,
                'status': status,
                'capacity': row['capacity_tons'],
                'selected': row['sensor_id'] in (selected_sensors or []),
                'index': i
            })
    
    if not sensor_data:
        st.warning(f"Aucun capteur à afficher pour le mode '{view_mode}' avec seuil {alert_threshold}%")
        return fig
    
    df_sensors = pd.DataFrame(sensor_data)
    
    # Capteurs normaux (non sélectionnés)
    normal_sensors = df_sensors[~df_sensors['selected']]
    if not normal_sensors.empty:
        colors_normal = normal_sensors['fill_level'].apply(
            lambda x: 'red' if x >= 85 else 'orange' if x >= alert_threshold else 'green'
        )
        
        fig.add_trace(go.Scattermapbox(
            lat=normal_sensors['lat'],
            lon=normal_sensors['lon'],
            mode='markers',
            marker=dict(
                size=12,
                color=colors_normal,
                opacity=0.8,
                symbol='circle'
            ),
            text=normal_sensors.apply(lambda row: 
                f"<b>{row['sensor_id']}</b><br>"
                f"Niveau: {row['fill_level']}%<br>"
                f"Status: {row['status']}<br>"
                f"📍 Cliquer pour sélectionner", axis=1),
            hovertemplate='%{text}<extra></extra>',
            name='Capteurs',
            customdata=normal_sensors['sensor_id']
        ))
    
    # Capteurs sélectionnés (plus visibles)
    selected_sensors_df = df_sensors[df_sensors['selected']]
    if not selected_sensors_df.empty:
        # Cercle blanc en arrière-plan pour l'effet de bordure
        fig.add_trace(go.Scattermapbox(
            lat=selected_sensors_df['lat'],
            lon=selected_sensors_df['lon'],
            mode='markers',
            marker=dict(
                size=22,
                color='white',
                opacity=0.9,
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Capteur sélectionné par-dessus
        fig.add_trace(go.Scattermapbox(
            lat=selected_sensors_df['lat'],
            lon=selected_sensors_df['lon'],
            mode='markers',
            marker=dict(
                size=16,
                color='purple',
                opacity=1.0,
                symbol='circle'
            ),
            text=selected_sensors_df.apply(lambda row: 
                f"<b>✓ {row['sensor_id']} SÉLECTIONNÉ</b><br>"
                f"Niveau: {row['fill_level']}%<br>"
                f"Status: {row['status']}<br>"
                f"🗑️ Cliquer pour désélectionner", axis=1),
            hovertemplate='%{text}<extra></extra>',
            name='Sélectionnés',
            customdata=selected_sensors_df['sensor_id']
        ))
    
    # Parcours planifié avec lignes directes (simplifiées)
    if planned_route and len(planned_route) > 1:
        # Debug : vérifier les données de la route
        st.write(f"🐛 DEBUG: planned_route contient {len(planned_route)} points")
        for i, point in enumerate(planned_route[:3]):  # Afficher les 3 premiers points pour debug
            st.write(f"   Point {i+1}: {point.get('sensor_id', 'NO_ID')} - Lat: {point.get('lat', 'NO_LAT')}, Lon: {point.get('lon', 'NO_LON')}")
        
        # Ajouter le dépôt au début et à la fin de la route pour affichage
        depot_point = {'lat': 48.8566, 'lon': 2.3522, 'sensor_id': 'DÉPÔT'}
        full_route = [depot_point] + planned_route + [depot_point]
        
        route_lats = [point['lat'] for point in full_route]
        route_lons = [point['lon'] for point in full_route]
        
        st.write(f"🐛 DEBUG: Route complète avec dépôt: {len(full_route)} points")
        st.write(f"   Latitudes: {route_lats[:5]}...")
        st.write(f"   Longitudes: {route_lons[:5]}...")
        
    # Tracé des vraies géométries de la route
    for point_idx in range(len(planned_route) - 1):
        segment_geom = planned_route[point_idx].get('geometry', [])
        if segment_geom:
            lons, lats = zip(*segment_geom)
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(width=4, color='blue'),
                name=f"Segment {point_idx + 1}",
                hoverinfo='skip',
                showlegend=False
            ))

    # Points de collecte avec numérotation (sans le dépôt pour la numérotation)
    for i, point in enumerate(planned_route):
        fig.add_trace(go.Scattermapbox(
            lat=[point['lat']],
            lon=[point['lon']],
            mode='markers+text',
            marker=dict(size=16, color='blue', symbol='circle'),
            text=[str(i+1)],
            textfont=dict(size=12, color='white'),
            showlegend=False,
            hovertemplate=f'<b>Étape {i+1}: {point["sensor_id"]}</b><br>' + 
                        (f'Distance réelle: {point.get("real_distance", 0):.1f}km<br>' if point.get("api_used") else 'Distance estimée<br>') +
                        f'Durée: {point.get("real_duration", 0):.0f}min<extra></extra>',
            name='Étapes'
        ))

    # Point de départ/arrivée (dépôt)
    depot_point = {'lat': 48.8566, 'lon': 2.3522, 'sensor_id': 'DÉPÔT'}
    fig.add_trace(go.Scattermapbox(
        lat=[depot_point['lat']],
        lon=[depot_point['lon']],
        mode='markers+text',
        marker=dict(size=20, color='green', symbol='star'),
        text=['🏠'],
        textfont=dict(size=16),
        name='Dépôt',
        showlegend=True,
        hovertemplate='<b>DÉPÔT DE DÉPART/RETOUR</b><extra></extra>'
    ))

    # Configuration de la carte
    center_lat = np.mean([s['lat'] for s in sensor_data]) if sensor_data else 48.8566
    center_lon = np.mean([s['lon'] for s in sensor_data]) if sensor_data else 2.3522

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12 if len(sensor_data) > 10 else 13
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )

    return fig


def create_decision_metrics(fill_levels, df_positions, df_tonnage):
    """Crée les métriques clés pour la prise de décision"""
    urgent_sensors = [i for i, level in enumerate(fill_levels) if level >= 85]
    warning_sensors = [i for i, level in enumerate(fill_levels) if 70 <= level < 85]
    
    # Calculs de capacité
    total_capacity = len(fill_levels) * 0.12  # tonnes
    current_waste = sum(fill_levels) * 0.12 / 100
    available_capacity = total_capacity - current_waste
    
    # Estimation revenus perdus si débordement
    if not df_tonnage.empty:
        avg_daily_tonnage = df_tonnage['daily_tons'].mean()
        potential_loss = len(urgent_sensors) * avg_daily_tonnage * 50  # 50€/tonne estimé
    else:
        potential_loss = len(urgent_sensors) * 25  # Estimation conservative
    
    # Temps optimal restant
    avg_urgent_level = np.mean([fill_levels[i] for i in urgent_sensors]) if urgent_sensors else 0
    time_to_overflow = max(0, (100 - avg_urgent_level) / 5)  # 5% par heure estimé
    
    return {
        'urgent_count': len(urgent_sensors),
        'warning_count': len(warning_sensors),
        'capacity_used': (current_waste / total_capacity) * 100,
        'potential_loss': potential_loss,
        'time_to_overflow': time_to_overflow,
        'total_capacity': total_capacity,
        'available_capacity': available_capacity
    }

def create_sensor_details_panel(sensor_id, fill_levels, df_positions, df_historic):
    """Panneau de détails pour un capteur sélectionné"""
    sensor_idx = int(sensor_id[1:]) - 1
    if sensor_idx >= len(fill_levels):
        return
    
    current_level = fill_levels[sensor_idx]
    sensor_pos = df_positions[df_positions['sensor_id'] == sensor_id].iloc[0]
    
    st.subheader(f"📊 Détails - {sensor_id}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Niveau Actuel", f"{current_level}%")
    with col2:
        capacity_kg = current_level * 1.2  # 120L = 120kg approximativement
        st.metric("Contenu", f"{capacity_kg:.1f} kg")
    with col3:
        status = "🔴 URGENT" if current_level >= 85 else "🟡 ATTENTION" if current_level >= 70 else "🟢 OK"
        st.metric("Status", status)
    
    # Prédiction
    prediction = predict_next_collection_time(sensor_id, df_historic, current_level/100)
    st.info(f"⏰ **Temps avant saturation:** {prediction}")
    
    # Position
    st.write(f"📍 **Position:** {sensor_pos['lat']:.6f}, {sensor_pos['lon']:.6f}")
    
    # Historique récent du capteur
    if not df_historic.empty:
        sensor_history = df_historic[df_historic['sensor_id'] == sensor_id].tail(20)
        if not sensor_history.empty:
            fig_trend = px.line(
                sensor_history, 
                x='ts', 
                y='fill_level',
                title=f"Évolution récente - {sensor_id}",
                labels={'ts': 'Temps', 'fill_level': 'Niveau'}
            )
            fig_trend.update_layout(height=250)
            st.plotly_chart(fig_trend, use_container_width=True)

# Interface principale
def main():
    # En-tête avec statut temps réel
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🗑️ Smart Waste Management - Paris")
        st.markdown("**Dashboard Interactif de Gestion Intelligente**")
    
    with col2:
        # Indicateur temps réel
        current_time = datetime.now().strftime('%H:%M:%S')
        st.metric("🕐 Temps Réel", current_time)
    
    # Chargement des données EN PREMIER (avant sidebar)
    with st.spinner("⏳ Chargement des données en temps réel..."):
        fill_levels = load_sensor_current_data()
        df_positions = load_sensor_positions()
        df_historic = load_historic_data()
        df_tonnage = load_tonnage_data()
        metrics = load_model_metrics()
    
    if not fill_levels or df_positions.empty:
        st.error("❌ Impossible de charger les données principales.")
        return
    
    # Récupération des variables de session avec valeurs par défaut
    view_mode = st.session_state.view_mode
    alert_threshold = st.session_state.alert_threshold
    
    # Sidebar avec contrôles avancés (après chargement des données)
    with st.sidebar:
        st.header("⚙️ Centre de Contrôle")
        
        # Mode de vue avec feedback
        st.subheader("👁️ Mode d'Affichage")
        new_view_mode = st.selectbox(
            "Choisir la vue:",
            ['all', 'urgent_only', 'planned_route'],
            format_func=lambda x: {
                'all': '🌍 Vue complète - Tous les capteurs',
                'urgent_only': '🚨 Urgences seulement - Focus priorité',
                'planned_route': '🚛 Parcours planifié - Trajet optimisé'
            }[x],
            index=['all', 'urgent_only', 'planned_route'].index(view_mode),
            key="view_mode_select"
        )
        
        # Mettre à jour le mode si changé
        if new_view_mode != view_mode:
            st.session_state.view_mode = new_view_mode
            view_mode = new_view_mode  # Mettre à jour la variable locale aussi
            st.rerun()
        
        # Feedback sur le mode actuel
        if view_mode == 'urgent_only':
            urgent_count = sum(1 for level in fill_levels if level >= alert_threshold)
            st.info(f"🎯 Mode actif: {urgent_count} capteurs ≥{alert_threshold}%")
        elif view_mode == 'planned_route':
            if st.session_state.planned_route:
                st.success(f"🚛 Parcours actif: {len(st.session_state.planned_route)} étapes")
            else:
                st.warning("⚠️ Aucun parcours planifié")
        else:
            st.success(f"🌍 Vue complète: {len(fill_levels)} capteurs")
        
        # Seuil d'alerte personnalisable avec feedback
        st.subheader("🔔 Seuils d'Alerte")
        new_alert_threshold = st.slider(
            "Seuil d'alerte personnalisé (%)", 
            50, 95, 
            alert_threshold,
            help="Définit le niveau à partir duquel un capteur est considéré en attention",
            key="alert_threshold_slider"
        )
        
        # Mettre à jour si changé
        if new_alert_threshold != alert_threshold:
            st.session_state.alert_threshold = new_alert_threshold
            alert_threshold = new_alert_threshold  # Mettre à jour la variable locale aussi
            # Compter les capteurs concernés
            affected_count = sum(1 for level in fill_levels if level >= alert_threshold)
            st.info(f"🔄 Seuil mis à jour: {affected_count} capteurs ≥{alert_threshold}%")
            st.rerun()
        
        # Indicateurs visuels du seuil
        current_threshold_count = sum(1 for level in fill_levels if level >= alert_threshold)
        urgent_count = sum(1 for level in fill_levels if level >= 85)
        
        col_thresh1, col_thresh2 = st.columns(2)
        with col_thresh1:
            st.metric(f"≥{alert_threshold}%", current_threshold_count, delta="À surveiller")
        with col_thresh2:
            st.metric("≥85%", urgent_count, delta="Urgents")
        
        st.markdown("---")
        
        # Actions rapides avec compteurs en temps réel
        st.subheader("⚡ Actions Rapides")
        
        # Mise à jour des compteurs
        urgent_sensors_count = sum(1 for level in fill_levels if level >= 85)
        warning_sensors_count = sum(1 for level in fill_levels if alert_threshold <= level < 85)
        selected_count = len(st.session_state.selected_sensors)
        
        if st.button(f"🚨 Mode Urgence ({urgent_sensors_count})", 
                    disabled=urgent_sensors_count == 0,
                    key="emergency_mode"):
            st.session_state.view_mode = 'urgent_only'
            urgent_sensors = [df_positions.iloc[i]['sensor_id'] for i, level in enumerate(fill_levels) if level >= 85]
            st.session_state.selected_sensors = urgent_sensors
            st.success(f"🚨 Mode urgence activé! {len(urgent_sensors)} capteurs sélectionnés")
            st.rerun()
        
        if st.button(f"🗑️ Vider Sélection ({selected_count})",
                    disabled=selected_count == 0,
                    key="clear_all"):
            st.session_state.selected_sensors = []
            st.session_state.planned_route = []
            st.session_state.route_geometries = []
            st.session_state.view_mode = 'all'
            st.success("🧹 Sélection vidée")
            st.rerun()
        
        if st.button(f"🔄 Actualiser Données",
                    key="refresh_data"):
            st.cache_data.clear()
            st.success("🔄 Données actualisées")
            st.rerun()
        
        # Statistiques en temps réel dans le sidebar
        st.markdown("---")
        st.subheader("📊 Statistiques Live")
        
        total_sensors = len(fill_levels)
        ok_sensors = total_sensors - urgent_sensors_count - warning_sensors_count
        avg_level = np.mean(fill_levels) if fill_levels else 0
        
        # Jauges compactes
        st.metric("🟢 OK", ok_sensors, delta=f"{(ok_sensors/total_sensors)*100:.0f}%")
        st.metric("🟡 Attention", warning_sensors_count, delta=f"{(warning_sensors_count/total_sensors)*100:.0f}%")
        st.metric("🔴 Urgents", urgent_sensors_count, delta=f"{(urgent_sensors_count/total_sensors)*100:.0f}%")
        st.metric("📊 Niveau Moyen", f"{avg_level:.1f}%")
        
        # Tendance (simulation)
        if avg_level > 60:
            st.warning("📈 Tendance: Augmentation")
        elif avg_level < 40:
            st.success("📉 Tendance: Stable")
        else:
            st.info("📊 Tendance: Normale")
        
        # Auto-refresh avec countdown
        st.markdown("---")
        st.subheader("🔄 Actualisation")
        
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False, key="auto_refresh_toggle")
        if auto_refresh:
            # Countdown visuel (version simplifiée pour éviter les blocages)
            st.info("⏱️ Auto-refresh activé")
            time.sleep(30)
            st.cache_data.clear()
            st.rerun()
        
        # Statut système
        st.markdown("---")
        current_time = datetime.now()
        st.success(f"🟢 **Système Opérationnel**")
        st.info(f"🕐 {current_time.strftime('%H:%M:%S')}")
        st.info(f"📅 {current_time.strftime('%d/%m/%Y')}")
        
        # Alertes sidebar
        if urgent_sensors_count > 0:
            st.error(f"🚨 {urgent_sensors_count} capteurs urgents!")
        elif warning_sensors_count > 5:
            st.warning(f"⚠️ {warning_sensors_count} capteurs à surveiller")
        else:
            st.success("✅ Situation normale")
    
    # Utilisation des variables locales mise à jour
    current_view_mode = view_mode
    current_alert_threshold = alert_threshold
    
    # Calcul des métriques de décision (après le chargement des données)
    decision_metrics = create_decision_metrics(fill_levels, df_positions, df_tonnage)
    
    # Générer les recommandations
    recommendations = generate_operational_recommendations(fill_levels, df_positions, df_tonnage)
    
    # Panel de recommandations (toujours visible en haut)
    if recommendations:
        st.subheader("🎯 Recommandations Opérationnelles")
        for rec in recommendations[:3]:  # Top 3 recommandations
            if rec['type'] == 'URGENT':
                st.error(f"🚨 **{rec['title']}** - {rec['description']} → *{rec['action']}*")
            elif rec['type'] == 'WARNING':
                st.warning(f"⚠️ **{rec['title']}** - {rec['description']} → *{rec['action']}*")
            else:
                st.info(f"💡 **{rec['title']}** - {rec['description']} → *{rec['action']}*")
    
    # Métriques globales avec focus décision
    st.subheader("📈 Tableau de Bord Décisionnel")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🔴 Urgents", 
            decision_metrics['urgent_count'],
            delta=f"-{decision_metrics['urgent_count']} à collecter" if decision_metrics['urgent_count'] > 0 else None
        )
    with col2:
        st.metric(
            "🟡 Attention", 
            decision_metrics['warning_count'],
            delta="Surveillance requise" if decision_metrics['warning_count'] > 0 else None
        )
    with col3:
        st.metric(
            "⚡ Capacité Utilisée", 
            f"{decision_metrics['capacity_used']:.1f}%",
            delta="Optimal < 70%" if decision_metrics['capacity_used'] > 70 else None
        )
    with col4:
        st.metric(
            "💰 Perte Potentielle", 
            f"{decision_metrics['potential_loss']:.0f}€",
            delta="Si débordement" if decision_metrics['potential_loss'] > 0 else None
        )
    with col5:
        st.metric(
            "⏰ Temps Critique", 
            f"{decision_metrics['time_to_overflow']:.1f}h",
            delta="Avant débordement" if decision_metrics['time_to_overflow'] < 6 else None
        )
    
    # Layout principal avec carte interactive
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Carte Interactive - Cliquez pour sélectionner")
        
        # Filtrage selon le mode de vue
        if view_mode == 'urgent_only':
            display_sensors = [i for i, level in enumerate(fill_levels) if level >= alert_threshold]
            st.info(f"🎯 Mode Focus: {len(display_sensors)} capteurs ≥{alert_threshold}%")
        else:
            display_sensors = list(range(len(fill_levels)))
        
        # Carte interactive
        map_fig = create_interactive_map(
            fill_levels, 
            df_positions, 
            st.session_state.selected_sensors,
            st.session_state.planned_route
        )
        
        # Affichage de la carte avec interaction simulée
        selected_point = st.plotly_chart(map_fig, use_container_width=True, key="main_map")
        
        # Simulation de sélection (dans un vrai déploiement, utiliser streamlit-plotly-events)
        st.info("💡 **Simulation d'interaction:** Utilisez les boutons ci-dessous pour sélectionner des capteurs")
        
        # Boutons de sélection rapide
        col_sel1, col_sel2, col_sel3 = st.columns(3)
        with col_sel1:
            if st.button("🔴 Sélectionner Urgents"):
                urgent_sensors = [df_positions.iloc[i]['sensor_id'] for i, level in enumerate(fill_levels) if level >= 85]
                st.session_state.selected_sensors = urgent_sensors
                st.rerun()
        
        with col_sel2:
            if st.button("🟡 Sélectionner Attention"):
                warning_sensors = [df_positions.iloc[i]['sensor_id'] for i, level in enumerate(fill_levels) if 70 <= level < 85]
                st.session_state.selected_sensors = warning_sensors
                st.rerun()
        
        with col_sel3:
            if st.button("🎯 Sélectionner Top 5"):
                top_sensors = sorted(enumerate(fill_levels), key=lambda x: x[1], reverse=True)[:5]
                top_sensor_ids = [df_positions.iloc[idx]['sensor_id'] for idx, level in top_sensors]
                st.session_state.selected_sensors = top_sensor_ids
                st.rerun()
    
    with col2:
        st.subheader("🎮 Centre de Contrôle")
        
        # Optimisation de parcours avec feedback
        if st.session_state.selected_sensors:
            if st.button("🚛 Optimiser Parcours", key="optimize_route"):
                # Utiliser un container pour les résultats qui persiste
                results_container = st.container()
                
                with results_container:
                    with st.spinner("🔄 Calcul du parcours optimal avec distances routières..."):
                        route, distance, time_est, route_geometries = calculate_route_optimization(
                            st.session_state.selected_sensors, 
                            df_positions
                        )
                        
                        # Sauvegarder les résultats dans session_state
                        st.session_state.planned_route = route
                        st.session_state.route_geometries = route_geometries
                        st.session_state.view_mode = 'planned_route'
                        st.session_state.route_calculated = True  # Flag pour forcer la mise à jour
                        
                        if route:
                            # Compter combien de routes utilisent l'API vs estimation
                            api_used_count = sum(1 for point in route if point.get('api_used', False))
                            
                            st.success("✅ **Parcours optimisé calculé!**")
                            
                            # Métriques principales
                            col_route1, col_route2, col_route3 = st.columns(3)
                            with col_route1:
                                st.metric("📏 Distance Totale", f"{distance:.1f} km")
                            with col_route2:
                                st.metric("⏱️ Temps Total", f"{time_est:.0f} min")
                            with col_route3:
                                st.metric("💰 Coût Estimé", f"{distance * 2:.0f}€")
                            
                            # Indicateur de qualité du routing
                            if api_used_count == len(route):
                                st.success(f"🛣️ **Routes calculées avec précision** (facteur route urbaine appliqué)")
                            else:
                                st.info(f"📏 **Routes estimées** avec correction urbaine (+30%)")
                            
                            # Détail du parcours
                            with st.expander("🗺️ Détail du Parcours Optimisé", expanded=True):
                                st.write("**Ordre de collecte optimal:**")
                                for i, point in enumerate(route):
                                    sensor_idx = int(point['sensor_id'][1:]) - 1
                                    level = fill_levels[sensor_idx] if sensor_idx < len(fill_levels) else 0
                                    distance_info = f" → {point.get('real_distance', 0):.1f}km"
                                    duration_info = f" ({point.get('real_duration', 0):.0f}min)"
                                    st.write(f"**{i+1}.** {point['sensor_id']} - Niveau: {level}% {distance_info}{duration_info}")
                                
                                # Résumé final
                                st.info(f"📍 **Résumé:** {len(route)} collectes → Distance: {distance:.1f}km → Temps: {time_est:.0f}min → Coût: {distance * 2:.0f}€")
                            
                            # Forcer la mise à jour de la carte
                            st.info("🗺️ **Carte mise à jour avec le nouveau parcours - Regardez la ligne bleue !**")
                        else:
                            st.error("❌ Impossible de calculer le parcours")
                
                # Forcer le rerun pour mettre à jour la carte
                st.rerun()
        
        # Afficher les résultats existants s'il y en a (après rerun)
        elif st.session_state.planned_route:
            st.info("📋 **Parcours déjà calculé** - Voir la carte pour la route")
            
            # Recalculer les métriques pour affichage
            total_distance = sum(point.get('real_distance', 0) for point in st.session_state.planned_route)
            total_time = sum(point.get('real_duration', 0) for point in st.session_state.planned_route) + (len(st.session_state.planned_route) * 5)
            
            col_route1, col_route2, col_route3 = st.columns(3)
            with col_route1:
                st.metric("📏 Distance", f"{total_distance:.1f} km")
            with col_route2:
                st.metric("⏱️ Temps", f"{total_time:.0f} min")
            with col_route3:
                st.metric("💰 Coût", f"{total_distance * 2:.0f}€")
            
            with st.expander("🗺️ Détail du Parcours Actuel", expanded=False):
                for i, point in enumerate(st.session_state.planned_route):
                    sensor_idx = int(point['sensor_id'][1:]) - 1
                    level = fill_levels[sensor_idx] if sensor_idx < len(fill_levels) else 0
                    distance_info = f" → {point.get('real_distance', 0):.1f}km"
                    duration_info = f" ({point.get('real_duration', 0):.0f}min)"
                    st.write(f"**{i+1}.** {point['sensor_id']} - Niveau: {level}% {distance_info}{duration_info}")
                
            if st.button("🔄 Recalculer Parcours", key="recalc_route"):
                st.session_state.planned_route = []
                st.session_state.route_geometries = []
                st.rerun()
        
        else:
            if st.button("🚛 Optimiser Parcours", key="optimize_route"):
                # Utiliser un container pour les résultats qui persiste
                results_container = st.container()
                
                with results_container:
                    with st.spinner("🔄 Calcul du parcours optimal avec distances routières..."):
                        route, distance, time_est, route_geometries = calculate_route_optimization(
                            st.session_state.selected_sensors, 
                            df_positions
                        )
                        
                        # Sauvegarder les résultats dans session_state
                        st.session_state.planned_route = route
                        st.session_state.route_geometries = route_geometries
                        st.session_state.view_mode = 'planned_route'
                        st.session_state.route_calculated = True  # Flag pour forcer la mise à jour
                        
                        if route:
                            # Compter combien de routes utilisent l'API vs estimation
                            api_used_count = sum(1 for point in route if point.get('api_used', False))
                            
                            st.success("✅ **Parcours optimisé calculé!**")
                            
                            # Métriques principales
                            col_route1, col_route2, col_route3 = st.columns(3)
                            with col_route1:
                                st.metric("📏 Distance Totale", f"{distance:.1f} km")
                            with col_route2:
                                st.metric("⏱️ Temps Total", f"{time_est:.0f} min")
                            with col_route3:
                                st.metric("💰 Coût Estimé", f"{distance * 2:.0f}€")
                            
                            # Indicateur de qualité du routing
                            if api_used_count == len(route):
                                st.success(f"🛣️ **Routes calculées avec précision** (facteur route urbaine appliqué)")
                            else:
                                st.info(f"📏 **Routes estimées** avec correction urbaine (+30%)")
                            
                            # Détail du parcours
                            with st.expander("🗺️ Détail du Parcours Optimisé", expanded=True):
                                st.write("**Ordre de collecte optimal:**")
                                for i, point in enumerate(route):
                                    sensor_idx = int(point['sensor_id'][1:]) - 1
                                    level = fill_levels[sensor_idx] if sensor_idx < len(fill_levels) else 0
                                    distance_info = f" → {point.get('real_distance', 0):.1f}km"
                                    duration_info = f" ({point.get('real_duration', 0):.0f}min)"
                                    st.write(f"**{i+1}.** {point['sensor_id']} - Niveau: {level}% {distance_info}{duration_info}")
                                
                                # Résumé final
                                st.info(f"📍 **Résumé:** {len(route)} collectes → Distance: {distance:.1f}km → Temps: {time_est:.0f}min → Coût: {distance * 2:.0f}€")
                            
                            # Forcer la mise à jour de la carte
                            st.info("🗺️ **Carte mise à jour avec le nouveau parcours - Regardez la ligne bleue !**")
                        else:
                            st.error("❌ Impossible de calculer le parcours")
                
                # Forcer le rerun pour mettre à jour la carte
                st.rerun()
            
            # Simulation de collecte avec confirmation
            st.write("---")
            st.write("**🗑️ Simulation de Collecte:**")
            
            # Vérifier si on attend une confirmation
            if not st.session_state.awaiting_confirmation:
                # Bouton initial pour déclencher la demande de confirmation
                if st.button("⚠️ Simuler Collecte Complète", key="simulate_collection"):
                    # Calculer les détails de la collecte
                    total_sensors = len(st.session_state.selected_sensors)
                    estimated_time = total_sensors * 5  # 5 min par capteur
                    
                    if st.session_state.planned_route and len(st.session_state.planned_route) > 0:
                        # Récupérer la distance déjà calculée depuis la route
                        total_distance = sum(point.get('real_distance', 0) for point in st.session_state.planned_route)
                        estimated_cost = total_distance * 2
                    else:
                        estimated_cost = total_sensors * 10
                    
                    # Stocker les détails et activer l'état de confirmation
                    st.session_state.confirmation_details = {
                        'total_sensors': total_sensors,
                        'estimated_time': estimated_time,
                        'estimated_cost': estimated_cost
                    }
                    st.session_state.awaiting_confirmation = True
                    st.rerun()
            
            else:
                # Afficher la demande de confirmation
                details = st.session_state.confirmation_details
                st.warning(f"⚠️ **Confirmer la collecte de {details['total_sensors']} capteurs?**")
                
                col_details1, col_details2 = st.columns(2)
                with col_details1:
                    st.write(f"⏱️ **Durée estimée:** {details['estimated_time']} min")
                with col_details2:
                    st.write(f"💰 **Coût estimé:** {details['estimated_cost']:.0f}€")
                
                col_conf1, col_conf2 = st.columns(2)
                
                with col_conf1:
                    if st.button("✅ Confirmer", key="confirm_collection"):
                        # Enregistrer l'action
                        st.session_state.last_action = {
                            'type': 'collection',
                            'sensors': st.session_state.selected_sensors.copy(),
                            'time': datetime.now(),
                            'estimated_time': details['estimated_time'],
                            'estimated_cost': details['estimated_cost']
                        }
                        
                        # Afficher le succès
                        st.success(f"✅ **Collecte confirmée!** {details['total_sensors']} capteurs vidés")
                        st.balloons()  # Animation de succès
                        
                        # Reset de la sélection et de l'état de confirmation
                        st.session_state.selected_sensors = []
                        st.session_state.planned_route = []
                        st.session_state.route_geometries = []
                        st.session_state.view_mode = 'all'
                        st.session_state.awaiting_confirmation = False
                        st.session_state.confirmation_details = None
                        
                        # Attendre un peu pour voir l'animation
                        time.sleep(2)
                        st.rerun()
                
                with col_conf2:
                    if st.button("❌ Annuler", key="cancel_collection"):
                        # Annuler la confirmation
                        st.session_state.awaiting_confirmation = False
                        st.session_state.confirmation_details = None
                        st.info("❌ Collecte annulée")
                        st.rerun()
        
        # Historique des actions avec plus de détails
        if st.session_state.last_action:
            st.write("---")
            st.write("**📝 Dernière Action Effectuée:**")
            action = st.session_state.last_action
            
            col_hist1, col_hist2 = st.columns(2)
            with col_hist1:
                st.write(f"🕐 **Heure:** {action['time'].strftime('%H:%M:%S')}")
                st.write(f"🎯 **Action:** {action['type'].title()}")
                st.write(f"📊 **Capteurs:** {len(action['sensors'])}")
            
            with col_hist2:
                if 'estimated_time' in action:
                    st.write(f"⏱️ **Durée:** {action['estimated_time']} min")
                if 'estimated_cost' in action:
                    st.write(f"💰 **Coût:** {action['estimated_cost']:.0f}€")
                
            # Bouton pour voir les détails
            if st.button("🔍 Voir détails", key="action_details"):
                st.write("**Capteurs traités:**")
                for sensor_id in action['sensors']:
                    st.write(f"- {sensor_id}")
        
        # Aide contextuelle
        with st.expander("💡 Aide - Comment utiliser les contrôles", expanded=False):
            st.markdown("""
            **🎯 Sélection de capteurs:**
            - Utilisez les boutons rapides (Urgents, Attention, Top 5)
            - Sélection individuelle avec la liste déroulante
            - Seuil personnalisé avec le slider
            
            **🚛 Planification de parcours:**
            1. Sélectionnez des capteurs
            2. Cliquez "Optimiser Parcours"
            3. Vérifiez l'ordre et les estimations
            4. Confirmez la collecte
            
            **👁️ Modes d'affichage:**
            - **Vue complète:** Tous les capteurs
            - **Urgences seulement:** Focus sur les priorités
            - **Parcours planifié:** Voir le trajet optimisé
            """)
    
    # Section détails des capteurs sélectionnés
    if st.session_state.selected_sensors:
        st.subheader("🔍 Analyse Détaillée des Capteurs")
        
        # Tabs pour chaque capteur sélectionné
        if len(st.session_state.selected_sensors) == 1:
            create_sensor_details_panel(
                st.session_state.selected_sensors[0], 
                fill_levels, 
                df_positions, 
                df_historic
            )
        else:
            tabs = st.tabs([f"📊 {sensor_id}" for sensor_id in st.session_state.selected_sensors[:5]])  # Max 5 tabs
            for i, sensor_id in enumerate(st.session_state.selected_sensors[:5]):
                with tabs[i]:
                    create_sensor_details_panel(sensor_id, fill_levels, df_positions, df_historic)
    
    # Métriques ML et performance système
    st.subheader("🤖 Performance & Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if metrics:
            # Métriques ML
            col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
            with col_ml1:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
            with col_ml2:
                st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
            with col_ml3:
                st.metric("R²", f"{metrics.get('r2', 0):.3f}")
            with col_ml4:
                if metrics.get('timestamp'):
                    try:
                        ts_str = metrics['timestamp']
                        if len(ts_str) == 16 and 'T' in ts_str and ts_str.endswith('Z'):
                            date_part = ts_str[:8]
                            time_part = ts_str[9:15]
                            formatted_ts = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                            ts = datetime.strptime(formatted_ts, '%Y-%m-%d %H:%M:%S')
                            st.metric("Dernière MAJ ML", ts.strftime('%H:%M:%S'))
                        else:
                            st.metric("Dernière MAJ ML", ts_str[-8:] if len(ts_str) > 8 else ts_str)
                    except Exception:
                        st.metric("Dernière MAJ ML", metrics['timestamp'])
        else:
            st.info("⏳ Métriques ML en cours de calcul...")
    
    with col2:
        # Indicateurs de performance système
        st.write("**⚡ Performances Système**")
        st.write(f"📊 Capteurs actifs: {len(fill_levels)}")
        st.write(f"🔄 Fréquence: 30s")
        st.write(f"💾 Cache: Actif")
        st.write(f"🌐 Statut: 🟢 Opérationnel")

if __name__ == "__main__":
    main()