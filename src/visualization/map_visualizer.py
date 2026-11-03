"""
Interactive Maritime Geospatial Visualizer.
Renders global shipping lanes, port bottleneck status, and maritime choke points using Folium.
"""

from typing import Dict, List, Optional, Any
import folium
from folium import plugins

from config.ports_routes import PORTS_CATALOG, CHOKE_POINTS, MAJOR_ROUTES


def get_risk_color(risk_level: str) -> str:
    """Returns hexadecimal color associated with risk level."""
    color_map = {
        "Low": "#10B981",       # Emerald
        "Medium": "#F59E0B",    # Amber
        "High": "#F97316",      # Orange
        "Critical": "#EF4444"   # Red
    }
    return color_map.get(risk_level, "#3B82F6")


def create_global_maritime_map(
    active_origin: Optional[str] = None,
    active_destination: Optional[str] = None,
    rerouted_via_cape: bool = False,
    predicted_bottleneck: Optional[str] = None
) -> folium.Map:
    """
    Constructs an interactive Folium map illustrating global maritime infrastructure,
    choke points, trade lanes, and real-time port bottleneck risks.
    """
    # Initialize Folium Map centered on equatorial trade belt
    m = folium.Map(
        location=[20.0, 30.0],
        zoom_start=3,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True
    )

    # 1. Feature Groups for clean layer toggling
    fg_routes = folium.FeatureGroup(name="Major Shipping Lanes", show=True)
    fg_choke = folium.FeatureGroup(name="Maritime Choke Points", show=True)
    fg_ports = folium.FeatureGroup(name="Global Container Ports", show=True)
    fg_active = folium.FeatureGroup(name="Active Evaluated Voyage", show=True)

    # 2. Add Baseline Shipping Lanes
    for route in MAJOR_ROUTES:
        orig = PORTS_CATALOG.get(route["origin"])
        dest = PORTS_CATALOG.get(route["destination"])
        if not orig or not dest:
            continue

        # Check if this route is currently selected by user
        is_selected = (route["origin"] == active_origin and route["destination"] == active_destination)

        # Approximate maritime path coordinates
        points = [[orig["lat"], orig["lon"]]]

        # Intermediate waypoint approximation for maritime realism
        if "Suez_Canal" in route["choke_points"]:
            points.append([12.58, 43.33])   # Bab el-Mandeb
            points.append([30.58, 32.26])   # Suez Canal
            points.append([36.00, 14.00])   # Central Med
            points.append([36.00, -5.35])   # Gibraltar
        elif "Panama_Canal" in route["choke_points"]:
            points.append([9.08, -79.68])   # Panama
        
        points.append([dest["lat"], dest["lon"]])

        if not is_selected:
            folium.PolyLine(
                locations=points,
                color="#38BDF8",
                weight=2.0,
                opacity=0.45,
                dash_array="4, 6",
                tooltip=f"<b>{route['route_name']}</b><br>Distance: {route['distance_nm']} NM | Baseline: {route['baseline_transit_days']} Days"
            ).add_to(fg_routes)

    # 3. Add Active Evaluated Voyage (Highlighted with glow)
    if active_origin and active_destination:
        orig = PORTS_CATALOG.get(active_origin)
        dest = PORTS_CATALOG.get(active_destination)
        if orig and dest:
            active_points = [[orig["lat"], orig["lon"]]]

            if rerouted_via_cape:
                # Cape of Good Hope routing points
                active_points.append([1.35, 103.82])    # Malacca
                active_points.append([-34.50, 20.00])   # Cape of Good Hope
                active_points.append([-10.00, -5.00])   # Mid-Atlantic
                active_points.append([36.00, -9.00])    # Off Portugal
                route_desc = "Cape of Good Hope Reroute (+10 to 14 days transit)"
                route_color = "#EF4444"  # Red detour
            else:
                if any("Suez" in r["choke_points"] for r in MAJOR_ROUTES if r["origin"] == active_origin and r["destination"] == active_destination):
                    active_points.append([12.58, 43.33])
                    active_points.append([30.58, 32.26])
                    active_points.append([36.00, 14.00])
                    active_points.append([36.00, -5.35])
                elif any("Panama" in r["choke_points"] for r in MAJOR_ROUTES if r["origin"] == active_origin and r["destination"] == active_destination):
                    active_points.append([9.08, -79.68])
                route_desc = "Primary Corridor Transit"
                route_color = "#F59E0B"  # Vibrant Amber

            active_points.append([dest["lat"], dest["lon"]])

            folium.PolyLine(
                locations=active_points,
                color=route_color,
                weight=5.0,
                opacity=0.90,
                dash_array="8, 8" if rerouted_via_cape else None,
                tooltip=f"<b>ACTIVE VOYAGE: {orig['name']} ➔ {dest['name']}</b><br>{route_desc}"
            ).add_to(fg_active)

    # 4. Add Maritime Choke Points
    for cp_id, cp in CHOKE_POINTS.items():
        is_red_sea = cp_id in ["Suez_Canal", "Bab_el_Mandeb"]
        icon_color = "red" if is_red_sea else "orange"
        
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 220px;">
            <h4 style="margin: 0; color: #DC2626;">⚠ {cp['name']}</h4>
            <p style="margin: 4px 0 8px 0; font-size: 12px; color: #555;">Location: {cp['location']}</p>
            <div style="font-size: 12px; margin-bottom: 4px;"><b>Daily Capacity:</b> {cp['daily_transit_capacity']} vessels</div>
            <div style="font-size: 12px; margin-bottom: 4px;"><b>Criticality:</b> <span style="color:red; font-weight:bold;">{cp['criticality']}</span></div>
            <div style="font-size: 11px; color: #444; border-top: 1px solid #eee; padding-top: 4px;">
                <b>Vulnerabilities:</b><br>• {'<br>• '.join(cp['vulnerability_factors'])}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[cp["lat"], cp["lon"]],
            radius=11,
            color="#EF4444" if cp["criticality"] in ["Critical", "Severe"] else "#F59E0B",
            weight=3,
            fill=True,
            fill_color="#EF4444" if cp["criticality"] in ["Critical", "Severe"] else "#F59E0B",
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"<b>Choke Point:</b> {cp['name']} ({cp['criticality']} Risk)"
        ).add_to(fg_choke)

    # 5. Add Global Container Ports
    for code, port in PORTS_CATALOG.items():
        # Determine port risk coloring
        if code == active_destination and predicted_bottleneck:
            port_risk = predicted_bottleneck
        else:
            # Baseline realistic risk distribution
            if port["baseline_dwell_days"] > 4.2:
                port_risk = "High"
            elif port["baseline_dwell_days"] > 3.2:
                port_risk = "Medium"
            else:
                port_risk = "Low"

        marker_color = get_risk_color(port_risk)

        port_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 230px;">
            <h4 style="margin: 0; color: #1E3A8A;">⚓ {port['name']} ({code})</h4>
            <p style="margin: 4px 0; font-size: 12px; color: #666;">{port['city']}, {port['country']} | {port['region']}</p>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin: 6px 0;">
                <span><b>Bottleneck Risk:</b></span>
                <span style="font-weight: bold; color: {marker_color};">{port_risk}</span>
            </div>
            <div style="font-size: 12px; margin-bottom: 2px;"><b>Avg Dwell Time:</b> {port['baseline_dwell_days']} days</div>
            <div style="font-size: 12px; margin-bottom: 2px;"><b>Annual Capacity:</b> {port['annual_teu_capacity_millions']}M TEU</div>
            <div style="font-size: 12px;"><b>Berths:</b> {port['berths']} container berths</div>
        </div>
        """

        folium.CircleMarker(
            location=[port["lat"], port["lon"]],
            radius=8 if code != active_destination else 13,
            color="#FFFFFF" if code in [active_origin, active_destination] else marker_color,
            weight=2 if code not in [active_origin, active_destination] else 3,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85,
            popup=folium.Popup(port_html, max_width=320),
            tooltip=f"<b>{port['name']}</b> ({port_risk} Risk)"
        ).add_to(fg_ports)

    # Add all feature groups and layer control
    fg_routes.add_to(m)
    fg_active.add_to(m)
    fg_choke.add_to(m)
    fg_ports.add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m
