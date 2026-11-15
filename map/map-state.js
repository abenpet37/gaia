(() => {
    const mapElement = document.querySelector(".folium-map");
    const map = mapElement && window[mapElement.id];
    if (!(map instanceof L.Map) || window.parent === window) {
        return;
    }

    const params = new URLSearchParams(window.location.search);
    map.setMinZoom(2);
    map.setMaxZoom(18);

    // Initial view coordinates: check URL params or default to East African Rift
    const initialLat = Number(params.get("mapLat"));
    const initialLng = Number(params.get("mapLng"));
    const initialZoom = Number(params.get("mapZoom"));
    if (
        params.has("mapLat") && params.has("mapLng") && params.has("mapZoom") &&
        Number.isFinite(initialLat) && Number.isFinite(initialLng) && Number.isFinite(initialZoom) &&
        initialLat >= -85 && initialLat <= 85 && initialLng >= -180 && initialLng <= 180 &&
        initialZoom >= 2 && initialZoom <= 18
    ) {
        map.setView([initialLat, initialLng], initialZoom, { animate: false });
    } else {
        map.setView([9.0, 38.75], 6, { animate: false });
    }

    // State variables
    let overlayEnabled = true;
    let overlayLayer = null;
    let overlayOpacity = 0.75;
    let activeOverlayConfig = null;
    let overlayConfigs = {};
    let selectedOverlay = null;

    // Earthquake and Fault layer groups
    const earthquakeLayer = L.layerGroup().addTo(map);
    const faultsLayer = L.layerGroup().addTo(map);
    const infrastructureLayer = L.layerGroup().addTo(map);
    const simulationLayer = L.layerGroup().addTo(map);

    let earthquakesEnabled = true;
    let earthquakeMinMag = "2.5";
    let earthquakePeriod = "month";
    let faultsEnabled = true;
    let infrastructureEnabled = true;

    // Radar Band State
    let activeRadarBand = "C-Band"; // "C-Band" or "L-Band"
    let autoNDVI = true;

    const publishStatus = (type, message, extra = {}) => {
        window.parent.postMessage({ type, message, ...extra }, "*");
    };

    // Custom CSS for map legend and custom pulsing markers
    const customStyle = document.createElement("style");
    customStyle.textContent = `
        .map-legend-box {
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid #14465F;
            border-radius: 8px;
            color: #f1f5f9;
            padding: 10px 14px;
            font-size: 11px;
            line-height: 1.45;
            box-shadow: 0 4px 14px rgba(0,0,0,0.6);
            max-width: 220px;
            backdrop-filter: blur(4px);
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 7px;
            margin-top: 4px;
        }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            border: 1px solid rgba(255,255,255,0.7);
            flex-shrink: 0;
        }
        .legend-bar {
            width: 16px;
            height: 4px;
            display: inline-block;
            border-radius: 2px;
            flex-shrink: 0;
        }
        /* Leaflet circle markers are SVG paths: without a fill-box origin, CSS scale()
           pivots on the SVG's top-left corner and the dot appears to slide across the map. */
        .pulsing-marker {
            transform-box: fill-box;
            transform-origin: center;
            animation: pulse-ring 2s infinite cubic-bezier(0.215, 0.61, 0.355, 1);
        }
        @keyframes pulse-ring {
            0% { transform: scale(0.95); opacity: 0.9; }
            50% { transform: scale(1.15); opacity: 0.4; }
            100% { transform: scale(0.95); opacity: 0.9; }
        }
    `;
    document.head.appendChild(customStyle);

    // Map Legend Control
    const legendControl = L.control({ position: "bottomright" });
    legendControl.onAdd = () => {
        const div = L.DomUtil.create("div", "map-legend-box");
        div.innerHTML = `
            <div style="font-weight: 700; margin-bottom: 5px; color: #54C7EC; font-size: 12px; display: flex; align-items: center; gap: 5px;">
                🛰️ Multi-Sensor Legend
            </div>
            <div style="font-weight: 600; color: #94a3b8; margin-top: 4px;">Hazard Risk Status</div>
            <div class="legend-item"><span class="legend-color" style="background:#10b981;"></span> Green: Baseline (0-29)</div>
            <div class="legend-item"><span class="legend-color" style="background:#f59e0b;"></span> Yellow: Monitor (30-54)</div>
            <div class="legend-item"><span class="legend-color" style="background:#f97316;"></span> Orange: Watch (55-79)</div>
            <div class="legend-item"><span class="legend-color" style="background:#ef4444;"></span> Red: Evacuate (80-100)</div>
            <div style="font-weight: 600; color: #94a3b8; margin-top: 6px;">Critical Infrastructure</div>
            <div class="legend-item"><span class="legend-color" style="background:#dc2626; border-radius:3px;"></span> Hospital / Health Post</div>
            <div class="legend-item"><span class="legend-color" style="background:#3b82f6; border-radius:3px;"></span> School / Shelter</div>
            <div class="legend-item"><span class="legend-bar" style="background:#54C7EC; height:3px;"></span> Strategic Highway</div>
            <div style="font-weight: 600; color: #94a3b8; margin-top: 6px;">Earthquakes (Depth)</div>
            <div class="legend-item"><span class="legend-color" style="background:#ef4444;"></span> &lt;30 km Shallow (High Risk)</div>
            <div class="legend-item"><span class="legend-color" style="background:#eab308;"></span> 70-150 km Deep</div>
        `;
        return div;
    };
    legendControl.addTo(map);

    // InSAR raster overlay updater
    const updateOverlay = () => {
        const overlayConfig = overlayConfigs[selectedOverlay];
        if (!overlayConfig) return;

        if (overlayEnabled) {
            if (activeOverlayConfig !== overlayConfig) {
                if (overlayLayer) map.removeLayer(overlayLayer);
                overlayLayer = L.imageOverlay(
                    overlayConfig.imageUrl,
                    overlayConfig.bounds,
                    { opacity: overlayOpacity, interactive: false }
                ).addTo(map);
                activeOverlayConfig = overlayConfig;
            } else if (overlayLayer) {
                overlayLayer.setOpacity(overlayOpacity);
            }
        } else if (!overlayEnabled && overlayLayer) {
            map.removeLayer(overlayLayer);
            overlayLayer = null;
            activeOverlayConfig = null;
        }
        publishStatus(
            "raster-overlay-status",
            overlayEnabled
                ? `${overlayConfig.label || selectedOverlay} active (Opacity: ${Math.round(overlayOpacity * 100)}%)`
                : "InSAR overlay disabled"
        );
    };

    // Load Active Fault Traces
    const loadActiveFaults = () => {
        const renderFaultGeoJson = (data, sourceName) => {
            faultsLayer.clearLayers();
            L.geoJSON(data, {
                style: {
                    color: "#f59e0b",
                    weight: 2.8,
                    opacity: 0.9
                },
                onEachFeature: (feature, layer) => {
                    const p = feature.properties || {};
                    const tooltipHtml = `
                        <div style="font-family: Arial, sans-serif; min-width: 170px;">
                            <div style="font-weight: 700; color: #d97706; margin-bottom: 2px;">${p.name || 'Active Fault'}</div>
                            <div style="font-size: 11px; color: #475569;"><b>System:</b> ${p.rift_system || 'Tectonic Fault'}</div>
                            <div style="font-size: 11px; color: #475569;"><b>Type:</b> ${p.type || 'Normal/Strike-slip'}</div>
                            <div style="font-size: 11px; color: #475569;"><b>Slip Rate:</b> ${p.slip_rate || 'Active'}</div>
                            <div style="font-size: 11px; color: #dc2626;"><b>Hazard:</b> ${p.hazard || 'Quaternary Active'}</div>
                        </div>
                    `;
                    layer.bindTooltip(tooltipHtml, { sticky: true });
                }
            }).addTo(faultsLayer);
            publishStatus("faults-loaded", `Active fault traces loaded (${sourceName})`, { count: (data.features || []).length });
        };

        // Attempt to load dynamic window clip first, then fall back to baseline active_faults.json
        fetch("faults_clip.geojson")
            .then(res => {
                if (!res.ok) throw new Error("No faults_clip.geojson");
                return res.json();
            })
            .then(data => renderFaultGeoJson(data, "Dynamic Clip"))
            .catch(() => {
                fetch("active_faults.json")
                    .then(res => {
                        if (!res.ok) throw new Error("Fault file not accessible via HTTP");
                        return res.json();
                    })
                    .then(data => renderFaultGeoJson(data, "Master Archive"))
                    .catch(err => console.warn("Could not load local active_faults.json:", err));
            });
    };
    loadActiveFaults();

    // Load Critical Infrastructure Exposure Layer
    const loadInfrastructure = () => {
        fetch("data/critical_infrastructure_ethiopia.geojson")
            .then(res => {
                if (!res.ok) throw new Error("Infrastructure file not found");
                return res.json();
            })
            .then(data => {
                infrastructureLayer.clearLayers();
                L.geoJSON(data, {
                    pointToLayer: (feature, latlng) => {
                        const p = feature.properties || {};
                        let markerColor = "#3b82f6";
                        let iconSymbol = "🏢";

                        if (p.category === "Healthcare") {
                            markerColor = "#dc2626";
                            iconSymbol = "🏥";
                        } else if (p.category && p.category.includes("Education")) {
                            markerColor = "#f59e0b";
                            iconSymbol = "🏫";
                        } else if (p.category && p.category.includes("Settlement")) {
                            markerColor = "#10b981";
                            iconSymbol = "🏘️";
                        } else if (p.category && p.category.includes("Critical Infrastructure")) {
                            markerColor = "#06b6d4";
                            iconSymbol = "🌊";
                        }

                        const marker = L.circleMarker(latlng, {
                            radius: 7,
                            fillColor: markerColor,
                            color: "#ffffff",
                            weight: 1.5,
                            fillOpacity: 0.9
                        });

                        const popupHtml = `
                            <div style="font-family: Arial, sans-serif; min-width: 200px;">
                                <div style="font-size: 13px; font-weight: 700; color: #011829; margin-bottom: 3px;">
                                    ${iconSymbol} ${p.name}
                                </div>
                                <div style="font-size: 11px; color: #64748b; margin-bottom: 5px;">
                                    <b>Category:</b> ${p.category} | ${p.type}
                                </div>
                                <table style="font-size: 11px; width: 100%; border-collapse: collapse;">
                                    ${p.capacity_beds ? `<tr><td style="color:#64748b;">Beds:</td><td><b>${p.capacity_beds}</b></td></tr>` : ''}
                                    ${p.population ? `<tr><td style="color:#64748b;">Population:</td><td><b>${p.population.toLocaleString()}</b></td></tr>` : ''}
                                    ${p.in_hazard_zone ? `<tr><td colspan="2" style="color:#dc2626; font-weight:bold;">⚠️ In High-Risk Landslide Hazard Corridor</td></tr>` : ''}
                                    ${p.in_volcanic_hazard_corridor ? `<tr><td colspan="2" style="color:#ea580c; font-weight:bold;">🌋 In Active Volcanic Dike Corridor</td></tr>` : ''}
                                </table>
                            </div>
                        `;
                        marker.bindPopup(popupHtml);
                        marker.bindTooltip(`${iconSymbol} ${p.name}`, { direction: "top" });
                        return marker;
                    },
                    style: (feature) => {
                        const p = feature.properties || {};
                        if (feature.geometry.type === "LineString") {
                            return {
                                color: p.in_hazard_zone ? "#f43f5e" : "#54C7EC",
                                weight: 3.5,
                                opacity: 0.85,
                                dashArray: p.in_hazard_zone ? "4, 6" : null
                            };
                        }
                        return {};
                    }
                }).addTo(infrastructureLayer);
            })
            .catch(err => {
                console.warn("Could not load critical infrastructure:", err);
            });
    };
    loadInfrastructure();

    // Fetch USGS Earthquakes
    const getDepthColor = (depth) => {
        if (depth < 30) return "#ef4444";
        if (depth < 70) return "#f97316";
        if (depth < 150) return "#eab308";
        return "#3b82f6";
    };

    const loadEarthquakes = () => {
        if (!earthquakesEnabled) {
            earthquakeLayer.clearLayers();
            publishStatus("earthquakes-status", "Earthquake feed off", { count: 0 });
            return;
        }

        publishStatus("earthquakes-status", "Fetching latest USGS seismic data...", { loading: true });
        const magFeed = Number(earthquakeMinMag) >= 4.5 ? "4.5" : Number(earthquakeMinMag) >= 2.5 ? "2.5" : "all";
        const periodFeed = earthquakePeriod === "month" ? "month" : "week";
        const feedUrl = `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/${magFeed}_${periodFeed}.geojson`;

        fetch(feedUrl)
            .then(res => {
                if (!res.ok) throw new Error(`USGS HTTP ${res.status}`);
                return res.json();
            })
            .then(data => {
                earthquakeLayer.clearLayers();
                const minM = parseFloat(earthquakeMinMag) || 2.5;
                const features = (data.features || []).filter(f => (f.properties.mag || 0) >= minM);

                features.forEach(f => {
                    const coords = f.geometry.coordinates;
                    const lon = coords[0];
                    const lat = coords[1];
                    const depth = coords[2] || 0;
                    const mag = f.properties.mag || 0;
                    const place = f.properties.place || "Unknown location";
                    const time = new Date(f.properties.time);
                    const url = f.properties.url;

                    const radius = Math.min(24, Math.max(4.5, (mag - 1.5) * 3.5));
                    const marker = L.circleMarker([lat, lon], {
                        radius: radius,
                        fillColor: getDepthColor(depth),
                        color: "#ffffff",
                        weight: 1.2,
                        opacity: 0.95,
                        fillOpacity: 0.8
                    });

                    const popupContent = `
                        <div style="font-family: Arial, sans-serif; min-width: 190px;">
                            <div style="font-size: 14px; font-weight: bold; color: #b91c1c; margin-bottom: 3px;">
                                M ${mag.toFixed(1)} Earthquake
                            </div>
                            <div style="font-size: 12px; color: #022842; margin-bottom: 6px;">${place}</div>
                            <table style="font-size: 11px; width: 100%; border-collapse: collapse; line-height: 1.5;">
                                <tr><td style="color:#64748b;">Depth:</td><td><b>${depth.toFixed(1)} km</b></td></tr>
                                <tr><td style="color:#64748b;">Time:</td><td>${time.toUTCString()}</td></tr>
                                <tr><td style="color:#64748b;">Coords:</td><td>${lat.toFixed(3)}°, ${lon.toFixed(3)}°</td></tr>
                            </table>
                            <div style="margin-top: 6px; border-top: 1px solid #B2D4E6; padding-top: 4px;">
                                <a href="${url}" target="_blank" style="color: #1B84AD; font-size: 11px; text-decoration: underline;">
                                    View on USGS &rarr;
                                </a>
                            </div>
                        </div>
                    `;
                    marker.bindPopup(popupContent);
                    marker.bindTooltip(`M${mag.toFixed(1)} - ${place}`, { direction: "top" });
                    marker.addTo(earthquakeLayer);
                });

                publishStatus("earthquakes-status", `${features.length} earthquakes loaded (USGS ${periodFeed})`, {
                    count: features.length,
                    loading: false
                });
            })
            .catch(err => {
                publishStatus("earthquakes-status", `Earthquake feed offline or network blocked`, {
                    count: 0,
                    loading: false,
                    error: true
                });
            });
    };
    loadEarthquakes();

    // Check for Automated InSAR Trigger Manifest
    const checkTriggerEvent = () => {
        fetch("data/latest_trigger_event.json")
            .then(res => {
                if (!res.ok) throw new Error("No trigger event");
                return res.json();
            })
            .then(data => {
                if (!data || !data.epicenter) return;
                const epi = data.epicenter;
                const marker = L.circleMarker([epi.latitude, epi.longitude], {
                    radius: 14,
                    fillColor: "#ef4444",
                    color: "#ffffff",
                    weight: 3,
                    opacity: 1.0,
                    fillOpacity: 0.85,
                    className: "pulsing-marker"
                }).addTo(earthquakeLayer);

                const popupHtml = `
                    <div style="font-family: Arial, sans-serif; min-width: 210px;">
                        <div style="font-size: 13px; font-weight: bold; color: #dc2626; margin-bottom: 3px;">
                            ⚡ Active InSAR Trigger Event
                        </div>
                        <div style="font-size: 12px; font-weight: 600; color: #011829;">
                            M ${data.magnitude} - ${data.place}
                        </div>
                        <div style="font-size: 11px; color: #64748b; margin: 4px 0;">
                            <b>Trigger Time:</b> ${data.time_utc}
                        </div>
                        <table style="font-size: 11px; width: 100%; border-collapse: collapse; line-height: 1.4;">
                            <tr><td style="color:#64748b;">Depth:</td><td><b>${data.depth_km} km</b></td></tr>
                            <tr><td style="color:#64748b;">Co-seismic Window:</td><td><b>${data.temporal_window ? data.temporal_window.start_date + ' &rarr; ' + data.temporal_window.end_date : 'N/A'}</b></td></tr>
                            <tr><td style="color:#64748b;">NISAR Granules:</td><td><b style="color:#1B84AD;">${data.matching_granules_count} matched</b></td></tr>
                        </table>
                    </div>
                `;
                marker.bindPopup(popupHtml);
                publishStatus("trigger-event-active", `Trigger Event: M${data.magnitude} - ${data.place}`, {
                    event: data
                });
            })
            .catch(() => {});
    };
    checkTriggerEvent();

    // Render Historical Case Study Simulation Footprints
    const renderSimulationStep = (caseStudyId, stepData) => {
        simulationLayer.clearLayers();
        if (!stepData) return;

        const riskLevel = stepData.risk_level || "GREEN";
        const riskColor = riskLevel === "RED" ? "#ef4444" : riskLevel === "ORANGE" ? "#f97316" : riskLevel === "YELLOW" ? "#f59e0b" : "#10b981";

        if (caseStudyId === "afar_hayli_gubbi_2025") {
            // Case Study A: Afar Hayli Gubbi Volcanic Dike Propagation
            const center = [13.51, 40.72];
            const ertaAle = [13.60, 40.67];
            const hayliGubbi = [13.51, 40.72];
            const progress = (stepData.week_index || 1) / 5.0;

            // Interpolate active dike tip
            const currentTip = [
                ertaAle[0] + (hayliGubbi[0] - ertaAle[0]) * progress,
                ertaAle[1] + (hayliGubbi[1] - ertaAle[1]) * progress
            ];

            // Dike intrusion line
            const dikeLine = L.polyline([ertaAle, currentTip], {
                color: riskColor,
                weight: 5 + (progress * 4),
                opacity: 0.95,
                dashArray: stepData.week_index < 5 ? "8, 6" : null
            }).addTo(simulationLayer);

            // Evacuation / Hazard Buffer Circle around active front
            const bufferRadiusM = 3000 + (progress * 12000); // 3 km to 15 km
            const bufferCircle = L.circle(currentTip, {
                radius: bufferRadiusM,
                fillColor: riskColor,
                fillOpacity: 0.18,
                color: riskColor,
                weight: 2,
                dashArray: "6, 6"
            }).addTo(simulationLayer);

            bufferCircle.bindTooltip(`<b>Dike Hazard Buffer:</b> ${(bufferRadiusM / 1000).toFixed(1)} km radius`, { sticky: true });

            // Hayli Gubbi Volcano Marker
            const volcanoMarker = L.circleMarker(hayliGubbi, {
                radius: 12,
                fillColor: riskLevel === "RED" ? "#b91c1c" : "#f59e0b",
                color: "#ffffff",
                weight: 2.5,
                fillOpacity: 0.95
            }).addTo(simulationLayer);

            volcanoMarker.bindPopup(`
                <div style="font-family: Arial, sans-serif; min-width: 190px;">
                    <div style="font-size: 13px; font-weight: bold; color: #dc2626; margin-bottom: 2px;">
                        🌋 Hayli Gubbi Shield Volcano
                    </div>
                    <div style="font-size: 11px; color: #475569; margin-bottom: 4px;">
                        Dormant ~12,000 yrs until Nov 23, 2025 eruption
                    </div>
                    <table style="font-size: 11px; width: 100%;">
                        <tr><td style="color:#64748b;">Crustal Extension:</td><td><b>+${stepData.insar_extension_cum_mm} mm</b></td></tr>
                        <tr><td style="color:#64748b;">TROPOMI SO2:</td><td><b>${stepData.tropomi_so2_du} DU</b></td></tr>
                        <tr><td style="color:#64748b;">Risk Level:</td><td><b style="color:${riskColor};">${riskLevel}</b></td></tr>
                    </table>
                </div>
            `);

        } else if (caseStudyId === "gofa_highlands_mudslide_2024") {
            // Case Study B: Gofa Highlands Catastrophic Mudslide
            const crownCenter = [6.315, 36.848];
            const valleyToe = [6.328, 36.858];
            const progress = (stepData.week_index || 1) / 5.0;

            // Crown tension crack polygon
            const crownPoly = L.polygon([
                [6.312, 36.845],
                [6.318, 36.845],
                [6.317, 36.852],
                [6.311, 36.850]
            ], {
                color: riskColor,
                fillColor: riskColor,
                fillOpacity: 0.35,
                weight: 2.5
            }).addTo(simulationLayer);

            // Downstream debris flow runout path
            if (stepData.week_index >= 3) {
                const runoutPoly = L.polygon([
                    [6.314, 36.847],
                    [6.318, 36.849],
                    [6.324, 36.854],
                    [6.330, 36.861],
                    [6.327, 36.864],
                    [6.320, 36.856],
                    [6.312, 36.849]
                ], {
                    color: "#dc2626",
                    fillColor: "#ef4444",
                    fillOpacity: 0.25 + (progress * 0.2),
                    weight: 2,
                    dashArray: "4, 4"
                }).addTo(simulationLayer);

                runoutPoly.bindTooltip(`<b>Downstream Mudslide Runout Path:</b> 3.4 km`, { sticky: true });
            }

            // Evacuation buffer around Kencho Shacha village
            const bufferM = 1500 + (progress * 2500);
            const evacBuffer = L.circle(crownCenter, {
                radius: bufferM,
                color: riskColor,
                fillColor: riskColor,
                fillOpacity: 0.15,
                weight: 2,
                dashArray: "6, 6"
            }).addTo(simulationLayer);

            evacBuffer.bindTooltip(`<b>Hazard Buffer:</b> ${(bufferM / 1000).toFixed(1)} km radius`, { sticky: true });
        }
    };

    // Check Terrain NDVI and Auto-select Radar Band
    const checkTerrainNDVI = (lat, lng) => {
        // Ethiopian Highlands (SNNPR, Gofa, Amhara highlands) have dense vegetation: NDVI ~ 0.65 - 0.85
        // Afar Depression, Danakil, and Red Sea coast are arid: NDVI ~ 0.05 - 0.20
        const isHighland = (lat < 12.0 && lng > 36.0 && lng < 39.5 && !(lat > 9.5 && lng > 39.8));
        const estimatedNDVI = isHighland ? 0.74 : 0.14;

        if (autoNDVI) {
            const previousBand = activeRadarBand;
            if (estimatedNDVI > 0.45) {
                activeRadarBand = "L-Band";
            } else {
                activeRadarBand = "C-Band";
            }
            publishStatus("radar-band-update", "Radar band updated by NDVI", {
                band: activeRadarBand,
                ndvi: estimatedNDVI,
                badge: activeRadarBand === "L-Band"
                    ? "L-Band Canopy Penetration Active (High Coherence)"
                    : "C-Band Arid Surface Scattering Active",
                isAuto: true
            });
        }
    };

    // Message receiver from parent window
    window.addEventListener("message", (event) => {
        if (event.source !== window.parent || !event.data) return;
        const data = event.data;

        if (data.type === "deformation-overlay") {
            if (typeof data.enabled === "boolean") overlayEnabled = data.enabled;
            if (typeof data.opacity === "number") overlayOpacity = Math.max(0.1, Math.min(1.0, data.opacity));
            if (typeof data.layer === "string" && overlayConfigs[data.layer]) selectedOverlay = data.layer;
            updateOverlay();
            return;
        }

        if (data.type === "fit-overlay-bounds") {
            const config = overlayConfigs[selectedOverlay] || activeOverlayConfig;
            if (config && config.bounds) {
                map.fitBounds(config.bounds, { padding: [20, 20], maxZoom: 14 });
            }
            return;
        }

        if (data.type === "set-view") {
            if (Number.isFinite(data.lat) && Number.isFinite(data.lng) && Number.isFinite(data.zoom)) {
                map.setView([data.lat, data.lng], data.zoom, { animate: true });
                checkTerrainNDVI(data.lat, data.lng);
            }
            return;
        }

        if (data.type === "earthquakes-config") {
            if (typeof data.enabled === "boolean") earthquakesEnabled = data.enabled;
            if (data.minMag) earthquakeMinMag = String(data.minMag);
            if (data.period) earthquakePeriod = String(data.period);
            loadEarthquakes();
            return;
        }

        if (data.type === "faults-toggle") {
            if (typeof data.enabled === "boolean") {
                faultsEnabled = data.enabled;
                if (faultsEnabled) {
                    if (!map.hasLayer(faultsLayer)) faultsLayer.addTo(map);
                } else {
                    if (map.hasLayer(faultsLayer)) map.removeLayer(faultsLayer);
                }
            }
            return;
        }

        if (data.type === "infrastructure-toggle") {
            if (typeof data.enabled === "boolean") {
                infrastructureEnabled = data.enabled;
                if (infrastructureEnabled) {
                    if (!map.hasLayer(infrastructureLayer)) infrastructureLayer.addTo(map);
                } else {
                    if (map.hasLayer(infrastructureLayer)) map.removeLayer(infrastructureLayer);
                }
            }
            return;
        }

        if (data.type === "set-radar-band") {
            if (data.band) activeRadarBand = data.band;
            if (typeof data.autoNDVI === "boolean") autoNDVI = data.autoNDVI;
            const center = map.getCenter();
            checkTerrainNDVI(center.lat, center.lng);
            return;
        }

        if (data.type === "replay-step") {
            renderSimulationStep(data.caseStudyId, data.stepData);
            return;
        }
    });

    // Load deformation-overlay.js config
    const overlayScript = document.createElement("script");
    overlayScript.src = "../trans/deformation-overlay.js";
    overlayScript.onload = () => {
        const candidate = window.DEFORMATION_OVERLAY;
        if (candidate && candidate.layers && typeof candidate.layers === "object") {
            overlayConfigs = candidate.layers;
        } else if (
            candidate &&
            typeof candidate.imageUrl === "string" &&
            Array.isArray(candidate.bounds) &&
            candidate.bounds.length === 2
        ) {
            overlayConfigs = { DEFAULT: candidate };
        }
        const availableOverlays = Object.keys(overlayConfigs).filter((name) => {
            const config = overlayConfigs[name];
            return (
                config &&
                typeof config.imageUrl === "string" &&
                Array.isArray(config.bounds) &&
                config.bounds.length === 2
            );
        });
        if (availableOverlays.length) {
            if (!availableOverlays.includes(selectedOverlay)) {
                selectedOverlay = availableOverlays[0];
            }
            window.parent.postMessage(
                {
                    type: "raster-overlay-ready",
                    message: "InSAR deformation layers ready",
                    layers: availableOverlays.map((name) => ({
                        name,
                        label: overlayConfigs[name].label || name,
                        bounds: overlayConfigs[name].bounds
                    }))
                },
                "*"
            );
            updateOverlay();
            return;
        }
        publishStatus("raster-overlay-error", "No valid deformation raster overlays configured.");
    };
    overlayScript.onerror = () => {
        publishStatus("raster-overlay-error", "Deformation overlay file missing. Run trans/prepare_overlay.py --mosaic.");
    };
    document.head.appendChild(overlayScript);

    // Publish map view changes back to parent
    const token = Number(params.get("mapToken") || 0);
    const publishView = () => {
        const center = map.getCenter();
        checkTerrainNDVI(center.lat, center.lng);
        window.parent.postMessage(
            {
                type: "map-view",
                token,
                lat: center.lat,
                lng: center.lng,
                zoom: map.getZoom()
            },
            "*"
        );
    };

    map.on("moveend", publishView);
    publishView();
})();
