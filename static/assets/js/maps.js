let map;
let markers = [];
let patientMarker;
let infoWindow;

function initMap() {
    if (!window.clinicsData || window.clinicsData.length === 0) {
        console.log('No clinic data available for map');
        return;
    }

    // Initialize map centered on first clinic or patient location
    const firstClinic = window.clinicsData[0];
    const center = {
        lat: parseFloat(firstClinic.lat),
        lng: parseFloat(firstClinic.lng)
    };

    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 12,
        center: center,
        styles: [
            {
                featureType: 'poi.business',
                stylers: [{ visibility: 'off' }]
            }
        ],
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: false
    });

    infoWindow = new google.maps.InfoWindow();

    // Add patient location marker if available
    if (window.patientLocation) {
        geocodePatientLocation();
    }

    // Add clinic markers
    addClinicMarkers();

    // Setup interactions
    setupMapInteractions();
}

function geocodePatientLocation() {
    const geocoder = new google.maps.Geocoder();
    geocoder.geocode({ address: window.patientLocation }, (results, status) => {
        if (status === 'OK' && results[0]) {
            const location = results[0].geometry.location;
            patientMarker = new google.maps.Marker({
                position: location,
                map: map,
                title: `Patient Location: ${window.patientLocation}`,
                icon: {
                    url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <circle cx="12" cy="12" r="10" fill="#3B82F6" stroke="white" stroke-width="2"/>
                            <circle cx="12" cy="12" r="3" fill="white"/>
                        </svg>
                    `),
                    scaledSize: new google.maps.Size(32, 32),
                    anchor: new google.maps.Point(16, 16)
                }
            });

            patientMarker.addListener('click', () => {
                infoWindow.setContent(`
                    <div style="padding: 8px;">
                        <h4 style="margin: 0 0 4px 0; color: #1F2937;">Patient Location</h4>
                        <p style="margin: 0; color: #6B7280; font-size: 14px;">${window.patientLocation}</p>
                    </div>
                `);
                infoWindow.open(map, patientMarker);
            });
        }
    });
}

function addClinicMarkers() {
    window.clinicsData.forEach((clinic, index) => {
        const position = {
            lat: parseFloat(clinic.lat),
            lng: parseFloat(clinic.lng)
        };

        const marker = new google.maps.Marker({
            position: position,
            map: map,
            title: clinic.name,
            label: {
                text: (index + 1).toString(),
                color: 'white',
                fontWeight: 'bold',
                fontSize: '12px'
            },
            icon: {
                url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(`
                    <svg width="32" height="40" viewBox="0 0 24 30" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 0C7.58 0 4 3.58 4 8C4 14 12 24 12 24S20 14 20 8C20 3.58 16.42 0 12 0Z" fill="#EF4444"/>
                        <circle cx="12" cy="8" r="4" fill="white"/>
                    </svg>
                `),
                scaledSize: new google.maps.Size(32, 40),
                anchor: new google.maps.Point(16, 40),
                labelOrigin: new google.maps.Point(16, 8)
            }
        });

        markers.push(marker);

        // Marker click handler
        marker.addListener('click', () => {
            highlightClinic(index);
            infoWindow.setContent(`
                <div style="padding: 12px; max-width: 250px;">
                    <h4 style="margin: 0 0 8px 0; color: #1F2937;">${clinic.name}</h4>
                    <p style="margin: 0 0 8px 0; color: #6B7280; font-size: 14px;">${clinic.address}</p>
                    <div style="display: flex; gap: 16px; font-size: 12px;">
                        <div style="color: #6B7280;">
                            <strong>Distance:</strong> ${clinic.distance_text || 'N/A'}
                        </div>
                        ${clinic.duration_text ? `
                        <div style="color: #6B7280;">
                            <strong>Drive:</strong> ${clinic.duration_text}
                        </div>
                        ` : ''}
                    </div>
                </div>
            `);
            infoWindow.open(map, marker);
        });
    });

    // Fit map to show all markers
    if (markers.length > 0) {
        const bounds = new google.maps.LatLngBounds();
        markers.forEach(marker => bounds.extend(marker.getPosition()));
        if (patientMarker) bounds.extend(patientMarker.getPosition());
        map.fitBounds(bounds);
        
        // Ensure minimum zoom level
        const listener = google.maps.event.addListener(map, 'idle', () => {
            if (map.getZoom() > 15) map.setZoom(15);
            google.maps.event.removeListener(listener);
        });
    }
}

function setupMapInteractions() {
    // Clinic card click handlers
    document.querySelectorAll('.clinic-card').forEach((card, index) => {
        card.addEventListener('click', () => {
            highlightClinic(index);
            focusOnClinic(index);
        });
    });

    // Directions button handlers
    document.querySelectorAll('.directions-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const lat = btn.dataset.lat;
            const lng = btn.dataset.lng;
            const name = btn.dataset.name;
            openDirections(lat, lng, name);
        });
    });

    // Recenter button
    document.getElementById('recenter-btn')?.addEventListener('click', () => {
        if (markers.length > 0) {
            const bounds = new google.maps.LatLngBounds();
            markers.forEach(marker => bounds.extend(marker.getPosition()));
            if (patientMarker) bounds.extend(patientMarker.getPosition());
            map.fitBounds(bounds);
        }
    });

    // Filter functionality
    setupFilters();
}

function highlightClinic(index) {
    // Remove previous highlights
    document.querySelectorAll('.clinic-card').forEach(card => {
        card.classList.remove('highlighted');
    });

    // Highlight selected clinic
    const clinicCard = document.querySelector(`[data-clinic-id="${index}"]`);
    if (clinicCard) {
        clinicCard.classList.add('highlighted');
        clinicCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function focusOnClinic(index) {
    if (markers[index]) {
        map.setCenter(markers[index].getPosition());
        map.setZoom(15);
        google.maps.event.trigger(markers[index], 'click');
    }
}

function openDirections(lat, lng, name) {
    const destination = `${lat},${lng}`;
    const url = `https://www.google.com/maps/dir/?api=1&destination=${destination}&destination_place_id=${encodeURIComponent(name)}`;
    window.open(url, '_blank');
}

function setupFilters() {
    const searchInput = document.getElementById('search-filter');
    const distanceSlider = document.getElementById('distance-filter');
    const distanceValue = document.getElementById('distance-value');
    const clinicCount = document.getElementById('clinic-count');

    // Search filter
    searchInput?.addEventListener('input', () => {
        filterClinics();
    });

    // Distance filter
    distanceSlider?.addEventListener('input', (e) => {
        const value = e.target.value;
        distanceValue.textContent = `${value} km`;
        filterClinics();
    });
}

function filterClinics() {
    const searchTerm = document.getElementById('search-filter')?.value.toLowerCase() || '';
    const maxDistance = parseFloat(document.getElementById('distance-filter')?.value || 100);
    
    let visibleCount = 0;
    
    document.querySelectorAll('.clinic-card').forEach((card, index) => {
        const name = card.dataset.name || '';
        const address = card.dataset.address || '';
        const distance = parseFloat(card.dataset.distance || 0);
        
        const matchesSearch = name.includes(searchTerm) || address.includes(searchTerm);
        const withinDistance = distance <= maxDistance;
        const shouldShow = matchesSearch && withinDistance;
        
        if (shouldShow) {
            card.style.display = 'block';
            markers[index].setVisible(true);
            visibleCount++;
        } else {
            card.style.display = 'none';
            markers[index].setVisible(false);
        }
    });
    
    // Update count
    const clinicCount = document.getElementById('clinic-count');
    if (clinicCount) {
        clinicCount.textContent = visibleCount;
    }
    
    // Show/hide no results message
    const noResults = document.getElementById('no-results');
    const clinicsList = document.getElementById('clinics-list');
    if (visibleCount === 0) {
        noResults?.classList.remove('hidden');
        clinicsList?.classList.add('hidden');
    } else {
        noResults?.classList.add('hidden');
        clinicsList?.classList.remove('hidden');
    }
}

// Initialize map when page loads
document.addEventListener('DOMContentLoaded', () => {
    if (window.clinicsData && window.clinicsData.length > 0) {
        // Map will be initialized by Google Maps callback
        console.log('Waiting for Google Maps to load...');
    }
});
