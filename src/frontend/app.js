const API_BASE = 'http://localhost:8000';

let scene, camera, renderer;
let terrainMesh, featureMesh = null, featurePositions = [];
let hoveredFeature = -1;
let knownLabels = {};        // feature_idx -> description, earned by digging
let diggable = [];           // features worth digging at all

// Digging
const DIG_RADIUS = 3.0;      // how close you must stand
const DIG_SECONDS = 1.2;
let digTarget = -1;
let digProgress = 0;
let waypoint = -1;           // the marked node you are currently heading for
let probeStrengths = {};     // feature_idx -> strength it fired at in the last probe
let lastFrame = performance.now();
const BASE_RADIUS = 0.16;
const ACTIVE_RADIUS = 0.38;
const FLAG_RADIUS = 0.55;
const COLOR_IDLE = 0x6b6b73;
const COLOR_ACTIVE = 0xff4444;
const COLOR_FLAG = 0x00ff88;
const COLOR_HOVER = 0xffffff;
const COLOR_KNOWN = 0x7f9fd8;
const COLOR_WAYPOINT = 0xffcc44;
const COLOR_DIG = 0xffdd33;
let digCandidate = -1;   // whichever node you're currently close enough to dig
let pointmapData = null;
let activatedFeatures = [];
let placedFlags = new Map();

// Navigation state
let playerPosition = { x: 0, y: 2, z: 0 };
let playerRotation = 0;
let cameraPitch = 0;
let keys = {};
let flyMode = false;
const MOVE_SPEED = 0.35;
const FLY_SPEED = 0.8;
const LOOK_SPEED = 0.002;
const ROTATE_SPEED = 0.03;
const CAMERA_HEIGHT = 2;

// Terrain, supplied by the backend
let heightmap = null;
let gridSize = 0;
let worldSize = 120;

// Mouse control
let mouseDown = false;
let lastMouseX = 0;

// Initialize Three.js scene
function initScene() {
    const canvas = document.getElementById('terrain-canvas');
    const container = document.getElementById('canvas-container');

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);

    camera = new THREE.PerspectiveCamera(
        75,
        container.clientWidth / container.clientHeight,
        0.1,
        400
    );
    camera.position.set(0, CAMERA_HEIGHT, 0);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 10);
    scene.add(directionalLight);

    // Handle window resize
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    // Keyboard controls. Typing in the side panel must not drive the camera, which is
    // also why W/A/S/D and the arrows appeared unreliable while a field had focus.
    const isTyping = () => {
        const el = document.activeElement;
        return el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA');
    };
    const NAV_KEYS = new Set(['w', 'a', 's', 'd', 'q', 'e', 'c', 'f', ' ', 'shift',
                              'arrowup', 'arrowdown', 'arrowleft', 'arrowright']);

    window.addEventListener('keydown', (e) => {
        const key = e.key.toLowerCase();
        if (isTyping() || !NAV_KEYS.has(key)) return;
        e.preventDefault();   // stop arrows scrolling the page and space paging down
        if (keys[key]) return;  // ignore auto-repeat so 'f' does not toggle continuously
        keys[key] = true;

        // Toggle fly mode with 'F' key
        if (key === 'f') {
            flyMode = !flyMode;
            const mode = flyMode ? 'FLY' : 'WALK';
            console.log(`Mode: ${mode}`);

            // Show notification
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: ${flyMode ? '#00ff88' : '#ff4444'};
                color: #000;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                z-index: 1000;
            `;
            notification.textContent = `${mode} MODE`;
            document.body.appendChild(notification);
            setTimeout(() => notification.remove(), 1500);
        }
    });
    window.addEventListener('keyup', (e) => {
        keys[e.key.toLowerCase()] = false;
    });

    // Releasing focus from a field hands control back to the world
    window.addEventListener('blur', () => { keys = {}; });

    // Mouse controls for looking
    canvas.addEventListener('mousedown', (e) => {
        if (e.button === 2) { // Right click
            mouseDown = true;
            lastMouseX = e.clientX;
        }
    });

    canvas.addEventListener('mouseup', (e) => {
        if (e.button === 2) {
            mouseDown = false;
        }
    });

    let lastMouseY = 0;
    canvas.addEventListener('mousemove', (e) => {
        if (mouseDown) {
            const deltaX = e.clientX - lastMouseX;
            const deltaY = e.clientY - lastMouseY;

            playerRotation -= deltaX * LOOK_SPEED;

            cameraPitch = clamp(cameraPitch - deltaY * LOOK_SPEED,
                                -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);

            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }
    });

    canvas.addEventListener('mousedown', (e) => {
        if (e.button === 2) {
            mouseDown = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }
    });

    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    // Hover highlight, so you can confirm the target before committing to a click
    canvas.addEventListener('mousemove', (e) => {
        if (mouseDown) return;
        const idx = pickFeature(e);
        if (idx !== hoveredFeature) {
            hoveredFeature = idx;
            updateFeatureColors();
            canvas.style.cursor = idx >= 0 ? 'pointer' : 'default';
        }
    });

    animate();
}

function animate() {
    requestAnimationFrame(animate);

    const now = performance.now();
    const dt = Math.min((now - lastFrame) / 1000, 0.1);
    lastFrame = now;

    // Update player movement
    updateMovement();
    updateDigging(dt);

    // Update camera position
    updateCamera();

    renderer.render(scene, camera);
    updateHud();
}

// Nearest diggable feature within reach, or -1. Undug nodes are just mounds in the
// ground; you have to stand on one to find out what it is.
function nearestDiggable() {
    if (!featurePositions.length || flyMode) return -1;

    let best = -1, bestDist = DIG_RADIUS;
    for (let i = 0; i < featurePositions.length; i++) {
        const p = featurePositions[i];
        const dx = p[0] - playerPosition.x, dz = p[2] - playerPosition.z;
        const dist = Math.hypot(dx, dz);
        if (dist < bestDist && diggable[i] !== false) { bestDist = dist; best = i; }
    }
    return best;
}

function updateDigging(dt) {
    const near = nearestDiggable();

    // Surface which node E would actually dig, so a crowded hill isn't a guessing game.
    if (near !== digCandidate) {
        digCandidate = near;
        updateFeatureColors();
    }

    // Moving off the node, or letting go, abandons the hole
    if (!keys['e'] || near !== digTarget) {
        digTarget = keys['e'] ? near : -1;
        digProgress = keys['e'] && near >= 0 ? digProgress : 0;
        if (!keys['e']) digProgress = 0;
    }

    if (keys['e'] && near >= 0) {
        digTarget = near;
        digProgress += dt / DIG_SECONDS;
        if (digProgress >= 1) {
            digProgress = 0;
            keys['e'] = false;          // require a fresh press for the next one
            revealFeature(near);
        }
    }

    renderDigUi(near);
}

async function revealFeature(featureIdx) {
    try {
        const res = await fetch(`${API_BASE}/dig`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ feature_idx: featureIdx })
        });
        const data = await res.json();
        if (data.error) return showToast(data.error, false);

        knownLabels[featureIdx] = data.label;
        if (waypoint === featureIdx) waypoint = -1;
        showToast(`${data.label} — click it to activate`, true);
        updateFeatureColors();
        refreshProbeLabels();
    } catch (err) {
        console.error('dig failed', err);
    }
}

function renderDigUi(near) {
    const el = document.getElementById('dig-prompt');
    if (!el) return;
    if (digProgress > 0 && digTarget >= 0) {
        el.className = 'digging';
        el.innerHTML = `<div class="dig-bar"><div class="dig-fill" style="width:${digProgress * 100}%"></div></div>
                        <div class="dig-text">digging…</div>`;
    } else if (near >= 0 && !knownLabels[near]) {
        el.className = 'ready';
        el.innerHTML = '<div class="dig-text">hold <kbd>E</kbd> to dig</div>';
    } else if (near >= 0) {
        el.className = 'ready';
        el.innerHTML = `<div class="dig-text known">${escapeHtml(knownLabels[near])}</div>`;
    } else {
        el.className = '';
        el.innerHTML = '';
    }
}

function showToast(text, good) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = good ? `uncovered: ${text}` : text;
    el.className = good ? 'show good' : 'show bad';
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => { el.className = ''; }, 3200);
}

let hudEl = null;
function updateHud() {
    if (!hudEl) hudEl = document.getElementById('hud');
    if (!hudEl) return;
    const heading = ((playerRotation * 180 / Math.PI) % 360 + 360) % 360;
    const pitch = cameraPitch * 180 / Math.PI;
    hudEl.innerHTML =
        `<span class="${flyMode ? 'hud-fly' : 'hud-walk'}">${flyMode ? 'FLY' : 'WALK'}</span>` +
        ` &nbsp; x ${playerPosition.x.toFixed(1)}  z ${playerPosition.z.toFixed(1)}` +
        ` &nbsp; heading ${heading.toFixed(0)}°  pitch ${pitch.toFixed(0)}°` +
        (placedFlags.size ? ` &nbsp; flags ${placedFlags.size}` : '') +
        (hoveredFeature >= 0
            ? ` &nbsp; <span class="hud-hover">${escapeHtml(knownLabels[hoveredFeature] || 'unmarked mound')}</span>`
            : '') +
        waypointReadout();
}

// Range and a relative arrow to the active waypoint.
function waypointReadout() {
    if (waypoint < 0 || !featurePositions[waypoint]) return '';
    const p = featurePositions[waypoint];
    const dx = p[0] - playerPosition.x, dz = p[2] - playerPosition.z;
    const dist = Math.hypot(dx, dz);

    // bearing relative to where the player is facing
    const rel = Math.atan2(dx, dz) - playerRotation;
    const arrows = ['\u2191', '\u2197', '\u2192', '\u2198', '\u2193', '\u2199', '\u2190', '\u2196'];
    const arrow = arrows[Math.round(((rel % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) / (Math.PI / 4)) % 8];

    const close = dist <= DIG_RADIUS;
    return ` &nbsp; <span class="hud-waypoint${close ? ' arrived' : ''}">` +
           `${arrow} site ${dist.toFixed(0)}m${close ? ' — dig here' : ''}</span>`;
}

// For a camera facing `forward` with +Y up, right is cross(forward, up).
// Getting this backwards is what made A and D swap.
function rightVector() {
    return new THREE.Vector3(-Math.cos(playerRotation), 0, Math.sin(playerRotation));
}

function updateMovement() {
    const speed = (flyMode ? FLY_SPEED : MOVE_SPEED) * (keys['shift'] ? 3 : 1);

    // Arrow keys look around. cameraPitch is positive-is-up, and applies in walk mode too.
    if (keys['arrowleft'])  playerRotation += ROTATE_SPEED;
    if (keys['arrowright']) playerRotation -= ROTATE_SPEED;
    if (keys['arrowup'])    cameraPitch = Math.min(Math.PI / 2 - 0.01, cameraPitch + ROTATE_SPEED);
    if (keys['arrowdown'])  cameraPitch = Math.max(-Math.PI / 2 + 0.01, cameraPitch - ROTATE_SPEED);

    if (flyMode) {
        // Fly mode - 3D movement
        const forward = new THREE.Vector3(
            Math.sin(playerRotation) * Math.cos(cameraPitch),
            Math.sin(cameraPitch),
            Math.cos(playerRotation) * Math.cos(cameraPitch)
        );
        const right = rightVector();

        if (keys['w']) {
            playerPosition.x += forward.x * speed;
            playerPosition.y += forward.y * speed;
            playerPosition.z += forward.z * speed;
        }
        if (keys['s']) {
            playerPosition.x -= forward.x * speed;
            playerPosition.y -= forward.y * speed;
            playerPosition.z -= forward.z * speed;
        }
        if (keys['a']) {
            playerPosition.x -= right.x * speed;
            playerPosition.z -= right.z * speed;
        }
        if (keys['d']) {
            playerPosition.x += right.x * speed;
            playerPosition.z += right.z * speed;
        }
        if (keys['q'] || keys[' ']) playerPosition.y += speed;
        if (keys['c'])              playerPosition.y -= speed;

        const lim = worldSize / 2;
        playerPosition.x = clamp(playerPosition.x, -lim, lim);
        playerPosition.y = clamp(playerPosition.y, 0.5, 120);
        playerPosition.z = clamp(playerPosition.z, -lim, lim);

    } else {
        // Walk mode - 2D movement on terrain
        const forward = new THREE.Vector3(
            Math.sin(playerRotation),
            0,
            Math.cos(playerRotation)
        );
        const right = rightVector();

        if (keys['w']) {
            playerPosition.x += forward.x * speed;
            playerPosition.z += forward.z * speed;
        }
        if (keys['s']) {
            playerPosition.x -= forward.x * speed;
            playerPosition.z -= forward.z * speed;
        }
        if (keys['a']) {
            playerPosition.x -= right.x * speed;
            playerPosition.z -= right.z * speed;
        }
        if (keys['d']) {
            playerPosition.x += right.x * speed;
            playerPosition.z += right.z * speed;
        }

        const lim = worldSize / 2;
        playerPosition.x = clamp(playerPosition.x, -lim, lim);
        playerPosition.z = clamp(playerPosition.z, -lim, lim);
    }
}

function updateCamera() {
    // In walk mode the camera rides the terrain; in fly mode it is wherever you put it.
    const eyeY = flyMode
        ? playerPosition.y
        : getTerrainHeight(playerPosition.x, playerPosition.z) + CAMERA_HEIGHT;

    camera.position.set(playerPosition.x, eyeY, playerPosition.z);

    // cameraPitch is positive-is-up, so it adds to the look target's height.
    const cp = Math.cos(cameraPitch);
    camera.lookAt(new THREE.Vector3(
        playerPosition.x + Math.sin(playerRotation) * cp,
        eyeY + Math.sin(cameraPitch),
        playerPosition.z + Math.cos(playerRotation) * cp
    ));
}

// Height of the rendered surface. Mirrors sample_surface in terrain.py: it interpolates
// across the same triangles PlaneGeometry builds, so the camera walks exactly on the
// ground it can see rather than on a curved bilinear approximation of it.
function getTerrainHeight(x, z) {
    if (!heightmap) return 0;

    const half = worldSize / 2;
    const step = worldSize / (gridSize - 1);
    const gx = clamp((x + half) / step, 0, gridSize - 1.0001);
    const gz = clamp((z + half) / step, 0, gridSize - 1.0001);

    const c0 = Math.floor(gx), r0 = Math.floor(gz);
    const fx = gx - c0, fz = gz - r0;

    const hA = heightmap[r0][c0];
    const hB = heightmap[r0 + 1][c0];
    const hC = heightmap[r0 + 1][c0 + 1];
    const hD = heightmap[r0][c0 + 1];

    return (fx + fz <= 1)
        ? hA + fx * (hD - hA) + fz * (hB - hA)
        : hC + (1 - fx) * (hB - hC) + (1 - fz) * (hD - hC);
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// Load pointmap from API
async function loadPointmap() {
    try {
        const response = await fetch(`${API_BASE}/pointmap`);
        const data = await response.json();

        pointmapData = data;
        renderTerrain(data);
        document.getElementById('loading').style.display = 'none';
    } catch (error) {
        console.error('Error loading pointmap:', error);
        document.getElementById('loading').textContent = 'Error loading terrain. Is the API running?';
    }
}

// Build the terrain from the backend's KDE heightmap. The grid is already computed,
// so this is O(grid) rather than the O(vertices * features) nearest-neighbour scan it
// replaced, which would have been 61M distance checks for 24k features.
function renderTerrain(data) {
    knownLabels = data.known || {};
    diggable = data.diggable || [];
    heightmap = data.heightmap;
    gridSize = data.grid_size;
    worldSize = data.world_size;

    const geometry = new THREE.PlaneGeometry(worldSize, worldSize, gridSize - 1, gridSize - 1);
    const verts = geometry.attributes.position.array;
    for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
            // After rotation.x = -PI/2, geometry row r sits at world z = -half + r*step,
            // and the heightmap is indexed [z][x] over the same range. Rows map straight
            // across - flipping them here mirrored the terrain against the features and
            // left most of them hanging in the air.
            verts[(row * gridSize + col) * 3 + 2] = heightmap[row][col];
        }
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();

    terrainMesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
        color: 0x2d5a3d, flatShading: false, side: THREE.DoubleSide
    }));
    terrainMesh.rotation.x = -Math.PI / 2;
    scene.add(terrainMesh);

    scene.fog = new THREE.Fog(0x0a0a0a, worldSize * 0.15, worldSize * 0.75);
    renderFeatureMarkers(data.points);
    placePlayerOnTerrain();
}

// Real 3D geometry, not camera-facing sprites: one InstancedMesh means 24k lit,
// shaded, depth-sorted spheres still cost a single draw call. Icosahedra are used
// because 20 triangles each keeps the whole field under 500k triangles.
function renderFeatureMarkers(points) {
    featurePositions = points;

    const geometry = new THREE.IcosahedronGeometry(1, 0);
    const material = new THREE.MeshStandardMaterial({
        roughness: 0.55, metalness: 0.1, flatShading: true
    });

    featureMesh = new THREE.InstancedMesh(geometry, material, points.length);
    featureMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    for (let i = 0; i < points.length; i++) {
        setInstance(i, BASE_RADIUS, COLOR_IDLE);
    }
    featureMesh.instanceMatrix.needsUpdate = true;
    if (featureMesh.instanceColor) featureMesh.instanceColor.needsUpdate = true;

    scene.add(featureMesh);
}

const _m = new THREE.Matrix4();
const _c = new THREE.Color();

// Scale lives in the instance matrix, so a highlighted feature grows in place.
function setInstance(i, radius, hex) {
    const p = featurePositions[i];
    _m.makeScale(radius, radius, radius);
    _m.setPosition(p[0], p[1] + radius, p[2]);   // rest on the surface, not sunk into it
    featureMesh.setMatrixAt(i, _m);
    featureMesh.setColorAt(i, _c.setHex(hex));
}

// Drop the player onto the surface at the start so they never spawn inside a hill.
function placePlayerOnTerrain() {
    playerPosition.y = getTerrainHeight(playerPosition.x, playerPosition.z) + CAMERA_HEIGHT;
}

async function probeSentence() {
    const sentence = document.getElementById('probe-input').value.trim();
    if (!sentence) return;
    if (/\s/.test(sentence)) {
        showToast('probe a single word only', false);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/probe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sentence, threshold: 10 })
        });

        const data = await response.json();
        if (data.error) return showToast(data.error, false);
        activatedFeatures = data.activated_features.map(f => f.feature_idx);
        updateFeatureColors();
        renderProbeResults(data.activated_features);
    } catch (error) {
        console.error('Error probing:', error);
    }
}

// Rewrites instance matrices and colours in place, so highlighting stays one draw call.
// The probe list is the main way to play: each row is a named concept, and clicking it
// flags that feature directly rather than hunting for its dot on the map.
// The probe marks where to dig, not what is buried. Clicking a row sets it as your
// waypoint; the HUD then gives range and bearing so the walk is navigable rather than a
// hunt across 24k identical mounds.
function renderProbeResults(features) {
    const el = document.getElementById('probe-result');
    if (!features.length) {
        el.innerHTML = '<em style="color:#888">nothing activated above threshold</em>';
        return;
    }
    probeStrengths = {};
    features.forEach(f => { probeStrengths[f.feature_idx] = f.suggested_strength; });

    el.innerHTML = features.slice(0, 12).map((f, n) => {
        const known = knownLabels[f.feature_idx];
        return `
        <div class="probe-row${placedFlags.has(f.feature_idx) ? ' flagged' : ''}${known ? '' : ' unknown'}"
             data-feature="${f.feature_idx}">
            <span class="probe-num">${n + 1}.</span>
            <span class="probe-act">${f.activation.toFixed(0)}</span>
            <span class="probe-label">${known ? escapeHtml(known) : `unmarked site ${n + 1}`}</span>
        </div>`;
    }).join('');

    el.querySelectorAll('.probe-row').forEach(row => {
        row.addEventListener('click', () => {
            const idx = Number(row.dataset.feature);
            if (knownLabels[idx]) {
                if (placedFlags.has(idx)) removeFlag(idx);
                else placeFlag(idx, probeStrengths[idx]);
            } else {
                waypoint = idx;         // undug: guide the player there instead
                syncProbeRows();
            }
        });
    });
}

function refreshProbeLabels() {
    document.querySelectorAll('.probe-row').forEach(row => {
        const idx = Number(row.dataset.feature);
        if (knownLabels[idx]) {
            row.classList.remove('unknown');
            row.querySelector('.probe-label').textContent = knownLabels[idx];
        }
    });
}

function syncProbeRows() {
    document.querySelectorAll('.probe-row').forEach(row => {
        const idx = Number(row.dataset.feature);
        row.classList.toggle('flagged', placedFlags.has(idx));
        row.classList.toggle('waypoint', idx === waypoint);
    });
}

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, ch => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

function updateFeatureColors() {
    if (!featureMesh) return;

    const activated = new Set(activatedFeatures);
    for (let i = 0; i < featurePositions.length; i++) {
        let radius = BASE_RADIUS, color = COLOR_IDLE;
        if (placedFlags.has(i))      { radius = FLAG_RADIUS;   color = COLOR_FLAG; }
        else if (i === waypoint)     { radius = FLAG_RADIUS;   color = COLOR_WAYPOINT; }
        else if (activated.has(i))   { radius = ACTIVE_RADIUS; color = COLOR_ACTIVE; }
        else if (knownLabels[i])     { radius = ACTIVE_RADIUS * 0.7; color = COLOR_KNOWN; }
        if (i === hoveredFeature)    { radius = Math.max(radius, ACTIVE_RADIUS) * 1.35; color = COLOR_HOVER; }
        if (i === digCandidate)      { radius = Math.max(radius, ACTIVE_RADIUS) * 1.35; color = COLOR_DIG; }
        setInstance(i, radius, color);
    }
    featureMesh.instanceMatrix.needsUpdate = true;
    if (featureMesh.instanceColor) featureMesh.instanceColor.needsUpdate = true;
}

async function placeFlag(featureIdx, suggested) {
    // A probed feature defaults to the strength it naturally fired at; the slider scales it.
    const slider = parseFloat(document.getElementById('boost-multiplier').value);
    const strength = suggested === undefined ? slider : suggested;

    try {
        const response = await fetch(`${API_BASE}/flag`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ feature_idx: featureIdx, strength: strength })
        });

        const data = await response.json();
        if (data.success) {
            placedFlags.set(featureIdx, strength);
            updateFeatureColors();
            updateFlagsList();
            syncProbeRows();
        }
    } catch (error) {
        console.error('Error placing flag:', error);
    }
}

async function removeFlag(featureIdx) {
    try {
        const response = await fetch(`${API_BASE}/flag/${featureIdx}`, { method: 'DELETE' });
        const data = await response.json();
        if (data.success) {
            placedFlags.delete(featureIdx);
            updateFeatureColors();
            updateFlagsList();
            syncProbeRows();
        }
    } catch (error) {
        console.error('Error removing flag:', error);
    }
}

function updateFlagsList() {
    const container = document.getElementById('flags-list');
    if (placedFlags.size === 0) {
        container.innerHTML = '<div style="color: #666; padding: 0.5rem;">No flags placed</div>';
        return;
    }

    container.innerHTML = '';
    placedFlags.forEach((strength, featureIdx) => {
        const item = document.createElement('div');
        item.className = 'flag-item';
        const label = featureLabels[featureIdx] || `feature ${featureIdx}`;
        item.innerHTML = `
            <span class="flag-info" title="feature ${featureIdx}">
                ${escapeHtml(label)} <span class="flag-strength">@${Math.round(strength)}</span>
            </span>
            <span class="flag-remove" data-feature="${featureIdx}">✕</span>
        `;
        item.querySelector('.flag-remove').addEventListener('click',
            () => removeFlag(featureIdx));
        container.appendChild(item);
    });
}

async function generateText() {
    const prompt = document.getElementById('generate-prompt').value;
    const temperature = parseFloat(document.getElementById('temperature').value);
    const target = document.getElementById('target-input').value.trim();

    if (!prompt) return;

    if (placedFlags.size === 0) {
        document.getElementById('generate-result').innerHTML = '<em style="color: #ff4444;">Place some flags first!</em>';
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, max_tokens: 50, temperature: temperature, target: target || null })
        });

        const data = await response.json();
        document.getElementById('generate-result').innerHTML = `
            <strong>Generated:</strong><br>
            ${data.generated}
        `;
        renderScore(data.score);
    } catch (error) {
        console.error('Error generating:', error);
    }
}

function renderScore(score) {
    const el = document.getElementById('score-display');
    if (score === undefined || score === null) {
        el.innerHTML = '';
        return;
    }
    // Unrelated text lands around 25, paraphrases around 45-70.
    const color = score >= 55 ? '#00ff88' : score >= 35 ? '#ffcc44' : '#ff4444';
    el.innerHTML = `
        <div class="score-value" style="color:${color}">${score.toFixed(1)}</div>
        <div class="score-bar"><div class="score-fill" style="width:${Math.min(100, score)}%;background:${color}"></div></div>
        <small>semantic similarity to target</small>
    `;
}

async function clearAllFlags() {
    try {
        await fetch(`${API_BASE}/flags/clear`, { method: 'DELETE' });
        placedFlags.clear();
        updateFeatureColors();
        updateFlagsList();
        syncProbeRows();
    } catch (error) {
        console.error('Error clearing flags:', error);
    }
}

function onCanvasClick(event) {
    const featureIdx = pickFeature(event);
    if (featureIdx < 0) return;

    if (!knownLabels[featureIdx]) {
        waypoint = featureIdx;      // unknown: mark it and walk over
        syncProbeRows();
        showToast('marked — walk there and hold E to dig', true);
        return;
    }
    if (placedFlags.has(featureIdx)) removeFlag(featureIdx);
    else placeFlag(featureIdx, probeStrengths[featureIdx]);
}

// Returns the feature under the cursor, or -1. Probed and flagged features win ties,
// since those are the ones you are actually trying to hit in a crowded patch.
function pickFeature(event) {
    if (!featureMesh) return -1;

    const canvas = document.getElementById('terrain-canvas');
    const rect = canvas.getBoundingClientRect();
    const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
    );

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);

    const hits = raycaster.intersectObject(featureMesh);
    if (hits.length === 0) return -1;

    const wanted = hits.find(h => placedFlags.has(h.instanceId)
                               || activatedFeatures.includes(h.instanceId));
    return (wanted || hits[0]).instanceId;
}

document.addEventListener('DOMContentLoaded', () => {
    initScene();
    loadPointmap();

    document.getElementById('probe-btn').addEventListener('click', probeSentence);
    document.getElementById('generate-btn').addEventListener('click', generateText);
    document.getElementById('clear-flags-btn').addEventListener('click', clearAllFlags);
    document.getElementById('terrain-canvas').addEventListener('click', onCanvasClick);

    document.getElementById('probe-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') probeSentence();
    });

    // Update slider value displays
    document.getElementById('boost-multiplier').addEventListener('input', (e) => {
        document.getElementById('multiplier-value').textContent = e.target.value;
    });

    document.getElementById('temperature').addEventListener('input', (e) => {
        document.getElementById('temp-value').textContent = e.target.value;
    });

    updateFlagsList();
});
