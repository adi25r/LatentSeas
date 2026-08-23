// Verifies the movement math in app.js by loading the real file in a VM with stubs,
// so these assertions track the shipped code rather than a copy of it.
//   node src/frontend/navigation.test.js
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const assert = require('assert');

class Vector3 {
    constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
    set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
}

// Faithful enough to check vertex layout: THREE lays out row-major with
// vertex y = half - row*step, which becomes world z = -half + row*step after rotation.
class PlaneGeometry {
    constructor(w, h, segW, segH) {
        const gx = segW + 1, gy = segH + 1;
        const stepX = w / segW, stepY = h / segH;
        this.gridX = gx;
        const arr = new Float32Array(gx * gy * 3);
        for (let iy = 0; iy < gy; iy++) {
            for (let ix = 0; ix < gx; ix++) {
                const i = (iy * gx + ix) * 3;
                arr[i]     = ix * stepX - w / 2;
                arr[i + 1] = -(iy * stepY - h / 2);
                arr[i + 2] = 0;
            }
        }
        this.attributes = { position: { array: arr, needsUpdate: false } };
    }
    computeVertexNormals() {}
}
class InstancedMesh {
    constructor(g, m, count) { this.count = count; this.instanceMatrix = { setUsage() {} };
                               this.instanceColor = { needsUpdate: false }; }
    setMatrixAt() {} setColorAt() {}
}
class Mesh {
    constructor(geometry, material) {
        this.geometry = geometry; this.material = material;
        this.rotation = { x: 0, y: 0, z: 0 };
    }
}
class Matrix4 { makeScale() { return this; } setPosition() { return this; } }
class Color { setHex(h) { this.hex = h; return this; } }
const noop = () => {};
const stubEl = { addEventListener: noop, style: {}, clientWidth: 800, clientHeight: 600,
                 getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
                 appendChild: noop, remove: noop, innerHTML: '', textContent: '' };

const ctx = {
    THREE: new Proxy({ Vector3, PlaneGeometry, InstancedMesh, Mesh, Matrix4, Color }, {
        get: (t, k) => k in t ? t[k] : function () { return new Proxy({}, { get: () => noop }); }
    }),
    window: { addEventListener: noop },
    document: { addEventListener: noop, getElementById: () => stubEl, createElement: () => stubEl,
                body: stubEl, activeElement: null },
    console, fetch: noop, requestAnimationFrame: noop, setTimeout: noop, clearTimeout: noop,
    performance: { now: () => 0 }, Math, Set, Map, Float32Array, JSON, Number, String, Object,
};
ctx.globalThis = ctx;
vm.createContext(ctx);

// Top-level `let` in a script lives in lexical scope, not on the global object, so the
// test cannot poke at it from outside. Appending an accessor block to the same script
// puts it in that same scope, which keeps the test bound to the real variables.
const source = fs.readFileSync(path.join(__dirname, 'app.js'), 'utf8') + `
;globalThis.__t = {
    rightVector, updateMovement,
    get rot()  { return playerRotation; },  set rot(v)  { playerRotation = v; },
    get pitch(){ return cameraPitch; },     set pitch(v){ cameraPitch = v; },
    get pos()  { return playerPosition; },  set pos(v)  { playerPosition = v; },
    set keys(v){ keys = v; },
    set fly(v) { flyMode = v; },
    set world(v){ worldSize = v; },
    set hm(v)  { heightmap = v; },
    set scene(v){ scene = v; },
    terrainVerts() { return terrainMesh.geometry.attributes.position.array; },
};`;
vm.runInContext(source, ctx);
const t = ctx.__t;

const close = (a, b, m) => assert.ok(Math.abs(a - b) < 1e-9, `${m}: got ${a}, want ${b}`);

// Facing +Z with +Y up, the player's right hand points to -X. right = cross(forward, up).
t.rot = 0;
let r = t.rightVector();
close(r.x, -1, 'right.x facing +Z'); close(r.z, 0, 'right.z facing +Z');
console.log('  OK  facing +Z -> right is -X (D moves -x, A moves +x)');

// Quarter turn left: forward becomes +X, so right becomes +Z.
t.rot = Math.PI / 2;
r = t.rightVector();
close(r.x, 0, 'right.x facing +X'); close(r.z, 1, 'right.z facing +X');
console.log('  OK  facing +X -> right is +Z');

// right must always be perpendicular to forward, at every heading.
for (const theta of [0, 0.7, 1.9, 3.4, 5.2]) {
    t.rot = theta;
    const f = { x: Math.sin(theta), z: Math.cos(theta) };
    const rv = t.rightVector();
    close(f.x * rv.x + f.z * rv.z, 0, `perpendicular at ${theta}`);
    // and it must be right-handed, not flipped: cross(f, up).x == -cos, .z == sin
    close(rv.x, -Math.cos(theta), `right.x at ${theta}`);
    close(rv.z, Math.sin(theta), `right.z at ${theta}`);
}
console.log('  OK  right stays perpendicular and correctly handed at all headings');

// Arrow keys: left turns left (rotation up), up pitches up (pitch up).
function press(keys) {
    t.keys = keys; t.fly = false;
    t.rot = 0; t.pitch = 0;
    t.pos = { x: 0, y: 2, z: 0 };
    t.world = 120; t.hm = null;
    t.updateMovement();
    return { rot: t.rot, pitch: t.pitch, pos: t.pos };
}
assert.ok(press({ arrowleft: true }).rot > 0, 'arrowleft should increase rotation (turn left)');
assert.ok(press({ arrowright: true }).rot < 0, 'arrowright should decrease rotation');
assert.ok(press({ arrowup: true }).pitch > 0, 'arrowup should pitch UP (positive)');
assert.ok(press({ arrowdown: true }).pitch < 0, 'arrowdown should pitch DOWN (negative)');
console.log('  OK  arrows: left/right turn correctly, up pitches up, down pitches down');

// Arrow pitch must work in walk mode too, not only while flying.
assert.ok(press({ arrowup: true }).pitch > 0, 'arrowup must work in walk mode');
console.log('  OK  arrow pitch works in walk mode');

// W/A/S/D directions while facing +Z.
close(press({ w: true }).pos.z > 0 ? 1 : 0, 1, 'W moves +z when facing +Z');
close(press({ s: true }).pos.z < 0 ? 1 : 0, 1, 'S moves -z when facing +Z');
assert.ok(press({ d: true }).pos.x < 0, 'D must move -x when facing +Z (was inverted)');
assert.ok(press({ a: true }).pos.x > 0, 'A must move +x when facing +Z (was inverted)');
console.log('  OK  W/S forward-back, A/D strafe the correct way round');

// Shift sprints rather than sinking you into the ground.
assert.ok(press({ w: true, shift: true }).pos.z > press({ w: true }).pos.z,
          'shift should sprint');
console.log('  OK  shift sprints');

// getTerrainHeight must agree with terrain.py's sample_surface. If the two drift apart
// the camera walks through the ground, so the fixture pins them together.
{
    const fx = JSON.parse(fs.readFileSync(path.join(__dirname, 'terrain.fixture.json'), 'utf8'));
    t.hm = fx.heightmap;
    ctx.__t.world = fx.world_size;
    vm.runInContext(`gridSize = ${fx.grid_size}; worldSize = ${fx.world_size};`, ctx);

    let worst = 0;
    for (let i = 0; i < fx.points.length; i++) {
        const got = ctx.getTerrainHeight(fx.points[i][0], fx.points[i][1]);
        worst = Math.max(worst, Math.abs(got - fx.expected[i]));
    }
    assert.ok(worst < 1e-9, `JS/Python terrain sampling disagree by ${worst}`);
    console.log(`  OK  getTerrainHeight matches terrain.py on ${fx.points.length} points (max diff ${worst.toExponential(1)})`);
}

// renderTerrain must not flip the heightmap. It previously indexed rows as
// [gridSize-1-row], mirroring the terrain against the features so most of them hung in
// the air. getTerrainHeight is already checked against Python, so requiring the mesh to
// agree with it catches any row-mapping mistake.
{
    const fx = JSON.parse(fs.readFileSync(path.join(__dirname, 'terrain.fixture.json'), 'utf8'));
    t.scene = { add() {}, fog: null };
    ctx.renderTerrain({
        heightmap: fx.heightmap, grid_size: fx.grid_size, world_size: fx.world_size, points: []
    });

    const G = fx.grid_size;
    const verts = ctx.__t.terrainVerts();
    let worstVertex = 0;
    for (let r = 0; r < G; r++) {
        for (let c = 0; c < G; c++) {
            worstVertex = Math.max(worstVertex,
                Math.abs(verts[(r * G + c) * 3 + 2] - fx.heightmap[r][c]));
        }
    }
    assert.ok(worstVertex < 1e-6, `mesh rows are flipped or misaligned (off by ${worstVertex})`);

    let worstPoint = 0;
    for (let i = 0; i < fx.points.length; i++) {
        worstPoint = Math.max(worstPoint,
            Math.abs(ctx.getTerrainHeight(fx.points[i][0], fx.points[i][1]) - fx.expected[i]));
    }
    assert.ok(worstPoint < 1e-9, `mesh disagrees with sampled height by ${worstPoint}`);
    console.log('  OK  renderTerrain rows are unflipped and agree with the sampled surface');
}

console.log('\nAll navigation tests passed.');
