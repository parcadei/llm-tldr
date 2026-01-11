/**
 * TLDR Brain v2 - Enhanced Visualization
 * Three view modes + cluster spheres + similarity heatmap + edge highlighting
 */

const Graph = ForceGraph3D()
    (document.getElementById('3d-graph'));

let graphData = { nodes: [], links: [], clusters: [], similarity_pairs: [] };
let currentView = 'force';
let hybridBlend = 0.5;
let isDarkTheme = true;
let showClusterSpheres = false;
let clusterSphereMeshes = [];
let selectedNode = null;
let comparisonPanel = null;

// Centrality weight sliders
let weights = { pagerank: 0.6, influence: 0.2, support: 0.2 };

const LAYER_COLORS = {
    'ENTRY': '#00ff88',
    'MIDDLE': '#00aaff',
    'LEAF': '#ff8800',
    'FILE': '#ffffff',
    'UNKNOWN': '#888888'
};

const CLUSTER_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
];

// Info descriptions
const INFO_TEXTS = {
    'toggle-theme': 'Switch between dark and light color themes',
    'zoom-fit': 'Reset camera to show all nodes in view',
    'cluster-spheres': 'Show translucent spheres around semantic clusters (works in Semantic/Hybrid views)',
    'similarity-map': 'Show a panel of most similar code pairs based on semantic embeddings',
    'blend-control': 'Adjust balance between structural (force) and semantic (embedding) positioning',
    'centrality-filter': 'Hide nodes below this PageRank centrality threshold',
    'weight-pagerank': 'Weight of global PageRank importance in node sizing',
    'weight-influence': 'Weight of incoming neighbor importance (nodes that call this function)',
    'weight-support': 'Weight of outgoing neighbor importance (nodes this function calls)',
    'legend': 'Entry Points: CLI/API entry functions | Workflow: Business logic | Leaves: Utilities'
};

async function init() {
    try {
        const res = await fetch('/api/brain');
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const data = await res.json();

        console.log(`Loaded: ${data.nodes?.length} nodes, ${data.edges?.length} edges, ${data.clusters?.length} clusters`);

        graphData = {
            nodes: data.nodes || [],
            links: data.edges || [],
            clusters: data.clusters || [],
            similarity_pairs: data.similarity_pairs || []
        };

        graphData.nodes.forEach(node => {
            node.force_x = null;
            node.force_y = null;
            node.force_z = null;
        });

        // Pre-compute neighbors for fast visibility checks
        graphData.nodes.forEach(n => { n.neighbors = new Set(); });
        graphData.links.forEach(link => {
            const srcId = typeof link.source === 'object' ? link.source.id : link.source;
            const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
            const a = graphData.nodes.find(n => n.id === srcId);
            const b = graphData.nodes.find(n => n.id === tgtId);
            if (a && b) {
                a.neighbors.add(b.id);
                b.neighbors.add(a.id);
            }
        });

        // Set initial filter 0.5%
        currentCentralityThreshold = 0.005;
        updateNodeVisibility();

        setupGraph();
        renderHeatmap();
        setupInfoButtons();

        setTimeout(() => {
            graphData.nodes.forEach(node => {
                node.force_x = node.x;
                node.force_y = node.y;
                node.force_z = node.z;
            });
        }, 5000);

    } catch (err) {
        console.error("Init error:", err);
        alert("Error loading graph: " + err.message);
    }
}

function setupGraph() {
    Graph
        .graphData(graphData)
        .nodeId('id')
        .nodeLabel(node => `${node.label}\n[${node.layer}] Cluster: ${node.cluster_id ?? 'none'}`)
        .nodeColor(node => {
            if (node.cluster_id !== undefined && node.cluster_id >= 0) {
                return CLUSTER_COLORS[node.cluster_id % CLUSTER_COLORS.length];
            }
            return LAYER_COLORS[node.layer] || '#888888';
        })
        .nodeVal(node => Math.max(3, computeCustomScore(node) * 300))
        .nodeOpacity(node => {
            if (!selectedNode) return 0.9;
            const isRelated = node.id === selectedNode.id || (node.neighbors && node.neighbors.has(selectedNode.id));
            return isRelated ? 1.0 : 0.1; // Deep fade for unrelated
        })
        .linkSource('source')
        .linkTarget('target')
        .linkColor(link => getLinkColor(link))
        .linkWidth(link => {
            if (!selectedNode) return link.type === 'call' ? 0.4 : 0.3;
            // Highlight connected edges
            const isConnected = link.source.id === selectedNode.id || link.target.id === selectedNode.id;
            return isConnected ? 0.5 : 0.1; // Thicker if connected, very thin if not
        })
        .linkOpacity(link => {
            if (!selectedNode) return 0.10;
            const isConnected = link.source.id === selectedNode.id || link.target.id === selectedNode.id;
            return isConnected ? 0.15 : 0.02; // Very visible if connected, almost invisible if not
        })

        .linkDirectionalArrowLength(link => link.type === 'call' ? 2 : 0)
        .linkDirectionalArrowRelPos(1)
        .linkDirectionalParticles(link => link.type === 'call' ? 5 : 0)  // More particles
        .linkDirectionalParticleSpeed(0.004)  // Faster for visibility
        .linkDirectionalParticleWidth(2)  // Larger than edge width!
        .linkDirectionalParticleColor(link => getParticleColor(link))  // Dynamic particle color
        .backgroundColor(isDarkTheme ? '#0a0a1a' : '#f0f0f0')
        .linkThreeObjectExtend(true)
        .linkThreeObject(link => {
            if (!link.label && !link.snippet) return null;
            if (!window.THREE) return null;

            // Simple text sprite for label
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 300;
            canvas.height = 64;
            ctx.fillStyle = isDarkTheme ? '#ffffff' : '#000000';
            ctx.font = 'bold 40px Sans-Serif';
            ctx.textAlign = 'center';
            ctx.fillText(link.label || link.type, 150, 46);

            const texture = new window.THREE.CanvasTexture(canvas);
            const material = new window.THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.95 });
            const sprite = new window.THREE.Sprite(material);
            sprite.scale.set(25, 3, 1); // Much bigger
            sprite.visible = false; // Hidden by default

            // Store specific data on sprite for updates
            sprite.userData = { isLabel: true, link: link };
            return sprite;
        })
        .linkPositionUpdate((sprite, { start, end }) => {
            if (!sprite) return false;
            const middlePos = Object.assign(...['x', 'y', 'z'].map(c => ({
                [c]: start[c] + (end[c] - start[c]) / 2
            })));
            Object.assign(sprite.position, middlePos);
            return true;
        })
        .onNodeClick(node => {
            selectedNode = node;
            focusNode(node);
            showDetails(node);
            updateLinkHighlighting();
        })
        .onLinkClick(link => {
            if (link.snippet) {
                showInfoModal(`Edge: ${link.source.label || link.source} -> ${link.target.label || link.target}\nType: ${link.type}\n\n${link.snippet}`);
            }
        })
        .onBackgroundClick(() => {
            selectedNode = null;
            updateLinkHighlighting();
        });

    setTimeout(() => Graph.zoomToFit(500, 50), 3000);
}

// Helper functions for colors/opacities
function getLinkColor(link) {
    const srcId = typeof link.source === 'object' ? link.source.id : link.source;
    const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
    const isConnected = selectedNode && (srcId === selectedNode.id || tgtId === selectedNode.id);

    // If we have a selection and this link is NOT connected, return a deeply faded gray
    if (selectedNode && !isConnected) {
        return isDarkTheme ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)';
    }

    // Otherwise standard colors (highlighted if connected)
    if (link.type === 'call') {
        if (isConnected) {
            // Check direction
            return srcId === selectedNode.id
                ? 'rgba(0, 191, 255, 0.8)'  // Outgoing Blue
                : 'rgba(255, 68, 68, 0.8)'; // Incoming Red
        }
        return isDarkTheme ? 'rgba(0,255,136,0.15)' : 'rgba(0,100,50,0.15)';
    }
    if (link.type === 'contains') return isDarkTheme ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';
    if (link.type === 'import') {
        if (isConnected) return 'rgba(255, 215, 0, 0.1)'; // Bright Gold
        return isDarkTheme ? 'rgba(255, 215, 0, 0.15)' : 'rgba(180, 140, 0, 0.15)';
    }
    return isDarkTheme ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
}

function getParticleColor(link) {
    if (!selectedNode) return '#00ff88';

    // If not connected, make it very faint or invisible
    const srcId = typeof link.source === 'object' ? link.source.id : link.source;
    const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
    const isConnected = srcId === selectedNode.id || tgtId === selectedNode.id;
    if (!isConnected) return 'rgba(0, 255, 136, 0.1)'; // Hide particles on unrelated edges

    // Distinct colors for flow
    if (srcId === selectedNode.id) return '#00bfffff'; // Outgoing Blue
    if (tgtId === selectedNode.id) return '#ff3333ff'; // Incoming Red
    return '#00ff88b9';
}




function updateLinkHighlighting() {
    Graph
        .linkColor(link => getLinkColor(link))
        .linkDirectionalParticleColor(link => getParticleColor(link));

    // Refresh node visibility (show neighbors)
    updateNodeVisibility();

    // Update label visibility
    const scene = Graph.scene();
    scene.traverse(obj => {
        if (obj.userData && obj.userData.isLabel) {
            const link = obj.userData.link;
            const srcId = typeof link.source === 'object' ? link.source.id : link.source;
            const tgtId = typeof link.target === 'object' ? link.target.id : link.target;

            let visible = false;

            if (selectedNode) {
                if (srcId === selectedNode.id || tgtId === selectedNode.id) {
                    visible = true;
                }
            }

            obj.visible = visible;
        }
    });
}


function computeCustomScore(node) {
    const pr = node.centrality || 0;
    const inf = node.neighbor_influence || 0;
    const sup = node.neighbor_support || 0;
    return pr * weights.pagerank + inf * weights.influence + sup * weights.support;
}

function updateWeights() {
    weights.pagerank = parseFloat(document.getElementById('weight-pagerank').value);
    weights.influence = parseFloat(document.getElementById('weight-influence').value);
    weights.support = parseFloat(document.getElementById('weight-support').value);

    document.getElementById('val-pagerank').innerText = weights.pagerank.toFixed(1);
    document.getElementById('val-influence').innerText = weights.influence.toFixed(1);
    document.getElementById('val-support').innerText = weights.support.toFixed(1);

    Graph.nodeVal(node => Math.max(3, computeCustomScore(node) * 300));

    // Refresh custom score display if a node is selected
    if (selectedNode) {
        refreshCustomScoreDisplay();
    }
}

function refreshCustomScoreDisplay() {
    if (!selectedNode) return;
    const maxScore = Math.max(...graphData.nodes.map(n => computeCustomScore(n)), 0.001);
    const scorePct = (computeCustomScore(selectedNode) / maxScore * 100).toFixed(1);
    const scoreEl = document.getElementById('custom-score-value');
    if (scoreEl) scoreEl.innerText = scorePct + '%';
}

// =============================================================================
// Info Tooltips
// =============================================================================

function setupInfoButtons() {
    document.querySelectorAll('.info-btn').forEach(btn => {
        const key = btn.dataset.info;
        if (key && INFO_TEXTS[key]) {
            btn.title = INFO_TEXTS[key];  // Hover tooltip
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                showInfoModal(INFO_TEXTS[key]);
            });
        }
    });
}

function showInfoModal(text) {
    let modal = document.getElementById('info-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'info-modal';
        modal.className = 'info-modal';
        modal.innerHTML = `<div class="info-modal-content"><p></p><button onclick="closeInfoModal()">OK</button></div>`;
        document.body.appendChild(modal);
    }
    modal.querySelector('p').innerText = text;
    modal.classList.add('visible');
}

function closeInfoModal() {
    const modal = document.getElementById('info-modal');
    if (modal) modal.classList.remove('visible');
}

// =============================================================================
// Cluster Spheres
// =============================================================================

function renderClusterSpheres() {
    const scene = Graph.scene();

    // Clear existing
    clusterSphereMeshes.forEach(m => scene.remove(m));
    clusterSphereMeshes = [];

    if (!showClusterSpheres) {
        console.log("Cluster spheres OFF");
        return;
    }

    // Safety check for THREE
    if (!window.THREE || !window.THREE.SphereGeometry) {
        console.error("THREE.js not fully loaded, cannot render spheres. Check window.THREE:", window.THREE);
        return;
    }
    const THREE = window.THREE;

    // Allow in all views now, as we have positions
    console.log(`Rendering spheres for ${graphData.clusters.length} clusters. View: ${currentView}`);

    const scale = 20;  // UMAP scale factor matching applyViewPositions

    graphData.clusters.forEach(cluster => {
        if (!cluster.centroid) {
            console.warn(`Cluster ${cluster.id} has no centroid`);
            return;
        }

        const color = CLUSTER_COLORS[cluster.id % CLUSTER_COLORS.length];

        // Sphere
        const radius = 6 + Math.sqrt(cluster.count) * 2.5;
        const geometry = new THREE.SphereGeometry(radius, 32, 32);
        const material = new THREE.MeshLambertMaterial({ // Use Lambert for better 3D look
            color: color,
            transparent: true,
            opacity: 0.25,
            side: THREE.DoubleSide
        });
        const sphere = new THREE.Mesh(geometry, material);

        // Position depends on view, but centroid is static UMAP coord usually
        // We need to match where nodes are. 
        // If view is 'force', clusters might not align unless we calculate force centroids.
        // For 'semantic'/'hybrid', we use the UMAP centroids.

        let cx = cluster.centroid.x * scale;
        let cy = cluster.centroid.y * scale;
        let cz = cluster.centroid.z * scale;

        // In force layout, clusters are scattered, so spheres don't make sense unless we track them.
        // But user wants them visible so we allow it, but maybe warn or assume semantic placement.
        if (currentView === 'force') {
            // For force view, we can't easily show static spheres unless we compute live centroids.
            // We'll skip for force to avoid confusion, or use average of current node positions.
            return;
        } else if (currentView === 'hybrid') {
            // Mix not needed for centroid as it's purely semantic property usually
            // But if nodes moved, sphere should too. 
            // Simplification: Render at scale
        }

        sphere.position.set(cx, cy, cz);
        scene.add(sphere);
        clusterSphereMeshes.push(sphere);

        // Label sprite
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 512;
        canvas.height = 128; // taller
        ctx.fillStyle = color;
        ctx.font = 'bold 32px Arial'; // Bigger font
        ctx.textAlign = 'center';
        ctx.fillText(cluster.name, 256, 64);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.9 });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.position.set(cx, cy + radius + 10, cz);
        sprite.scale.set(40, 10, 1);
        scene.add(sprite);
        clusterSphereMeshes.push(sprite);
    });

    console.log(`Rendered ${clusterSphereMeshes.length / 2} spheres`);
}

function toggleClusterSpheres() {
    showClusterSpheres = !showClusterSpheres;
    console.log(`Cluster spheres: ${showClusterSpheres}`);
    renderClusterSpheres();
}

// =============================================================================
// Similarity Heatmap with Code Comparison
// =============================================================================

function renderHeatmap() {
    const container = document.getElementById('heatmap-container');
    if (!container) return;

    const pairs = graphData.similarity_pairs.slice(0, 25);
    if (pairs.length === 0) {
        container.innerHTML = '<p>No similarity data</p>';
        return;
    }

    let html = '<table class="heatmap-table"><thead><tr><th>Source</th><th>Target</th><th>Sim</th></tr></thead><tbody>';

    pairs.forEach((pair, idx) => {
        const srcLabel = pair.source_label || pair.source.split('::').pop();
        const tgtLabel = pair.target_label || pair.target.split('::').pop();
        const pct = (pair.similarity * 100).toFixed(0);
        const hue = Math.round(pair.similarity * 120);
        html += `<tr onclick="showCodeComparison(${idx})" class="clickable-row">
            <td title="${pair.source}">${srcLabel}</td>
            <td title="${pair.target}">${tgtLabel}</td>
            <td style="background:hsl(${hue},70%,35%);text-align:center">${pct}%</td>
        </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

async function showCodeComparison(pairIndex) {
    const pair = graphData.similarity_pairs[pairIndex];
    if (!pair) return;

    const panel = document.getElementById('comparison-panel');
    panel.classList.remove('hidden');

    document.getElementById('comp-src-label').innerText = pair.source_label || pair.source.split('::').pop();
    document.getElementById('comp-tgt-label').innerText = pair.target_label || pair.target.split('::').pop();
    document.getElementById('comp-similarity').innerText = `${(pair.similarity * 100).toFixed(1)}% similar`;

    // Load source code
    const srcFile = pair.source_file || pair.source.split('::')[0];
    const tgtFile = pair.target_file || pair.target.split('::')[0];

    try {
        const [srcRes, tgtRes] = await Promise.all([
            fetch(`/api/source/${encodeURIComponent(srcFile)}`),
            fetch(`/api/source/${encodeURIComponent(tgtFile)}`)
        ]);

        document.getElementById('comp-src-code').innerText = srcRes.ok ?
            (await srcRes.text()).slice(0, 2000) : 'Source unavailable';
        document.getElementById('comp-tgt-code').innerText = tgtRes.ok ?
            (await tgtRes.text()).slice(0, 2000) : 'Source unavailable';
    } catch (e) {
        document.getElementById('comp-src-code').innerText = 'Error loading';
        document.getElementById('comp-tgt-code').innerText = 'Error loading';
    }

    // Highlight nodes
    const srcNode = graphData.nodes.find(n => n.id === pair.source);
    if (srcNode) focusNode(srcNode);
}

function closeComparison() {
    document.getElementById('comparison-panel').classList.add('hidden');
}

function toggleHeatmap() {
    document.getElementById('heatmap-panel').classList.toggle('hidden');
}

// =============================================================================
// Info Tooltips
// =============================================================================

function showInfo(key) {
    const text = INFO_TEXTS[key] || 'No description available';
    alert(text);  // Simple implementation - could be replaced with tooltip
}

// =============================================================================
// View Modes
// =============================================================================

function setView(view) {
    currentView = view;

    document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-${view}`).classList.add('active');
    document.getElementById('blend-control').style.display = view === 'hybrid' ? 'block' : 'none';

    applyViewPositions();

    // Re-render cluster spheres for new view
    setTimeout(() => renderClusterSpheres(), 500);
}

function applyViewPositions() {
    graphData.nodes.forEach(node => {
        const forceX = node.force_x ?? node.x ?? 0;
        const forceY = node.force_y ?? node.y ?? 0;
        const forceZ = node.force_z ?? node.z ?? 0;

        const umapX = (node.umap_x ?? 0) * 20;
        const umapY = (node.umap_y ?? 0) * 20;
        const umapZ = (node.umap_z ?? 0) * 20;

        if (currentView === 'force') {
            node.fx = forceX; node.fy = forceY; node.fz = forceZ;
        } else if (currentView === 'semantic') {
            node.fx = umapX; node.fy = umapY; node.fz = umapZ;
        } else {
            node.fx = forceX * (1 - hybridBlend) + umapX * hybridBlend;
            node.fy = forceY * (1 - hybridBlend) + umapY * hybridBlend;
            node.fz = forceZ * (1 - hybridBlend) + umapZ * hybridBlend;
        }
    });

    Graph.graphData(graphData);

    setTimeout(() => {
        if (currentView === 'force') {
            graphData.nodes.forEach(node => { node.fx = null; node.fy = null; node.fz = null; });
        }
    }, 1000);
}

function setBlend(value) {
    hybridBlend = parseFloat(value);
    document.getElementById('blend-value').innerText = `${Math.round(hybridBlend * 100)}%`;
    if (currentView === 'hybrid') {
        applyViewPositions();
        setTimeout(() => renderClusterSpheres(), 600);
    }
}

function focusNode(node) {
    const distance = 60;
    const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
    Graph.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
        node, 1500
    );
}

function toggleTheme() {
    isDarkTheme = !isDarkTheme;
    Graph.backgroundColor(isDarkTheme ? '#0a0a1a' : '#f0f0f0');
    document.body.classList.toggle('light-theme', !isDarkTheme);
}

async function showDetails(node) {
    const panel = document.getElementById('details-panel');
    panel.classList.remove('hidden');

    document.getElementById('node-label').innerText = node.label;

    const cluster = graphData.clusters.find(c => c.id === node.cluster_id);
    const clusterName = cluster ? cluster.name : 'None';

    // Calculate max values for percentage display
    const maxInfluence = Math.max(...graphData.nodes.map(n => n.neighbor_influence || 0), 0.001);
    const maxSupport = Math.max(...graphData.nodes.map(n => n.neighbor_support || 0), 0.001);
    const maxScore = Math.max(...graphData.nodes.map(n => computeCustomScore(n)), 0.001);

    const infPct = ((node.neighbor_influence || 0) / maxInfluence * 100).toFixed(1);
    const supPct = ((node.neighbor_support || 0) / maxSupport * 100).toFixed(1);
    const scorePct = (computeCustomScore(node) / maxScore * 100).toFixed(1);

    document.getElementById('node-meta').innerHTML = `
        <div><strong>Type:</strong> ${node.type}</div>
        <div><strong>Layer:</strong> <span style="color:${LAYER_COLORS[node.layer]}">${node.layer}</span></div>
        <div><strong>Cluster:</strong> ${clusterName}</div>
        <div><strong>PageRank:</strong> ${(node.centrality * 100).toFixed(2)}%</div>
        <div><strong>Neighbor Influence:</strong> ${infPct}%</div>
        <div><strong>Neighbor Support:</strong> ${supPct}%</div>
        <div><strong>Custom Score:</strong> <span id="custom-score-value">${scorePct}%</span></div>
        <div><strong>In/Out:</strong> ${node.in_degree ?? 0} / ${node.out_degree ?? 0}</div>
        <div class="edge-legend"><span style="color:#ff5050">● Incoming</span> <span style="color:#0096ff">● Outgoing</span></div>
    `;

    if (node.file) {
        document.getElementById('node-source').innerText = "Loading...";
        try {
            const res = await fetch(`/api/source/${encodeURIComponent(node.file)}`);
            if (res.ok) {
                const text = await res.text();
                document.getElementById('node-source').innerText = text.slice(0, 3000) + (text.length > 3000 ? "\n..." : "");
            } else {
                document.getElementById('node-source').innerText = "Source unavailable";
            }
        } catch (e) {
            document.getElementById('node-source').innerText = "Error";
        }
    }
}

function closeDetails() {
    document.getElementById('details-panel').classList.add('hidden');
    selectedNode = null;
    updateLinkHighlighting();
}

async function performSearch() {
    const query = document.getElementById('semantic-search').value;
    if (!query) return;

    const btn = document.querySelector('.search-box button');
    btn.disabled = true;
    btn.innerText = 'Searching...';

    try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        alert(`Found ${data.results?.length || 0} results. Check console.`);
    } catch (e) {
        alert("Search error: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerText = 'Ask';
    }
}

// Global threshold
let currentCentralityThreshold = 0.005; // 0.5% default

function filterByCentrality(threshold) {
    currentCentralityThreshold = parseFloat(threshold);
    document.getElementById('centrality-value').innerText = `${(currentCentralityThreshold * 100).toFixed(1)}%`;
    Graph.refresh(); // Trigger visibility update
}

// ... inside init() or wherever Graph is configured ...
// We need to move the nodeVisibility logic into the Graph definition or update it.
// Since Graph is const defined at top, we can use Graph.nodeVisibility(...).

function updateNodeVisibility() {
    Graph.nodeVisibility(node => {
        // Always show files ??
        if (node.type === 'file') return true;

        // Always show selected node
        if (selectedNode && node.id === selectedNode.id) return true;

        // Auto-show neighbors of selected node
        if (selectedNode) {
            const isNeighbor = node.neighbors && node.neighbors.has(selectedNode.id);
            if (isNeighbor) return true;
        }

        // Otherwise check threshold
        return (node.centrality || 0) >= currentCentralityThreshold;
    });
}


init();
