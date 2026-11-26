/* Memory-bounded gravity viewer.
 *
 * The manifest and viewer_summary.json are intentionally the only large-scale
 * metadata read during startup. CSV partitions are parsed in a Web Worker and
 * cached per bundle identity/time slot in IndexedDB. The viewer never builds a
 * JavaScript array for the complete OD table.
 */
(() => {
  'use strict';

  const CACHE_VERSION = 4;
  const REQUIRED = ['bundle_manifest.json', 'model_specification.json', 'report.json'];
  const $ = (id) => document.getElementById(id);
  const status = $('status');
  const leafletScript = document.createElement('script');
  leafletScript.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
  leafletScript.integrity = 'sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=';
  leafletScript.crossOrigin = '';
  document.head.appendChild(leafletScript);
  const workerSource = `
    const parseLine = (source) => {
      const cells = []; let cell = ''; let quoted = false;
      for (let i = 0; i < source.length; i += 1) {
        const char = source[i];
        if (char === '"') {
          if (quoted && source[i + 1] === '"') { cell += '"'; i += 1; }
          else quoted = !quoted;
        } else if (char === ',' && !quoted) { cells.push(cell); cell = ''; }
        else cell += char;
      }
      cells.push(cell); return cells;
    };
    self.onmessage = async (event) => {
      const {file, filters, limit, chunkSize, offset = 0, scanAll = false, rankBy = null, rankAbs = false, streamMatches = false} = event.data;
      let header = null; let carry = ''; let matched = 0; let sent = 0; const ranked = [];
      let streamed = [];
      const postStreamBatch = async () => {
        if (!streamed.length) return;
        const rows = streamed; streamed = [];
        await new Promise((resolve) => { self.onmessage = () => resolve(); self.postMessage({type: 'rows', rows}); });
      };
      const accept = (row) => Object.entries(filters || {}).every(([key, value]) => Array.isArray(value) ? value.map(String).includes(String(row[key] ?? '')) : String(row[key] ?? '') === String(value));
      const rank = (row) => { const value = Number(row[rankBy]); return rankAbs ? Math.abs(value) : value; };
      const siftUp = (index) => {
        while (index > 0) {
          const parent = Math.floor((index - 1) / 2);
          if (ranked[parent].score <= ranked[index].score) break;
          [ranked[parent], ranked[index]] = [ranked[index], ranked[parent]];
          index = parent;
        }
      };
      const siftDown = (index) => {
        while (true) {
          const left = index * 2 + 1; const right = left + 1;
          let smallest = index;
          if (left < ranked.length && ranked[left].score < ranked[smallest].score) smallest = left;
          if (right < ranked.length && ranked[right].score < ranked[smallest].score) smallest = right;
          if (smallest === index) break;
          [ranked[index], ranked[smallest]] = [ranked[smallest], ranked[index]];
          index = smallest;
        }
      };
      const retain = (row) => {
        const score = rank(row); if (!Number.isFinite(score)) return;
        const candidate = {row, score};
        if (ranked.length < limit) { ranked.push(candidate); siftUp(ranked.length - 1); return; }
        if (score > ranked[0].score) { ranked[0] = candidate; siftDown(0); }
      };
      const emit = async (lines, final = false) => {
        const rows = [];
        for (const line of lines) {
          if (!line) continue;
          const cells = parseLine(line);
          if (!header) { header = cells; continue; }
          const row = Object.fromEntries(header.map((name, index) => [name, cells[index] ?? '']));
          if (accept(row)) {
            if (streamMatches) { streamed.push(row); matched += 1; if (streamed.length >= 512) await postStreamBatch(); }
            else if (rankBy) { retain(row); matched += 1; }
            else { if (matched >= offset && sent < limit) { rows.push(row); sent += 1; } matched += 1; }
          }
        }
        if (rows.length) self.postMessage({type: 'rows', rows});
        if (streamMatches && final && streamed.length) await postStreamBatch();
        if (final && !rankBy && !streamMatches) self.postMessage({type: 'done', truncated: matched > offset + sent, total: matched});
      };
      for (let offset = 0; offset < file.size && (scanAll || rankBy || sent < limit); offset += chunkSize) {
        carry += await file.slice(offset, Math.min(file.size, offset + chunkSize)).text();
        const lines = carry.split(/\\r?\\n/); carry = lines.pop() || ''; await emit(lines);
      }
      if (carry) await emit([carry], true);
      if (streamMatches) {
        self.postMessage({type: 'done', truncated: false, total: matched});
      } else if (!carry && !rankBy) self.postMessage({type: 'done', truncated: matched > offset + sent, total: matched});
      if (rankBy) { ranked.sort((a, b) => b.score - a.score); self.postMessage({type: 'rows', rows: ranked.map((item) => item.row)}); self.postMessage({type: 'done', truncated: matched > ranked.length, total: matched}); }
    };
  `;
  const workerUrl = URL.createObjectURL(new Blob([workerSource], {type: 'text/javascript'}));

  const viewer = {
    files: [], fileMap: new Map(), manifest: null, summary: {}, model: {}, report: {},
    map: null, stops: [], stopsById: new Map(), stopMarkers: new Map(), markerLayer: null, selectedOd: null,
    cacheDb: null, cacheIdentity: '', odRows: [],
    flowRows: [], flowBufferedRows: [], flowMatchedCount: 0, flowLoadedLimit: 0, flowPage: 0,
    flowSortColumn: '', flowSortDirection: 'desc', flowTitle: '',
    mapClickMode: 'od', measurementLocationManifest: null, selectedMeasurementStop: '',
    selectedMeasurement: null, measurementRows: [], measurementRowPage: 0, measurementRowsPerPage: 50,
    measurementContributionPage: 0, measurementContributions: new Map(), measurementOdRows: [],
    measurementOdSortColumn: 'modeled_mean_contribution', measurementOdSortDirection: 'desc', measurementQueryId: 0,
    measurementQueryTimer: null, measurementQueryController: null, measurementQueryRequestId: '',
    measurementStopSelectionId: 0, departureIntervals: new Map(), departureTimeZone: '',
    departureIntervalMetadataError: '',
    routePathStatusId: 0, routePathRequestId: '', routePathController: null,
    routePathSourceAvailable: false,
  };

  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
  const num = (value) => { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : null; };
  const display = (value) => { const parsed = num(value); return parsed === null ? '—' : parsed.toLocaleString(undefined, {maximumFractionDigits: 2}); };
  const setStatus = (message, kind = '') => { status.className = `status ${kind}`; status.textContent = message; };
  const setFlowStatus = (message, kind = '') => { $('flow-status').className = `status ${kind}`; $('flow-status').textContent = message; };
  const fileName = (path) => path.split('/').pop();
  const findFile = (name) => viewer.fileMap.get(name) || viewer.files.find((file) => file.name === fileName(name) || file.webkitRelativePath.endsWith(`/${name}`));
  const readText = async (name) => { const file = findFile(name); if (!file) throw new Error(`Missing ${name}`); return file.text(); };
  const readJson = async (name) => JSON.parse(await readText(name));

  function parseSmallCsv(source) {
    const rows = []; let row = []; let cell = ''; let quoted = false;
    for (let i = 0; i < source.length; i += 1) {
      const char = source[i];
      if (char === '"') { if (quoted && source[i + 1] === '"') { cell += '"'; i += 1; } else quoted = !quoted; }
      else if (char === ',' && !quoted) { row.push(cell); cell = ''; }
      else if ((char === '\n' || char === '\r') && !quoted) { if (char === '\r' && source[i + 1] === '\n') i += 1; row.push(cell); cell = ''; if (row.some((item) => item !== '')) rows.push(row); row = []; }
      else cell += char;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    const headers = rows.shift() || [];
    return rows.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ''])));
  }

  function idbRequest(request) {
    return new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
  }
  async function openCache() {
    if (viewer.cacheDb) return viewer.cacheDb;
    viewer.cacheDb = await new Promise((resolve, reject) => {
      const request = indexedDB.open('gravity-viewer-cache', 2);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains('partitions')) db.createObjectStore('partitions', {keyPath: 'key'});
        if (!db.objectStoreNames.contains('measurementContributions')) {
          const store = db.createObjectStore('measurementContributions', {keyPath: ['identity', 'measurement_index', 'rank', 'od_index']});
          store.createIndex('byMeasurement', ['identity', 'measurement_index'], {unique: false});
          store.createIndex('byRank', ['identity', 'measurement_index', 'rank', 'od_index'], {unique: true});
        }
      };
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    return viewer.cacheDb;
  }
  async function cacheGet(key) { try { const db = await openCache(); return idbRequest(db.transaction('partitions').objectStore('partitions').get(key)); } catch (_) { return null; } }
  async function cachePut(record) { try { const db = await openCache(); await idbRequest(db.transaction('partitions', 'readwrite').objectStore('partitions').put(record)); } catch (_) { setFlowStatus('Browser storage quota or IndexedDB is unavailable; continuing without persistent cache.', ''); } }
  async function clearCache() { try { const db = await openCache(); const tx = db.transaction(['partitions', 'measurementContributions'], 'readwrite'); tx.objectStore('partitions').clear(); tx.objectStore('measurementContributions').clear(); await new Promise((resolve, reject) => { tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error); }); viewer.measurementContributions.clear(); setFlowStatus('Viewer cache cleared.'); } catch (_) { setFlowStatus('Viewer cache could not be cleared.', 'bad'); } }

  async function sha256(file) {
    const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
    return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
  }
  async function validatePartition(file, descriptor) {
    if (!file) throw new Error(`Partition file is missing: ${descriptor.path || descriptor.file}`);
    const expectedSize = descriptor.size_bytes ?? descriptor.size;
    if (expectedSize !== undefined && file.size !== expectedSize) throw new Error(`Partition size mismatch: ${descriptor.path || descriptor.file}`);
    if (descriptor.sha256) {
      const actual = await sha256(file);
      if (actual !== descriptor.sha256) throw new Error(`Partition checksum mismatch: ${descriptor.path || descriptor.file}`);
    }
  }
  async function readPartition(kind, bin, filters = {}, limit = 5000, options = {}) {
    const returnDetails = options.returnDetails === true;
    const workerOptions = {...options}; delete workerOptions.returnDetails;
    const group = viewer.summary[kind];
    if (!group?.available) return returnDetails ? {rows: [], total: 0} : [];
    const matching = (group.partitions || []).filter((item) => item.key === bin);
    const descriptors = bin === 'all'
      ? (group.partitions || [])
      : (kind === 'measurement_od_contributions' ? matching : matching.slice(0, 1));
    const output = []; let total = 0;
    const rankOutput = () => {
      output.sort((a, b) => {
        const leftValue = Number(a[workerOptions.rankBy]); const rightValue = Number(b[workerOptions.rankBy]);
        const left = workerOptions.rankAbs ? Math.abs(leftValue) : leftValue;
        const right = workerOptions.rankAbs ? Math.abs(rightValue) : rightValue;
        return right - left;
      });
      if (output.length > limit) output.length = limit;
    };
    for (const item of descriptors) {
      const path = item.path || item.file; const file = findFile(path); await validatePartition(file, item);
      const identity = `${viewer.cacheIdentity}|${kind}|${path}|${item.sha256 || file.size}|${JSON.stringify(filters)}|${JSON.stringify(workerOptions)}|${limit}|v${CACHE_VERSION}`;
      const cacheKey = identity;
      const cached = await cacheGet(cacheKey);
      if (cached && Array.isArray(cached.rows)) {
        output.push(...cached.rows);
        total += Number(cached.total) || cached.rows.length;
        if (workerOptions.rankBy) rankOutput();
        continue;
      }
      const result = await new Promise((resolve, reject) => {
        const worker = new Worker(workerUrl); const collected = [];
        let matched = 0;
        worker.onmessage = (event) => {
          if (event.data.type === 'rows') collected.push(...event.data.rows);
          else { matched = Number(event.data.total) || 0; worker.terminate(); resolve({rows: collected, total: matched}); }
        };
        worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'CSV worker failed.')); };
        worker.postMessage({file, filters, limit: Math.max(limit, 1), chunkSize: 1024 * 1024, ...workerOptions});
      });
      await cachePut({key: cacheKey, identity, partition: path, rows: result.rows, total: result.total});
      output.push(...result.rows); total += result.total;
      if (workerOptions.rankBy) rankOutput();
    }
    return returnDetails ? {rows: output.slice(0, limit), total} : output.slice(0, limit);
  }

  async function readDescriptorPage(descriptor, filters, offset, limit, cacheScope) {
    const path = descriptor.path || descriptor.file; const file = findFile(path);
    await validatePartition(file, descriptor);
    const cacheKey = `${viewer.cacheIdentity}|${cacheScope}|${path}|${JSON.stringify(filters)}|${offset}|${limit}|v${CACHE_VERSION}`;
    const cached = await cacheGet(cacheKey);
    if (cached && Array.isArray(cached.rows)) return cached.rows;
    const rows = await new Promise((resolve, reject) => {
      const worker = new Worker(workerUrl); const collected = [];
      worker.onmessage = (event) => { if (event.data.type === 'rows') collected.push(...event.data.rows); else { worker.terminate(); resolve(collected); } };
      worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'CSV worker failed.')); };
      worker.postMessage({file, filters, limit, offset, chunkSize: 1024 * 1024});
    });
    await cachePut({key: cacheKey, identity: cacheKey, partition: path, rows});
    return rows;
  }

  async function measurementLocationIndex() {
    const summary = viewer.summary.measurement_locations || {};
    if (!summary.available) return null;
    if (viewer.measurementLocationManifest) return viewer.measurementLocationManifest;
    if (typeof summary.manifest_path !== 'string' || !summary.manifest_path.startsWith('partitions/measurement_locations/') || summary.manifest_path.includes('..')) throw new Error('Measurement location lookup path is unsafe.');
    const file = findFile(summary.manifest_path);
    await validatePartition(file, {path: summary.manifest_path, size_bytes: undefined, sha256: summary.manifest_sha256});
    viewer.measurementLocationManifest = JSON.parse(await file.text());
    const index = viewer.measurementLocationManifest; const expected = Number(viewer.manifest.row_counts?.['predicted_measurements.csv']);
    if (index.schema_version !== 1 || index.dataset_type !== 'gravity_measurement_location_lookup' || !Number.isInteger(index.number_of_measurements) || index.number_of_measurements !== expected || index.number_of_stops !== Object.keys(index.stops || {}).length || index.number_of_measurements !== Number(index.indexed_measurements) + Number(index.missing_stop_rows)) throw new Error('Measurement location lookup schema or row counts are invalid.');
    if (!Array.isArray(index.columns) || !['measurement_index', 'measurement_type', 'line', 'direction', 'observation_time', 'time_period', 'stop', 'observed', 'modeled'].every((column, position) => index.columns[position] === column)) throw new Error('Measurement location lookup columns are invalid.');
    const source = index.source_identity || {}; const bundle = viewer.manifest;
    for (const [field, actual] of [['fit_fingerprint', bundle.model_fingerprint], ['assignment_fingerprint', bundle.assignment_fingerprint], ['canonical_index_fingerprint', bundle.canonical_index_fingerprint], ['binding_fingerprint', bundle.binding_fingerprint], ['od_layout_fingerprint', bundle.od_layout_fingerprint]]) if (actual != null && source[field] !== actual) throw new Error(`Measurement location lookup ${field} differs from bundle provenance.`);
    return viewer.measurementLocationManifest;
  }

  async function measurementsAtStop(stopId, page = 0, pageSize = 50) {
    const manifest = await measurementLocationIndex();
    if (!manifest) return {rows: [], total: 0};
    const stops = manifest.stops || {};
    if (!Object.prototype.hasOwnProperty.call(stops, stopId)) return {rows: [], total: 0};
    const entry = stops[stopId]; const total = Number(entry.row_count) || 0;
    const start = page * pageSize; const end = Math.min(total, start + pageSize); const rows = [];
    if (start >= end) return {rows, total};
    let partitionStart = 0;
    for (const descriptor of entry.partitions || []) {
      if (descriptor.key !== stopId || Number(descriptor.row_count) < 1 || Number(descriptor.row_count) > Number(manifest.partition_rows_limit) || !String(descriptor.path || '').startsWith('partitions/measurement_locations/') || String(descriptor.path || '').includes('..')) throw new Error('Measurement location partition descriptor is invalid.');
      const partitionEnd = partitionStart + Number(descriptor.row_count || 0);
      if (partitionEnd > start && partitionStart < end) {
        const localStart = Math.max(0, start - partitionStart); const localEnd = Math.min(Number(descriptor.row_count), end - partitionStart);
        const chunk = await readDescriptorPage(descriptor, {stop: stopId}, localStart, localEnd - localStart, 'measurement-location');
        rows.push(...chunk);
      }
      partitionStart = partitionEnd;
      if (partitionStart >= end) break;
    }
    if (rows.length !== end - start || rows.some((row) => row.stop !== stopId)) throw new Error('Measurement location lookup page is incomplete or inconsistent.');
    return {rows, total};
  }

  function measurementContributionCacheIdentity() {
    return `${viewer.cacheIdentity}|measurement-contributions|modeled-mean-desc-v2`;
  }
  function measurementContributionMarker(measurementIndex) {
    return `${measurementContributionCacheIdentity()}|${measurementIndex}|v${CACHE_VERSION}`;
  }

  async function writeMeasurementContributionBatch(measurementIndex, rows) {
    if (!rows.length) return;
    const db = await openCache(); const tx = db.transaction('measurementContributions', 'readwrite'); const store = tx.objectStore('measurementContributions');
    rows.forEach((row) => {
      const value = Number(row.modeled_mean_contribution); const odIndex = Number(row.od_index);
      if (!Number.isFinite(value) || !Number.isSafeInteger(odIndex)) throw new Error('Contribution partition contains an invalid value or OD index.');
      store.put({...row, identity: measurementContributionCacheIdentity(), measurement_index: Number(measurementIndex), od_index: odIndex, rank: -value});
    });
    await new Promise((resolve, reject) => { tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error); });
  }

  async function cacheMeasurementContributions(measurementIndex) {
    const marker = measurementContributionMarker(measurementIndex); const cached = await cacheGet(marker);
    if (cached?.complete) return Number(cached.rowCount) || 0;
    if (viewer.measurementContributions.has(measurementIndex)) return viewer.measurementContributions.get(measurementIndex);
    const task = (async () => {
      const summary = viewer.summary.measurement_od_contributions || {};
      let total = 0;
      for (const descriptor of summary.partitions || []) {
        const path = descriptor.path || descriptor.file; const file = findFile(path);
        await validatePartition(file, descriptor);
        await new Promise((resolve, reject) => {
          const worker = new Worker(workerUrl);
          worker.onmessage = async (event) => {
            if (event.data.type === 'rows') {
              try { total += event.data.rows.length; await writeMeasurementContributionBatch(measurementIndex, event.data.rows); worker.postMessage({type: 'ack'}); }
              catch (error) { worker.terminate(); reject(error); }
            } else { worker.terminate(); resolve(); }
          };
          worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'Contribution CSV worker failed.')); };
          worker.postMessage({file, filters: {measurement_index: String(measurementIndex)}, scanAll: true, streamMatches: true, chunkSize: 1024 * 1024});
        });
      }
      await cachePut({key: marker, identity: marker, complete: true, rowCount: total});
      return total;
    })();
    viewer.measurementContributions.set(measurementIndex, task);
    try { return await task; } finally { viewer.measurementContributions.delete(measurementIndex); }
  }

  async function measurementContributionPage(measurementIndex, page, pageSize) {
    const rowCount = await cacheMeasurementContributions(measurementIndex);
    const db = await openCache(); const tx = db.transaction('measurementContributions', 'readonly'); const store = tx.objectStore('measurementContributions');
    const identity = measurementContributionCacheIdentity();
    const count = await idbRequest(store.index('byMeasurement').count(IDBKeyRange.only([identity, Number(measurementIndex)])));
    if (count !== rowCount) throw new Error('Cached contribution index row count is inconsistent. Clear the viewer cache and retry.');
    const start = page * pageSize; const rows = [];
    if (start >= count) return {rows, total: count};
    const range = IDBKeyRange.bound([identity, Number(measurementIndex), -Number.MAX_VALUE, 0], [identity, Number(measurementIndex), Number.MAX_VALUE, Number.MAX_SAFE_INTEGER]);
    await new Promise((resolve, reject) => {
      let skipped = 0; const request = store.index('byRank').openCursor(range);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const cursor = request.result;
        if (!cursor || rows.length >= pageSize) { resolve(); return; }
        if (skipped < start) skipped += 1;
        else { const {identity, rank, ...row} = cursor.value; rows.push(row); }
        cursor.continue();
      };
    });
    return {rows, total: count};
  }

  async function queryLocalMeasurementPage(measurementIndex, page, pageSize, requestId, signal) {
    const summary = viewer.summary.measurement_od_contributions || {};
    const base = new URL($('measurement-query-service').value.trim());
    if (base.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname)) {
      throw new Error('The on-demand query service must be an HTTP service on this computer.');
    }
    const endpoint = new URL('/api/measurement-od-contributions', base.origin);
    endpoint.searchParams.set('measurement_index', String(measurementIndex));
    endpoint.searchParams.set('page', String(page));
    endpoint.searchParams.set('page_size', String(pageSize));
    endpoint.searchParams.set('request_id', requestId);
    const response = await fetch(endpoint, {cache: 'no-store', signal});
    let payload;
    try { payload = await response.json(); }
    catch (_) { throw new Error(`Local query service returned HTTP ${response.status} without a valid JSON response.`); }
    if (!response.ok) throw new Error(payload.error || `Local query service returned HTTP ${response.status}.`);
    if (payload.fit_fingerprint !== summary.fit_fingerprint || payload.mapping_artifact_fingerprint !== summary.mapping_artifact_fingerprint) {
      throw new Error('Local query service fit or mapping identity differs from this viewer bundle.');
    }
    if (Number(payload.measurement_index) !== measurementIndex || !Array.isArray(payload.rows) || !Number.isSafeInteger(Number(payload.total))) {
      throw new Error('Local query service returned an invalid measurement page.');
    }
    return payload;
  }

  async function queryLocalMeasurementProgress(measurementIndex, requestId, signal) {
    const summary = viewer.summary.measurement_od_contributions || {};
    const base = new URL($('measurement-query-service').value.trim());
    if (base.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname)) {
      throw new Error('The on-demand query service must be an HTTP service on this computer.');
    }
    const endpoint = new URL('/api/measurement-query-progress', base.origin);
    endpoint.searchParams.set('measurement_index', String(measurementIndex));
    endpoint.searchParams.set('request_id', requestId);
    const response = await fetch(endpoint, {cache: 'no-store', signal});
    let payload;
    try { payload = await response.json(); }
    catch (_) { throw new Error(`Local query service returned HTTP ${response.status} without valid progress data.`); }
    if (!response.ok) throw new Error(payload.error || `Local query service returned HTTP ${response.status}.`);
    if (Number(payload.measurement_index) !== measurementIndex || payload.request_id !== requestId || payload.fit_fingerprint !== summary.fit_fingerprint || payload.mapping_artifact_fingerprint !== summary.mapping_artifact_fingerprint) {
      throw new Error('Local query progress belongs to a different measurement, fit, or mapping artifact.');
    }
    return payload;
  }

  async function queryLocalOdPage(odIndex, pageSize, requestId, signal) {
    const summary = viewer.summary.measurement_od_contributions || {};
    const base = new URL($('measurement-query-service').value.trim());
    if (base.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname)) {
      throw new Error('The on-demand query service must be an HTTP service on this computer.');
    }
    const endpoint = new URL('/api/od-measurement-contributions', base.origin);
    endpoint.searchParams.set('od_index', String(odIndex));
    endpoint.searchParams.set('page', '0');
    endpoint.searchParams.set('page_size', String(pageSize));
    endpoint.searchParams.set('request_id', requestId);
    const response = await fetch(endpoint, {cache: 'no-store', signal});
    let payload;
    try { payload = await response.json(); }
    catch (_) { throw new Error(`Local query service returned HTTP ${response.status} without a valid JSON response.`); }
    if (!response.ok) throw new Error(payload.error || `Local query service returned HTTP ${response.status}.`);
    if (payload.fit_fingerprint !== summary.fit_fingerprint || payload.mapping_artifact_fingerprint !== summary.mapping_artifact_fingerprint) {
      throw new Error('Local query service fit or mapping identity differs from this viewer bundle.');
    }
    if (Number(payload.od_index) !== odIndex || !Array.isArray(payload.rows) || !Number.isSafeInteger(Number(payload.total))) {
      throw new Error('Local query service returned an invalid OD contribution page.');
    }
    return payload;
  }

  function cancelLocalQuery(requestId) {
    const summary = viewer.summary.measurement_od_contributions || {};
    if (!requestId || summary.query_mode !== 'local_service') return;
    try {
      const base = new URL($('measurement-query-service').value.trim());
      if (base.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname)) return;
      const endpoint = new URL('/api/cancel-query', base.origin);
      endpoint.searchParams.set('request_id', requestId);
      void fetch(endpoint, {method: 'POST', cache: 'no-store', keepalive: true}).catch(() => {});
    } catch (_) {
      // The query service may have been stopped while the viewer was open.
    }
  }

  function hasInformativeValue(value) {
    if (value === null || value === undefined) return false;
    if (typeof value === 'number') return Number.isFinite(value);
    const normalized = String(value).trim().toLowerCase();
    return normalized !== '' && normalized !== 'nan' && normalized !== 'null';
  }
  function flowColumnLabel(column) {
    return ({
      estimated_to_prior_ratio: 'Estimated / baseline ratio',
      inference_score: 'Free/fixed indicator (1=estimated, 0=fixed)',
    })[column] || column;
  }
  function compareFlowValues(left, right, column) {
    const leftValue = left[column]; const rightValue = right[column];
    const leftText = String(leftValue ?? '').trim(); const rightText = String(rightValue ?? '').trim();
    if (!leftText && rightText) return 1;
    if (leftText && !rightText) return -1;
    if (column === 'departure_time_bin') {
      const leftInterval = viewer.departureIntervals.get(leftText); const rightInterval = viewer.departureIntervals.get(rightText);
      if (leftInterval && rightInterval) return Number(leftInterval.start_seconds) - Number(rightInterval.start_seconds);
    }
    const leftNumber = leftText === '' ? null : Number(leftText); const rightNumber = rightText === '' ? null : Number(rightText);
    if (leftNumber !== null && rightNumber !== null && Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) return leftNumber - rightNumber;
    return leftText.localeCompare(rightText, undefined, {numeric: true, sensitivity: 'base'});
  }
  function sortFlowRows() {
    if (!viewer.flowSortColumn) return;
    const direction = viewer.flowSortDirection === 'asc' ? 1 : -1;
    viewer.flowRows.sort((left, right) => direction * compareFlowValues(left, right, viewer.flowSortColumn));
  }
  function updateFlowLoadControls() {
    const loaded = viewer.flowRows.length; const total = viewer.flowMatchedCount;
    $('flow-more').disabled = loaded === 0 || loaded >= total;
    $('flow-all').disabled = loaded === 0 || loaded >= total;
  }
  function renderTable(rows, title = '', total = rows.length) {
    const target = $('flow-table');
    if (!rows.length) {
      target.querySelector('thead').innerHTML = ''; target.querySelector('tbody').innerHTML = '<tr><td class="muted">No rows for this selection.</td></tr>';
      $('flow-pagination').classList.add('hidden'); $('flow-page-label').textContent = ''; updateFlowLoadControls(); return;
    }
    const headers = Object.keys(rows[0]).filter((header) => header !== 'prior_demand' && rows.some((row) => hasInformativeValue(row[header])));
    if (!headers.length) { target.querySelector('thead').innerHTML = ''; target.querySelector('tbody').innerHTML = '<tr><td class="muted">No populated fields in these rows.</td></tr>'; return; }
    target.querySelector('thead').innerHTML = `<tr>${headers.map((header) => {
      const active = viewer.flowSortColumn === header; const direction = active ? viewer.flowSortDirection : '';
      const marker = direction === 'asc' ? ' ▲' : direction === 'desc' ? ' ▼' : '';
      return `<th aria-sort="${direction === 'asc' ? 'ascending' : direction === 'desc' ? 'descending' : 'none'}"><button type="button" class="flow-sort" data-flow-sort="${esc(header)}">${esc(flowColumnLabel(header))}${marker}</button></th>`;
    }).join('')}</tr>`;
    const pageSize = 50; const pageCount = Math.max(1, Math.ceil(rows.length / pageSize));
    viewer.flowPage = Math.min(viewer.flowPage, pageCount - 1);
    const start = viewer.flowPage * pageSize; const pageRows = rows.slice(start, start + pageSize);
    target.querySelector('tbody').innerHTML = pageRows.map((row) => `<tr>${headers.map((header) => `<td>${esc(header === 'departure_time_bin' && hasInformativeValue(row[header]) ? departureIntervalLabel(row[header]) : row[header])}</td>`).join('')}</tr>`).join('');
    $('flow-detail').innerHTML = `<dl><dt>View</dt><dd>${esc(title)}</dd><dt>Rows loaded</dt><dd>${display(rows.length)} of ${display(total)} matching rows</dd><dt>Table page</dt><dd>${display(start + 1)}–${display(Math.min(start + pageSize, rows.length))} of ${display(rows.length)} loaded</dd><dt>Sort</dt><dd>${viewer.flowSortColumn ? `${esc(flowColumnLabel(viewer.flowSortColumn))} (${viewer.flowSortDirection}); sorted rows currently loaded` : 'Select a column heading to sort loaded rows'}</dd></dl>`;
    $('flow-pagination').classList.toggle('hidden', rows.length <= pageSize);
    $('flow-page-label').textContent = `Page ${viewer.flowPage + 1} of ${pageCount}`;
    $('flow-page-previous').disabled = viewer.flowPage === 0;
    $('flow-page-next').disabled = viewer.flowPage + 1 >= pageCount;
    target.querySelectorAll('button[data-flow-sort]').forEach((button) => button.addEventListener('click', () => {
      const column = button.dataset.flowSort;
      viewer.flowSortDirection = viewer.flowSortColumn === column && viewer.flowSortDirection === 'asc' ? 'desc' : 'asc';
      viewer.flowSortColumn = column; viewer.flowPage = 0; sortFlowRows(); renderTable(viewer.flowRows, viewer.flowTitle, viewer.flowMatchedCount);
    }));
    drawFlowMap(pageRows, title); updateFlowLoadControls();
  }
  function options(select, values, placeholder = 'All') { select.innerHTML = `<option value="">${esc(placeholder)}</option>${[...new Set(values.filter(Boolean))].sort().map((value) => `<option value="${esc(value)}">${esc(value)}</option>`).join('')}`; }
  function clockTime(seconds) {
    const value = Number(seconds);
    if (!Number.isSafeInteger(value) || value < 0) return null;
    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const remainder = value % 60;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}${remainder ? `:${String(remainder).padStart(2, '0')}` : ''}`;
  }
  function departureIntervalLabel(intervalId) {
    const interval = viewer.departureIntervals.get(String(intervalId));
    if (!interval) return `${intervalId} (time range unavailable)`;
    const start = clockTime(interval.start_seconds); const end = clockTime(interval.end_seconds);
    return start && end ? `${start}–${end}` : `${intervalId} (invalid time range)`;
  }
  function measurementPeriodLabel(intervalId) {
    const interval = viewer.departureIntervals.get(String(intervalId ?? ''));
    if (!interval) return 'Time range unavailable';
    const start = clockTime(interval.start_seconds); const end = clockTime(interval.end_seconds);
    return start && end ? `${start}–${end}` : 'Time range unavailable';
  }
  function measurementTimeLabel(value) {
    const interval = viewer.departureIntervals.get(String(value ?? ''));
    return interval ? measurementPeriodLabel(value) : String(value ?? '');
  }
  function intervalOptions(select, values, placeholder) {
    const intervals = canonicalBins(values);
    select.replaceChildren(new Option(placeholder, ''));
    intervals.forEach((intervalId) => {
      const option = new Option(departureIntervalLabel(intervalId), intervalId);
      option.title = String(intervalId);
      select.add(option);
    });
  }
  async function loadDepartureIntervalMetadata() {
    viewer.departureIntervals = new Map();
    viewer.departureTimeZone = String(viewer.manifest?.time_zone || '');
    viewer.departureIntervalMetadataError = '';
    const bundled = viewer.summary.od?.departure_intervals;
    if (Array.isArray(bundled)) {
      bundled.forEach((interval) => {
        if (interval && typeof interval.interval_id === 'string') viewer.departureIntervals.set(interval.interval_id, interval);
      });
    }
    const requestedBins = [...new Set([...(viewer.summary.od?.bins || [])])].map(String);
    const measurementLocationsAvailable = viewer.summary.measurement_locations?.available === true;
    if (requestedBins.length && requestedBins.every((intervalId) => viewer.departureIntervals.has(intervalId)) && !measurementLocationsAvailable) return;
    const querySummary = viewer.summary.measurement_od_contributions || {};
    if (!querySummary.available || querySummary.query_mode !== 'local_service') return;
    try {
      const base = new URL($('measurement-query-service').value.trim());
      if (base.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname)) {
        throw new Error('The local query service URL must use loopback HTTP.');
      }
      const response = await fetch(new URL('/api/departure-intervals', base.origin), {cache: 'no-store'});
      let payload;
      try { payload = await response.json(); }
      catch (_) { throw new Error(`Local query service returned HTTP ${response.status} without valid interval metadata.`); }
      if (!response.ok) throw new Error(payload.error || `Local query service returned HTTP ${response.status}.`);
      if (payload.fit_fingerprint !== querySummary.fit_fingerprint || payload.mapping_artifact_fingerprint !== querySummary.mapping_artifact_fingerprint || !Array.isArray(payload.intervals)) {
        throw new Error('Local interval metadata belongs to a different fit or mapping artifact.');
      }
      payload.intervals.forEach((interval) => {
        if (interval && typeof interval.interval_id === 'string') viewer.departureIntervals.set(interval.interval_id, interval);
      });
      if (payload.time_zone) viewer.departureTimeZone = String(payload.time_zone);
    } catch (error) {
      viewer.departureIntervalMetadataError = error.message;
    }
  }
  function updateIntervalTimeNotes() {
    const values = [...new Set([...(viewer.summary.od?.bins || [])])];
    const hasAllRanges = values.every((intervalId) => viewer.departureIntervals.has(String(intervalId)));
    const timeZone = viewer.departureTimeZone ? ` (${viewer.departureTimeZone})` : '';
    const note = hasAllRanges
      ? `Local schedule time${timeZone}; interval end is exclusive.`
      : `Some interval ranges are unavailable; internal slot IDs are shown. ${viewer.departureIntervalMetadataError || 'Use the combined local query service to load exact times.'}`;
    ['flow-bin-time-note', 'measurement-bin-time-note'].forEach((id) => {
      const target = $(id);
      if (target) target.textContent = note;
    });
  }
  function canonicalBins(values) {
    return [...new Set(values || [])].sort((a, b) => {
      const ai = Number(String(a).match(/\d+/)?.[0]); const bi = Number(String(b).match(/\d+/)?.[0]);
      if (Number.isFinite(ai) && Number.isFinite(bi) && ai !== bi) return ai - bi;
      return String(a).localeCompare(String(b));
    });
  }
  function defaultDepartureInterval(values) {
    const intervals = canonicalBins(values).map(String);
    // Bundle producers can set this generic preference without baking a case's
    // timetable into the reusable viewer. Invalid or absent preferences fall back
    // to the first interval available in the bundle.
    const preferred = String(viewer.summary.od?.default_departure_interval_id || '');
    return intervals.includes(preferred) ? preferred : intervals[0] || '';
  }
  function populateFlowSelectors() {
    const od = viewer.summary.od || {};
    intervalOptions($('flow-bin'), od.bins || [], 'Select departure interval');
    options($('flow-origin'), od.origins || [], 'All origins'); options($('flow-destination'), od.destinations || [], 'All destinations');
    $('flow-bin').value = defaultDepartureInterval(od.bins || []);
  }

  function populateMeasurementOdControls() {
    const summary = viewer.summary.measurement_od_contributions || {};
    const locations = viewer.summary.measurement_locations || {};
    const onDemand = summary.available && summary.query_mode === 'local_service';
    const panel = $('measurement-od'); panel.classList.remove('hidden');
    $('measurement-od-direction').disabled = !summary.available;
    $('measurement-od-load').disabled = !summary.available;
    $('measurement-od-limit').disabled = !summary.available;
    $('measurement-query-service-label').classList.toggle('hidden', !onDemand);
    if (!summary.available) {
      $('measurement-od-unavailable').textContent = summary.message || 'OD-by-measurement contributions are unavailable in this bundle.';
      $('measurement-od-unavailable').classList.remove('hidden');
    } else {
      $('measurement-od-unavailable').classList.add('hidden');
    }
    const bins = canonicalBins((viewer.summary.od || {}).bins || []);
    intervalOptions($('measurement-bin'), bins, 'Select departure interval');
    options($('measurement-origin'), (viewer.summary.od || {}).origins || [], 'Select origin');
    options($('measurement-destination'), (viewer.summary.od || {}).destinations || [], 'Select destination');
    $('measurement-bin').value = defaultDepartureInterval(bins);
    $('measurement-row-select').disabled = !locations.available;
    $('measurement-row-previous').disabled = !locations.available;
    $('measurement-row-next').disabled = !locations.available;
    $('map-measurement-mode').disabled = !locations.available;
    if (!locations.available) {
      const message = locations.message || 'Map-based measurement selection is unavailable in this bundle.';
      $('measurement-stop-label').textContent = message;
      $('map-click-status').textContent = message;
    }
    if (onDemand) {
      $('measurement-od-unavailable').textContent = 'Contributions are computed only for the selected measurement row by the local query service. Start the service documented in the viewer guide; no full OD-by-measurement expansion is stored in this bundle.';
      $('measurement-od-unavailable').classList.remove('hidden');
    }
    updateMeasurementOdDirection();
  }

  function updateMeasurementOdDirection() {
    const byMeasurement = $('measurement-od-direction').value === 'measurement';
    $('measurement-mode-measurement').classList.toggle('hidden', !byMeasurement);
    $('measurement-mode-od').classList.toggle('hidden', byMeasurement);
    $('measurement-od').dataset.queryMode = byMeasurement ? 'measurement' : 'od';
    $('measurement-od-load').textContent = byMeasurement
      ? 'Load OD contributors'
      : 'Load affected measurements';
    $('measurement-contribution-previous').disabled = byMeasurement ? !viewer.selectedMeasurement : true;
    $('measurement-contribution-next').disabled = byMeasurement ? !viewer.selectedMeasurement : true;
  }

  function setMapClickMode(mode) {
    viewer.mapClickMode = mode;
    if (mode === 'measurement') viewer.map?.closePopup();
    $('map-od-mode').classList.toggle('active', mode === 'od');
    $('map-measurement-mode').classList.toggle('active', mode === 'measurement');
    if (mode === 'measurement') {
      $('map-click-status').textContent = 'Hover to see the stop ID and name. Click a stop to inspect measurement rows matched to that exact stop ID.';
    } else {
      updateOdMapSelectionStatus('Click a stop to choose an endpoint.');
    }
    if (mode === 'measurement') {
      const unmatched = viewer.measurementLocationManifest?.unmatched_stop_ids || [];
      if (unmatched.length) $('map-click-status').textContent += ` ${unmatched.length} measurement stop ID(s) have no exact network stop match; examples: ${unmatched.slice(0, 5).join(', ')}.`;
      const missingIds = Number(viewer.measurementLocationManifest?.missing_stop_rows) || 0;
      if (missingIds) $('map-click-status').textContent += ` ${missingIds} measurement row(s) have no stop ID.`;
    }
  }

  function formatSelectedStop(stopId) {
    if (!stopId) return 'not set';
    const stop = viewer.stopsById.get(String(stopId));
    const name = String(stop?.name || '').trim();
    return name ? `${name} (${stopId})` : String(stopId);
  }

  function updateOdMapSelectionStatus(message = 'OD selection') {
    if (viewer.mapClickMode !== 'od') return;
    const origin = formatSelectedStop($('flow-origin').value);
    const destination = formatSelectedStop($('flow-destination').value);
    $('map-click-status').textContent = `${message} Current endpoints — Origin: ${origin}; Destination: ${destination}.`;
  }

  function stopMarkerStyle(stopId, hasMeasurements) {
    const originSelected = String($('flow-origin').value || '') === String(stopId);
    const destinationSelected = String($('flow-destination').value || '') === String(stopId);
    const selected = originSelected || destinationSelected;
    const color = originSelected && destinationSelected
      ? '#6b21a8'
      : originSelected ? '#166534' : destinationSelected ? '#c2410c'
        : hasMeasurements ? '#087f8c' : '#235aa6';
    return {
      radius: selected ? 9 : hasMeasurements ? 6 : 4,
      color,
      weight: selected ? 4 : 1.5,
      fillColor: hasMeasurements ? '#40c4cf' : '#235aa6',
      fillOpacity: selected ? 0.9 : 0.75,
    };
  }

  function updateOdEndpointMarkers() {
    for (const [stopId, entry] of viewer.stopMarkers) {
      entry.marker.setStyle(stopMarkerStyle(stopId, entry.hasMeasurements));
    }
  }

  function assignOdEndpoint(endpoint, stopId) {
    const selectorId = endpoint === 'origin' ? 'flow-origin' : 'flow-destination';
    const selector = $(selectorId);
    const available = [...selector.options].some((option) => option.value === String(stopId));
    if (!available) {
      updateOdMapSelectionStatus(
        `Cannot set ${endpoint === 'origin' ? 'Origin' : 'Destination'} to stop ${stopId}: this stop is not an available ${endpoint} in the bundle.`,
      );
      return false;
    }
    selector.value = String(stopId);
    viewer.selectedOd = null;
    updateOdEndpointMarkers();
    updateOdMapSelectionStatus(`${endpoint === 'origin' ? 'Origin' : 'Destination'} set to ${formatSelectedStop(stopId)}.`);
    cancelRoutePathQuery();
    void updateRoutePathBrowser({clearResults: true});
    const origin = $('flow-origin').value;
    const destination = $('flow-destination').value;
    const departureBin = $('flow-bin').value || 'all';
    void refreshFlows().then((result) => {
      if (
        result?.total === 0
        && viewer.mapClickMode === 'od'
        && $('flow-origin').value === origin
        && $('flow-destination').value === destination
        && ($('flow-bin').value || 'all') === departureBin
        && origin
        && destination
      ) {
        updateOdMapSelectionStatus(
          `No OD rows match ${formatSelectedStop(origin)} → ${formatSelectedStop(destination)} in departure interval ${departureIntervalLabel(departureBin)}. The pair may be excluded by the OD universe; no other endpoint was changed.`,
        );
      }
    });
    return true;
  }

  function makeOdEndpointPopup(stop) {
    const content = document.createElement('div');
    content.className = 'od-endpoint-popup';
    const heading = document.createElement('strong');
    heading.textContent = stop.name ? String(stop.name) : 'Unnamed stop';
    const identifier = document.createElement('span');
    identifier.textContent = `Stop ID: ${stop.stop_id}`;
    const actions = document.createElement('div');
    actions.className = 'od-endpoint-popup-actions';
    actions.setAttribute('role', 'group');
    actions.setAttribute('aria-label', `Choose an endpoint for stop ${stop.stop_id}`);
    for (const [endpoint, label] of [['origin', 'Set as origin'], ['destination', 'Set as destination']]) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.setAttribute(
        'aria-label',
        `${label}: ${stop.name ? `${stop.name}, ` : ''}stop ID ${stop.stop_id}`,
      );
      button.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (assignOdEndpoint(endpoint, stop.stop_id)) viewer.map.closePopup();
      });
      actions.append(button);
    }
    content.append(heading, identifier, actions);
    if (window.L?.DomEvent) {
      L.DomEvent.disableClickPropagation(content);
      L.DomEvent.disableScrollPropagation(content);
    }
    return content;
  }

  async function measurementComponents(measurementIndex) {
    const table = (viewer.summary.measurement_od_contributions || {}).component_table || {};
    if (Array.isArray(table.partitions)) {
      const descriptor = table.partitions.find((item) => measurementIndex >= Number(item.start_index) && measurementIndex < Number(item.start_index) + Number(item.row_count));
      if (!descriptor) return {};
      if (!String(descriptor.path || '').startsWith('partitions/measurement_od_contributions/components/') || String(descriptor.path || '').includes('..')) throw new Error('Measurement component partition path is unsafe.');
      const rows = await readDescriptorPage(descriptor, {measurement_index: String(measurementIndex)}, measurementIndex - Number(descriptor.start_index), 1, 'measurement-component');
      return rows[0] || {};
    }
    if (table.path) return (await readCsvFile(table.path, {measurement_index: String(measurementIndex)}, 1))[0] || {};
    return {};
  }

  async function renderSelectedMeasurementRow(row) {
    cancelActiveMeasurementOdRequest();
    viewer.selectedMeasurement = row;
    viewer.measurementContributionPage = 0;
    viewer.measurementOdSortColumn = 'modeled_mean_contribution';
    viewer.measurementOdSortDirection = 'desc';
    $('measurement-row-select').value = String(row.measurement_index);
    $('measurement-row-detail').innerHTML = `<dl><dt>Measurement row</dt><dd>${esc(row.measurement_index)}</dd><dt>Measurement type</dt><dd>${esc(row.measurement_type || '—')}</dd><dt>Line</dt><dd>${esc(row.line || '—')}</dd><dt>Direction</dt><dd>${esc(row.direction || '—')}</dd><dt>Stop</dt><dd>${esc(row.stop || '—')}</dd><dt>Observation time</dt><dd>${esc(measurementTimeLabel(row.observation_time) || '—')}</dd><dt>Observation period</dt><dd>${esc(row.time_period ? measurementPeriodLabel(row.time_period) : '—')}</dd><dt>Observed count</dt><dd>${display(row.observed)}</dd><dt>Modeled mean</dt><dd>${display(row.modeled)}</dd></dl>`;
    if (!(viewer.summary.measurement_od_contributions || {}).available) {
      $('measurement-od-detail').textContent = 'OD contributions are unavailable for this bundle; the measured row values above remain available.';
      renderMeasurementOdRows([]);
      return;
    }
    if ((viewer.summary.measurement_od_contributions || {}).query_mode === 'local_service') {
      $('measurement-od-detail').textContent = $('measurement-od-direction').value === 'measurement'
        ? 'The local query service will calculate this row’s components and ranked OD contributors.'
        : 'Refreshing the reverse OD-to-measurement query for the current selection.';
      await refreshMeasurementOdForCurrentSelection();
      return;
    }
    try {
      const components = await measurementComponents(Number(row.measurement_index));
      const fixedAttribution = (viewer.summary.measurement_od_contributions || {}).fixed_positive_od_attribution_available === false
        ? `<p class="muted">${esc(viewer.summary.measurement_od_contributions.fixed_positive_od_attribution_message || 'Fixed-positive demand is aggregated in the mapping offset and cannot be assigned to individual OD pairs.')}</p>` : '';
      $('measurement-od-detail').innerHTML = `<dl><dt>Measurement row</dt><dd>${esc(row.measurement_index)}</dd><dt>Observed count</dt><dd>${display(row.observed)}</dd><dt>Modeled mean</dt><dd>${display(components.modeled_mean ?? row.modeled)}</dd><dt>Fixed offset</dt><dd>${display(components.fixed_offset_contribution)}</dd><dt>Additive component</dt><dd>${display(components.additive_contribution)}</dd><dt>Mean-floor adjustment</dt><dd>${display(components.mean_floor_adjustment)}</dd></dl>${fixedAttribution}<p class="muted">OD contributors decompose the modeled mean; they are not a decomposition of the observed count.</p>`;
      await refreshMeasurementOdForCurrentSelection();
    } catch (error) {
      $('measurement-od-detail').textContent = `Measurement components unavailable: ${error.message}`;
    }
  }

  async function selectMeasurementStop(stopId) {
    stopMeasurementOdQuery('Select a measured row to inspect its OD decomposition.');
    const selectionId = ++viewer.measurementStopSelectionId;
    viewer.selectedMeasurementStop = stopId;
    viewer.selectedMeasurement = null;
    viewer.measurementRowPage = 0;
    $('measurement-stop-label').textContent = `Measurements at stop ${stopId}`;
    $('measurement-row-select').innerHTML = '';
    $('measurement-row-select').disabled = true;
    $('measurement-row-detail').textContent = 'Loading measurements for this stop…';
    $('measurement-contribution-page').textContent = '';
    $('measurement-od-detail').textContent = 'No measurement selected.';
    renderMeasurementOdRows([]);
    try {
      const {rows, total} = await measurementsAtStop(stopId, 0, viewer.measurementRowsPerPage);
      if (selectionId !== viewer.measurementStopSelectionId) return;
      if (!total) {
        viewer.measurementRows = [];
        $('measurement-row-detail').textContent = 'No measurements at this location.';
        $('measurement-stop-label').textContent = `No measurements at this location: ${stopId}`;
        $('measurement-row-page').textContent = '0 rows';
        $('map-click-status').textContent = `No measurements at this location: ${stopId}`;
        $('measurement-row-previous').disabled = true; $('measurement-row-next').disabled = true;
        return;
      }
      viewer.measurementRows = rows;
      $('measurement-row-select').disabled = false;
      $('measurement-row-select').innerHTML = rows.map((row) => {
        const details = [row.measurement_type, row.line, row.direction, row.observation_time ? `observed ${measurementTimeLabel(row.observation_time)}` : '', row.time_period ? `period ${measurementPeriodLabel(row.time_period)}` : '', row.stop].filter(Boolean).join(' · ');
        return `<option value="${esc(row.measurement_index)}">${esc(row.measurement_index)} — ${esc(details)}</option>`;
      }).join('');
      $('measurement-row-page').textContent = `Rows ${viewer.measurementRowPage * viewer.measurementRowsPerPage + 1}–${viewer.measurementRowPage * viewer.measurementRowsPerPage + rows.length} of ${total}`;
      $('measurement-row-previous').disabled = viewer.measurementRowPage === 0;
      $('measurement-row-next').disabled = (viewer.measurementRowPage + 1) * viewer.measurementRowsPerPage >= total;
      await renderSelectedMeasurementRow(rows[0]);
      $('measurement-od').scrollIntoView({behavior: 'smooth', block: 'start'});
    } catch (error) {
      if (selectionId !== viewer.measurementStopSelectionId) return;
      $('measurement-row-detail').textContent = `Could not read measurements for this stop: ${error.message}`;
      $('map-click-status').textContent = `Measurement lookup failed for ${stopId}: ${error.message}`;
    }
  }

  async function changeMeasurementRowPage(delta) {
    if (!viewer.selectedMeasurementStop) return;
    stopMeasurementOdQuery('Loading measurements for the selected stop…');
    const selectionId = ++viewer.measurementStopSelectionId;
    viewer.measurementRowPage = Math.max(0, viewer.measurementRowPage + delta);
    try {
      const {rows, total} = await measurementsAtStop(viewer.selectedMeasurementStop, viewer.measurementRowPage, viewer.measurementRowsPerPage);
      if (selectionId !== viewer.measurementStopSelectionId) return;
      viewer.measurementRows = rows;
      $('measurement-row-select').innerHTML = rows.map((row) => {
        const details = [row.measurement_type, row.line, row.direction, row.observation_time ? `observed ${measurementTimeLabel(row.observation_time)}` : '', row.time_period ? `period ${measurementPeriodLabel(row.time_period)}` : '', row.stop].filter(Boolean).join(' · ');
        return `<option value="${esc(row.measurement_index)}">${esc(row.measurement_index)} — ${esc(details)}</option>`;
      }).join('');
      $('measurement-row-page').textContent = `Rows ${viewer.measurementRowPage * viewer.measurementRowsPerPage + 1}–${viewer.measurementRowPage * viewer.measurementRowsPerPage + rows.length} of ${total}`;
      $('measurement-row-previous').disabled = viewer.measurementRowPage === 0;
      $('measurement-row-next').disabled = (viewer.measurementRowPage + 1) * viewer.measurementRowsPerPage >= total;
      if (rows.length) await renderSelectedMeasurementRow(rows[0]);
    } catch (error) {
      if (selectionId !== viewer.measurementStopSelectionId) return;
      $('measurement-row-detail').textContent = `Could not read this measurement page: ${error.message}`;
    }
  }

  async function readCsvFile(name, filters = {}, limit = 5000) {
    const file = findFile(name); if (!file) throw new Error(`File is missing: ${name}`);
    return new Promise((resolve, reject) => {
      const worker = new Worker(workerUrl); const collected = [];
      worker.onmessage = (event) => { if (event.data.type === 'rows') collected.push(...event.data.rows); else { worker.terminate(); resolve(collected); } };
      worker.onerror = (event) => { worker.terminate(); reject(new Error(event.message || 'CSV worker failed.')); };
      worker.postMessage({file, filters, limit: Math.max(1, limit), chunkSize: 1024 * 1024});
    });
  }

  async function labelMeasurementOdRows(rows, direction) {
    if (!rows.length) return rows;
    if (direction === 'measurement') {
      const byBin = new Map();
      rows.forEach((row) => { const bin = row.departure_time_bin; if (!byBin.has(bin)) byBin.set(bin, []); byBin.get(bin).push(String(row.od_index)); });
      const odLabels = new Map();
      for (const [bin, ids] of byBin) {
        const partitionRows = await readPartition('od', bin, {od_index: [...new Set(ids)]}, new Set(ids).size);
        partitionRows.forEach((row) => odLabels.set(String(row.od_index), row));
      }
      const measurement = viewer.selectedMeasurement || {};
      return rows.map((row) => ({
        od_index: row.od_index,
        origin_stop_id: odLabels.get(String(row.od_index))?.origin_stop_id || '',
        destination_stop_id: odLabels.get(String(row.od_index))?.destination_stop_id || '',
        departure_time_bin: row.departure_time_bin,
        estimated_demand: odLabels.get(String(row.od_index))?.estimated_demand ?? '',
        measurement_index: row.measurement_index,
        measurement_type: measurement.measurement_type || '',
        line: measurement.line || '',
        direction: measurement.direction || '',
        stop: measurement.stop || '',
        observation_time: measurement.observation_time || '',
        time_period: measurement.time_period || '',
        observed: measurement.observed || '',
        modeled: measurement.modeled || '',
        linear_flow_contribution: row.linear_flow_contribution,
        modeled_mean_contribution: row.modeled_mean_contribution,
      }));
    }
    const indices = [...new Set(rows.map((row) => String(row.measurement_index)))];
    const measurements = await readCsvFile('predicted_measurements.csv', {row_index: indices}, indices.length);
    const byIndex = new Map(measurements.map((row) => [String(row.row_index), row]));
    return rows.map((row) => {
      const measurement = byIndex.get(String(row.measurement_index)) || {};
      return {
        measurement_index: row.measurement_index,
        measurement_type: measurement.measurement_type || '',
        line: measurement.line || '',
        direction: measurement.direction || '',
        stop: measurement.stop || '',
        observation_time: measurement.observation_time || '',
        time_period: measurement.time_period || '',
        observed: measurement.observed || '',
        modeled: measurement.modeled || '',
        departure_time_bin: row.departure_time_bin,
        operator_coefficient: row.operator_coefficient,
        linear_flow_contribution: row.linear_flow_contribution,
        modeled_mean_contribution: row.modeled_mean_contribution,
      };
    });
  }

  function measurementOdColumnLabel(column) {
    return ({
      od_index: 'OD index',
      origin_stop_id: 'Origin',
      destination_stop_id: 'Destination',
      departure_time_bin: 'Departure interval',
      estimated_demand: 'Estimated demand',
      modeled_mean_contribution: 'Modeled flow contribution',
      linear_flow_contribution: 'Linear flow contribution',
      operator_coefficient: 'Mapping coefficient',
      measurement_index: 'Measurement row',
      measurement_type: 'Measurement type',
      observation_time: 'Observation time',
      time_period: 'Observation interval',
      observed: 'Observed count',
      modeled: 'Modeled count',
    })[column] || column.replace(/_/g, ' ');
  }

  function compareMeasurementOdValues(left, right, column) {
    const leftText = String(left[column] ?? '').trim();
    const rightText = String(right[column] ?? '').trim();
    if (!leftText && rightText) return 1;
    if (leftText && !rightText) return -1;
    if (column === 'departure_time_bin') {
      const leftInterval = viewer.departureIntervals.get(leftText);
      const rightInterval = viewer.departureIntervals.get(rightText);
      if (leftInterval && rightInterval) {
        return Number(leftInterval.start_seconds) - Number(rightInterval.start_seconds);
      }
    }
    const leftNumber = leftText === '' ? null : Number(leftText);
    const rightNumber = rightText === '' ? null : Number(rightText);
    if (leftNumber !== null && rightNumber !== null && Number.isFinite(leftNumber) && Number.isFinite(rightNumber)) {
      return leftNumber - rightNumber;
    }
    return leftText.localeCompare(rightText, undefined, {numeric: true, sensitivity: 'base'});
  }

  function renderMeasurementOdRows(rows) {
    const target = $('measurement-od-table');
    if (!rows.length) {
      viewer.measurementOdRows = [];
      target.querySelector('thead').innerHTML = '';
      target.querySelector('tbody').innerHTML = '<tr><td class="muted">No OD contribution entries match this selection.</td></tr>';
      return;
    }
    if (rows !== viewer.measurementOdRows) viewer.measurementOdRows = rows.slice();
    const direction = viewer.measurementOdSortDirection === 'asc' ? 1 : -1;
    const sortedRows = viewer.measurementOdRows.slice().sort((left, right) => direction * compareMeasurementOdValues(left, right, viewer.measurementOdSortColumn));
    const headers = Object.keys(sortedRows[0]);
    target.querySelector('thead').innerHTML = `<tr>${headers.map((header) => {
      const active = viewer.measurementOdSortColumn === header;
      const sortDirection = active ? viewer.measurementOdSortDirection : '';
      const marker = sortDirection === 'asc' ? ' ▲' : sortDirection === 'desc' ? ' ▼' : '';
      return `<th aria-sort="${sortDirection === 'asc' ? 'ascending' : sortDirection === 'desc' ? 'descending' : 'none'}"><button type="button" class="flow-sort" data-measurement-od-sort="${esc(header)}">${esc(measurementOdColumnLabel(header))}${marker}</button></th>`;
    }).join('')}</tr>`;
    target.querySelector('tbody').innerHTML = sortedRows.map((row) => `<tr>${headers.map((header) => {
      const value = header === 'departure_time_bin' && hasInformativeValue(row[header])
        ? departureIntervalLabel(row[header])
        : row[header];
      return `<td>${esc(value)}</td>`;
    }).join('')}</tr>`).join('');
    target.querySelectorAll('button[data-measurement-od-sort]').forEach((button) => button.addEventListener('click', () => {
      const column = button.dataset.measurementOdSort;
      viewer.measurementOdSortDirection = viewer.measurementOdSortColumn === column && viewer.measurementOdSortDirection === 'asc' ? 'desc' : 'asc';
      viewer.measurementOdSortColumn = column;
      renderMeasurementOdRows(viewer.measurementOdRows);
    }));
  }

  function formatElapsed(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds % 60;
    return minutes ? `${minutes}m ${String(remainder).padStart(2, '0')}s` : `${remainder}s`;
  }

  function cancelActiveMeasurementOdRequest() {
    const requestId = viewer.measurementQueryRequestId;
    const controller = viewer.measurementQueryController;
    viewer.measurementQueryRequestId = '';
    viewer.measurementQueryController = null;
    if (controller) controller.abort();
    if (requestId) cancelLocalQuery(requestId);
  }

  function stopMeasurementOdQuery(message) {
    cancelActiveMeasurementOdRequest();
    viewer.measurementQueryId += 1;
    if (viewer.measurementQueryTimer) clearInterval(viewer.measurementQueryTimer);
    viewer.measurementQueryTimer = null;
    const status = $('measurement-od-status');
    status.classList.remove('is-loading');
    status.classList.remove('bad', 'ok');
    status.setAttribute('aria-busy', 'false');
    const button = $('measurement-od-load');
    button.disabled = false;
    button.textContent = $('measurement-od-direction').value === 'measurement'
      ? 'Load OD contributors'
      : 'Load affected measurements';
    $('measurement-query-progress-wrap').classList.add('hidden');
    if (message) status.textContent = message;
  }

  async function refreshMeasurementOdForCurrentSelection() {
    const direction = $('measurement-od-direction').value;
    if (direction === 'measurement' && !viewer.selectedMeasurement) {
      stopMeasurementOdQuery('Select a measured row to inspect its OD decomposition.');
      renderMeasurementOdRows([]);
      return;
    }
    if (direction === 'od') {
      const bin = $('measurement-bin').value;
      const origin = $('measurement-origin').value;
      const destination = $('measurement-destination').value;
      if (!bin || !origin || !destination) {
        stopMeasurementOdQuery('Choose a departure interval, origin, and destination.');
        $('measurement-od-detail').textContent = 'Choose a departure interval, origin, and destination to query this OD cell.';
        renderMeasurementOdRows([]);
        return;
      }
    }
    await refreshMeasurementOd();
  }

  async function refreshMeasurementOd() {
    const summary = viewer.summary.measurement_od_contributions || {};
    if (!summary.available) return;
    const direction = $('measurement-od-direction').value;
    const limit = Math.max(1, Math.min(500, Number($('measurement-od-limit').value) || 50));
    const localService = summary.available && summary.query_mode === 'local_service';
    if (localService && direction === 'od' && summary.reverse_query_available === false) {
      stopMeasurementOdQuery();
      $('measurement-od-status').textContent = summary.reverse_query_message || 'Reverse OD-to-measurement contributions are unavailable for this fit.';
      $('measurement-od-status').classList.add('bad');
      $('measurement-od-detail').textContent = summary.reverse_query_message || 'Reverse OD-to-measurement contributions are unavailable for this fit.';
      renderMeasurementOdRows([]);
      return;
    }
    const onDemand = direction === 'measurement' && summary.query_mode === 'local_service';
    cancelActiveMeasurementOdRequest();
    const requestId = ++viewer.measurementQueryId;
    const requestToken = globalThis.crypto?.randomUUID
      ? globalThis.crypto.randomUUID()
      : `${Date.now()}-${requestId}-${Math.random().toString(36).slice(2)}`;
    const controller = new AbortController();
    viewer.measurementQueryRequestId = requestToken;
    viewer.measurementQueryController = controller;
    if (viewer.measurementQueryTimer) clearInterval(viewer.measurementQueryTimer);
    const status = $('measurement-od-status');
    const button = $('measurement-od-load');
    const startedAt = Date.now();
    const measurementIndex = direction === 'measurement' ? Number(viewer.selectedMeasurement?.measurement_index) : null;
    const operation = onDemand
      ? `Calculating OD contributions for measurement row ${measurementIndex} from the local mapping`
      : localService ? 'Querying the selected OD cell in the local mapping service'
        : direction === 'measurement' ? 'Loading this measurement’s sparse contributors into the local cache' : 'Reading indexed sparse contribution partitions';
    const progressWrap = $('measurement-query-progress-wrap');
    const progressBar = $('measurement-query-progress-bar');
    const progressLabel = $('measurement-query-progress-label');
    progressWrap.classList.toggle('hidden', !onDemand);
    progressBar.removeAttribute('value');
    progressLabel.textContent = onDemand ? 'Starting local query…' : '';
    renderMeasurementOdRows([]);
    $('measurement-contribution-page').textContent = '';
    const updateLoadingStatus = () => {
      const elapsed = Math.floor((Date.now() - startedAt) / 1000);
      const progressHint = onDemand
        ? 'Live mapping progress is shown below.'
        : elapsed >= 15
          ? 'Still processing the selected contribution data.'
          : 'Please wait while the selected contribution data is processed.';
      status.textContent = `${operation}… Elapsed ${formatElapsed(elapsed)}. ${progressHint}`;
    };
    let progressPollInFlight = false;
    const pollQueryProgress = async () => {
      if (!onDemand || progressPollInFlight || requestId !== viewer.measurementQueryId) return;
      progressPollInFlight = true;
      try {
        const progress = await queryLocalMeasurementProgress(measurementIndex, requestToken, controller.signal);
        if (requestId !== viewer.measurementQueryId) return;
        if (progress.status === 'running' && progress.phase === 'mapping_scan' && Number(progress.total) > 0) {
          const total = Number(progress.total); const completed = Math.min(total, Math.max(0, Number(progress.completed) || 0));
          progressBar.max = total; progressBar.value = completed;
          const percent = Math.floor((completed / total) * 100);
          progressLabel.textContent = `Scanning sparse mapping entries: ${percent}% · ${formatCount(completed)} of ${formatCount(total)} entries.`;
        } else if (progress.status === 'complete') {
          progressBar.max = 1; progressBar.value = 1;
          progressLabel.textContent = progress.message || 'OD contributions are ready.';
        } else {
          progressBar.removeAttribute('value');
          const phaseLabels = {
            starting: 'Starting the query…',
            mapping_scan: progress.message || 'Scanning sparse mapping entries…',
            finalizing: 'Combining OD contributions and ranking contributors…',
            validating: 'Checking results against the saved measurement prediction…',
            waiting: progress.message || 'Waiting for the local query…',
            cancelled: 'Query cancelled because the selection changed.',
            failed: `Query failed: ${progress.message || 'unknown error'}`,
          };
          progressLabel.textContent = phaseLabels[progress.phase] || progress.message || 'Processing the measurement query…';
        }
      } catch (_) {
        if (requestId === viewer.measurementQueryId) {
          progressBar.removeAttribute('value');
          progressLabel.textContent = 'Live progress is temporarily unavailable; the elapsed timer continues.';
        }
      } finally {
        progressPollInFlight = false;
      }
    };
    status.classList.remove('bad', 'ok');
    status.classList.add('is-loading');
    status.setAttribute('aria-busy', 'true');
    button.disabled = true;
    button.textContent = 'Calculating…';
    updateLoadingStatus();
    void pollQueryProgress();
    viewer.measurementQueryTimer = setInterval(() => { updateLoadingStatus(); void pollQueryProgress(); }, 2000);
    try {
      let rows; let selectedOd = null; let total = 0; let queryResult = null;
      if (direction === 'measurement') {
        if (!viewer.selectedMeasurement) throw new Error('Select a measurement row from the map first.');
        queryResult = onDemand
          ? await queryLocalMeasurementPage(measurementIndex, viewer.measurementContributionPage, limit, requestToken, controller.signal)
          : await measurementContributionPage(measurementIndex, viewer.measurementContributionPage, limit);
        rows = queryResult.rows; total = queryResult.total;
        if (onDemand) {
          const components = queryResult.components || {};
          const fixedAttribution = summary.fixed_positive_od_attribution_available === false
            ? `<p class="muted">${esc(summary.fixed_positive_od_attribution_message || 'Fixed-positive demand is aggregated in the mapping offset and cannot be assigned to individual OD pairs.')}</p>` : '';
          $('measurement-od-detail').innerHTML = `<dl><dt>Measurement row</dt><dd>${esc(measurementIndex)}</dd><dt>Observed count</dt><dd>${display(viewer.selectedMeasurement.observed)}</dd><dt>Modeled mean</dt><dd>${display(components.modeled_mean ?? viewer.selectedMeasurement.modeled)}</dd><dt>Fixed offset</dt><dd>${display(components.fixed_offset_contribution)}</dd><dt>Additive component</dt><dd>${display(components.additive_contribution)}</dd><dt>OD component</dt><dd>${display(components.od_contribution)}</dd><dt>Mean-floor adjustment</dt><dd>${display(components.mean_floor_adjustment)}</dd></dl>${fixedAttribution}<p class="muted">OD contributors decompose the modeled mean; they are not a decomposition of the observed count.</p>`;
        }
      } else {
        const bin = $('measurement-bin').value; const origin = $('measurement-origin').value; const destination = $('measurement-destination').value;
        if (!bin || !origin || !destination) throw new Error('Choose a departure interval, origin, and destination.');
        const odRows = await readPartition('od', bin, {origin_stop_id: origin, destination_stop_id: destination}, 100);
        selectedOd = odRows.find((row) => row.departure_time_bin === bin);
        if (!selectedOd) { renderMeasurementOdRows([]); $('measurement-od-status').textContent = 'No fitted OD cell matches that selection.'; return; }
        if (localService) {
          queryResult = await queryLocalOdPage(Number(selectedOd.od_index), limit, requestToken, controller.signal);
          rows = queryResult.rows; total = queryResult.total;
        } else {
          rows = await readPartition('measurement_od_contributions', bin, {od_index: String(selectedOd.od_index)}, limit, {rankBy: 'modeled_mean_contribution', rankAbs: false});
        }
      }
      if (requestId !== viewer.measurementQueryId) return;
      rows = await labelMeasurementOdRows(rows, direction);
      if (requestId !== viewer.measurementQueryId) return;
      renderMeasurementOdRows(rows);
      if (direction === 'measurement') {
        const pageCount = Math.max(1, Math.ceil(total / limit));
        $('measurement-contribution-page').textContent = `Page ${viewer.measurementContributionPage + 1} of ${pageCount} · ${total} OD contributors. Pages start with the largest modeled-mean contributions; column sorting applies to this page.`;
        $('measurement-contribution-previous').disabled = viewer.measurementContributionPage === 0;
        $('measurement-contribution-next').disabled = (viewer.measurementContributionPage + 1) * limit >= total;
      } else {
        $('measurement-od-detail').innerHTML = `<dl><dt>OD pair</dt><dd>${esc(selectedOd.origin_stop_id)} → ${esc(selectedOd.destination_stop_id)} (${esc(selectedOd.departure_time_bin)})</dd><dt>Estimated demand</dt><dd>${display(selectedOd.estimated_demand)}</dd><dt>Measurement rows shown</dt><dd>${esc(rows.length)}${localService ? ` of ${total}` : ''} (ranked by modeled-mean contribution)</dd></dl>`;
      }
      $('measurement-od-status').textContent = direction === 'measurement'
        ? `Loaded ${rows.length} of ${total} OD contributors in ${formatElapsed(Math.floor((Date.now() - startedAt) / 1000))}. Signed modeled-mean and linear-flow contributions are preserved.`
        : `Loaded ${rows.length} bounded contribution rows in ${formatElapsed(Math.floor((Date.now() - startedAt) / 1000))}.`;
      status.classList.add('ok');
      if (onDemand) {
        progressBar.max = 1; progressBar.value = 1;
        progressLabel.textContent = 'OD contributors loaded.';
      }
    } catch (error) {
      if (requestId === viewer.measurementQueryId) {
        const elapsed = formatElapsed(Math.floor((Date.now() - startedAt) / 1000));
        status.textContent = `Could not load OD contributions after ${elapsed}: ${error.message}`;
        status.classList.add('bad');
        if (onDemand) { progressBar.removeAttribute('value'); progressLabel.textContent = `Query failed: ${error.message}`; }
      }
    } finally {
      if (requestId === viewer.measurementQueryId) {
        if (viewer.measurementQueryTimer) clearInterval(viewer.measurementQueryTimer);
        viewer.measurementQueryTimer = null;
        if (viewer.measurementQueryRequestId === requestToken) {
          viewer.measurementQueryRequestId = '';
          viewer.measurementQueryController = null;
        }
        status.classList.remove('is-loading');
        status.setAttribute('aria-busy', 'false');
        button.disabled = false;
        button.textContent = direction === 'measurement'
          ? 'Load OD contributors'
          : 'Load affected measurements';
      }
    }
  }

  async function refreshFlows(limit = 50, preserveSort = false, bufferLimit = Math.max(500, limit)) {
    const bin = $('flow-bin').value || 'all';
    const filters = {};
    if ($('flow-origin').value) filters.origin_stop_id = $('flow-origin').value;
    if ($('flow-destination').value) filters.destination_stop_id = $('flow-destination').value;
    const rankBy = 'estimated_demand';
    if (!preserveSort || !viewer.flowSortColumn) { viewer.flowSortColumn = rankBy; viewer.flowSortDirection = 'desc'; }
    $('flow-more').disabled = true; $('flow-all').disabled = true;
    setFlowStatus(`Scanning estimated OD demand for ${bin}; preparing rows ranked by estimated demand…`, 'is-loading');
    try {
      const result = await readPartition('od', bin, filters, bufferLimit, {rankBy, rankAbs: false, returnDetails: true});
      viewer.flowBufferedRows = result.rows; viewer.flowRows = result.rows.slice(0, limit);
      viewer.flowMatchedCount = result.total; viewer.flowLoadedLimit = viewer.flowRows.length; viewer.flowPage = 0;
      viewer.flowTitle = 'Estimated OD demand';
      viewer.odRows = result.rows;
      sortFlowRows(); renderTable(viewer.flowRows, viewer.flowTitle, result.total);
      setFlowStatus(`Prepared ${display(result.rows.length)} ranked rows; showing ${display(viewer.flowRows.length)} of ${display(result.total)} matches. Use Load 50 more or Load all matching rows to continue.`, 'ok');
      return result;
    } catch (error) { setFlowStatus(`Could not load partition: ${error.message}`, 'bad'); return null; }
  }

  async function loadMoreFlows(loadAll = false) {
    if (!viewer.flowRows.length) return;
    if (loadAll) {
      if (viewer.flowBufferedRows.length < viewer.flowMatchedCount) {
        if (viewer.flowMatchedCount > 5000 && !window.confirm(`Load all ${display(viewer.flowMatchedCount)} matching rows into browser memory? Large results can make the viewer slow or cause the browser tab to run out of memory. Continue?`)) return;
        await refreshFlows(viewer.flowMatchedCount, true, viewer.flowMatchedCount);
        return;
      }
      viewer.flowRows = viewer.flowBufferedRows.slice(0, viewer.flowMatchedCount);
    } else {
      const nextLimit = Math.min(viewer.flowRows.length + 50, viewer.flowMatchedCount);
      if (nextLimit <= viewer.flowRows.length) { updateFlowLoadControls(); return; }
      if (nextLimit <= viewer.flowBufferedRows.length) {
        viewer.flowRows = viewer.flowBufferedRows.slice(0, nextLimit);
      } else {
        const nextBufferLimit = Math.min(viewer.flowMatchedCount, Math.max(nextLimit, viewer.flowBufferedRows.length + 500));
        await refreshFlows(nextLimit, true, nextBufferLimit);
        return;
      }
    }
    viewer.flowLoadedLimit = viewer.flowRows.length; viewer.flowPage = 0;
    sortFlowRows(); renderTable(viewer.flowRows, viewer.flowTitle, viewer.flowMatchedCount);
    setFlowStatus(`Loaded ${display(viewer.flowRows.length)} of ${display(viewer.flowMatchedCount)} matching rows.`, 'ok');
  }

  function routePathServiceBase() {
    const base = new URL($('route-path-service').value.trim());
    if (base.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(base.hostname)) {
      throw new Error('The route-path query service must be HTTP on this computer.');
    }
    return base;
  }

  function cancelRoutePathQuery() {
    if (viewer.routePathController) viewer.routePathController.abort();
    viewer.routePathController = null;
    const requestId = viewer.routePathRequestId;
    viewer.routePathRequestId = '';
    if (!requestId) return;
    try {
      const base = routePathServiceBase();
      const endpoint = new URL('/api/cancel-query', base.origin);
      endpoint.searchParams.set('request_id', requestId);
      void fetch(endpoint, {method: 'POST', cache: 'no-store', keepalive: true}).catch(() => {});
    } catch (_) {
      // The local query service may have been stopped while the viewer was open.
    }
  }

  async function updateRoutePathBrowser({checkService = true, clearResults = false} = {}) {
    const browser = $('route-path-browser');
    if (!browser) return;
    const statusId = ++viewer.routePathStatusId;
    const origin = $('flow-origin').value;
    const destination = $('flow-destination').value;
    const selected = Boolean(origin && destination);
    browser.classList.toggle('hidden', !selected);
    if (!selected) {
      viewer.routePathSourceAvailable = false;
      $('route-path-load').disabled = true;
      $('route-path-status').textContent = 'Select both an origin and destination to browse modeled paths.';
      if (clearResults) { $('route-path-summary').textContent = ''; $('route-path-results').replaceChildren(); }
      cancelRoutePathQuery();
      return;
    }
    if (clearResults) { $('route-path-summary').textContent = ''; $('route-path-results').replaceChildren(); }
    const departureInterval = $('flow-bin').value;
    const hasDepartureInterval = Boolean(departureInterval && departureInterval !== 'all');
    if (!hasDepartureInterval) {
      viewer.routePathSourceAvailable = false;
      $('route-path-load').disabled = true;
      $('route-path-status').textContent = 'Choose one departure-time interval to browse paths.';
      return;
    }
    if (!checkService) {
      $('route-path-load').disabled = !viewer.routePathSourceAvailable;
      return;
    }
    viewer.routePathSourceAvailable = false;
    $('route-path-load').disabled = true;
    $('route-path-status').className = 'status is-loading';
    $('route-path-status').textContent = 'Checking the local route-choice query service…';
    try {
      const base = routePathServiceBase();
      const response = await fetch(new URL('/api/status', base.origin), {cache: 'no-store'});
      let payload;
      try { payload = await response.json(); }
      catch (_) { throw new Error(`Local query service returned HTTP ${response.status} without valid status data.`); }
      if (!response.ok) throw new Error(payload.error || `Local query service returned HTTP ${response.status}.`);
      if (statusId !== viewer.routePathStatusId) return;
      const expectedFit = viewer.manifest.model_fingerprint || viewer.summary.measurement_od_contributions?.fit_fingerprint;
      const expectedMapping = viewer.manifest.artifact_identity_fingerprint || viewer.summary.measurement_od_contributions?.mapping_artifact_fingerprint;
      if (payload.fit_fingerprint !== expectedFit || payload.mapping_artifact_fingerprint !== expectedMapping) {
        throw new Error('The local service is using a different fit or mapping artifact.');
      }
      viewer.routePathSourceAvailable = payload.route_paths_available === true;
      if (!viewer.routePathSourceAvailable) {
        $('route-path-status').className = 'status';
        $('route-path-status').textContent = 'This local service has no compatible prepared route-choice source. It can still serve the other viewer queries.';
        return;
      }
      $('route-path-status').className = 'status ok';
      $('route-path-status').textContent = `Ready for ${formatSelectedStop(origin)} → ${formatSelectedStop(destination)} in departure interval ${departureInterval}.`;
      $('route-path-load').disabled = false;
    } catch (error) {
      if (statusId !== viewer.routePathStatusId) return;
      viewer.routePathSourceAvailable = false;
      $('route-path-status').className = 'status';
      $('route-path-status').textContent = `Route-path service unavailable: ${error.message}`;
    }
  }

  function renderRoutePaths(payload) {
    const percent = (value) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : 'unavailable';
    const nameFor = (stopId) => {
      const stop = viewer.stopsById.get(String(stopId));
      return stop?.name ? `${stop.name} (${stopId})` : String(stopId);
    };
    const paths = Array.isArray(payload.paths) ? payload.paths : [];
    $('route-path-summary').textContent = `Listed probability mass: ${percent(payload.listed_probability_mass)} · Total modeled reach: ${percent(payload.total_path_probability)} · Remaining/unlisted mass: ${percent(payload.unlisted_probability_mass)}${payload.truncated ? ' · Result is bounded and may omit additional paths.' : ''}`;
    $('route-path-results').innerHTML = paths.length ? paths.map((path, index) => {
      const legs = Array.isArray(path.legs) ? path.legs : [];
      const legMarkup = legs.length ? `<ol class="route-path-legs">${legs.map((leg) => {
        const line = leg.line_id ? `Line ${esc(leg.line_id)} · ` : '';
        const trip = leg.trip_id ? ` · trip ${esc(leg.trip_id)}` : '';
        return `<li>${line}${esc(nameFor(leg.from_stop_id))} → ${esc(nameFor(leg.to_stop_id))} · ${esc(leg.departure_time || '—')}–${esc(leg.arrival_time || '—')}${trip}</li>`;
      }).join('')}</ol>` : '<p class="muted small">No ride legs were recorded for this path.</p>';
      const stops = Array.isArray(path.stops) ? path.stops.map(nameFor).join(' → ') : '';
      return `<details><summary>Path ${index + 1} · ${percent(path.probability)} · ${formatCount(path.transfers || 0)} transfers</summary><p class="muted small">${esc(stops)}</p>${legMarkup}</details>`;
    }).join('') : '<p class="muted">No complete modeled path met this probability threshold.</p>';
  }

  async function loadRoutePaths() {
    cancelRoutePathQuery();
    const origin = $('flow-origin').value;
    const destination = $('flow-destination').value;
    const departureInterval = $('flow-bin').value;
    const thresholdPercent = Number($('route-path-threshold').value);
    if (!origin || !destination) { $('route-path-status').textContent = 'Select both an origin and destination.'; return; }
    if (!departureInterval || departureInterval === 'all') { $('route-path-status').textContent = 'Choose one departure-time interval.'; return; }
    if (!Number.isFinite(thresholdPercent) || thresholdPercent <= 0 || thresholdPercent > 100) {
      $('route-path-status').className = 'status bad';
      $('route-path-status').textContent = 'Minimum probability must be greater than 0 and at most 100 percent.';
      return;
    }
    const controller = new AbortController();
    const requestId = globalThis.crypto?.randomUUID ? crypto.randomUUID() : `route-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    viewer.routePathController = controller;
    viewer.routePathRequestId = requestId;
    $('route-path-load').disabled = true;
    $('route-path-summary').textContent = '';
    $('route-path-results').replaceChildren();
    $('route-path-status').className = 'status is-loading';
    $('route-path-status').textContent = 'Calculating bounded modeled paths…';
    try {
      const base = routePathServiceBase();
      const endpoint = new URL('/api/route-paths', base.origin);
      endpoint.searchParams.set('origin_stop_id', origin);
      endpoint.searchParams.set('destination_stop_id', destination);
      endpoint.searchParams.set('departure_interval_id', departureInterval);
      endpoint.searchParams.set('minimum_probability', String(thresholdPercent / 100));
      endpoint.searchParams.set('request_id', requestId);
      const response = await fetch(endpoint, {cache: 'no-store', signal: controller.signal});
      let payload;
      try { payload = await response.json(); }
      catch (_) { throw new Error(`Local query service returned HTTP ${response.status} without valid path data.`); }
      if (!response.ok) throw new Error(payload.error || `Local query service returned HTTP ${response.status}.`);
      const expectedFit = viewer.manifest.model_fingerprint || viewer.summary.measurement_od_contributions?.fit_fingerprint;
      const expectedMapping = viewer.manifest.artifact_identity_fingerprint || viewer.summary.measurement_od_contributions?.mapping_artifact_fingerprint;
      if (payload.fit_fingerprint !== expectedFit || payload.mapping_artifact_fingerprint !== expectedMapping) {
        throw new Error('The returned paths belong to a different fit or mapping artifact.');
      }
      if (payload.origin_stop_id !== origin || payload.destination_stop_id !== destination || payload.departure_interval_id !== departureInterval) {
        throw new Error('The local service returned paths for a different endpoint selection or interval.');
      }
      renderRoutePaths(payload);
      $('route-path-status').className = 'status ok';
      $('route-path-status').textContent = `Found ${formatCount((payload.paths || []).length)} qualifying path${(payload.paths || []).length === 1 ? '' : 's'} at or above ${thresholdPercent}% probability.`;
    } catch (error) {
      if (error.name === 'AbortError') return;
      $('route-path-status').className = 'status bad';
      $('route-path-status').textContent = `Could not calculate modeled paths: ${error.message}`;
    } finally {
      if (viewer.routePathRequestId === requestId) {
        viewer.routePathController = null;
        viewer.routePathRequestId = '';
        $('route-path-load').disabled = !viewer.routePathSourceAvailable;
      }
    }
  }

  // Kept as a named action for integrations that select a result row.
  function selectOd(row) { viewer.selectedOd = row; }

  function drawFlowMap(rows, title) {
    if (!viewer.map || !window.L) return;
    if (!viewer.flowLayer) viewer.flowLayer = L.layerGroup().addTo(viewer.map); else viewer.flowLayer.clearLayers();
    rows.slice(0, 100).forEach((row) => {
      const originId = title === 'Estimated OD demand' ? row.origin_stop_id : row.boarding_stop_id || row.origin_stop_id;
      const destinationId = title === 'Estimated OD demand' ? row.destination_stop_id : row.alighting_stop_id || row.destination_stop_id;
      const origin = viewer.stopsById.get(originId); const destination = viewer.stopsById.get(destinationId);
      if (!origin || !destination) return;
      const a = [num(origin.lat), num(origin.lon)]; const b = [num(destination.lat), num(destination.lon)];
      if (a.some((value) => value === null) || b.some((value) => value === null)) return;
      L.polyline([a, b], {color: title === 'Estimated OD demand' ? '#235aa6' : '#d14b4b', weight: 2, opacity: .45}).bindTooltip(`${esc(originId)} → ${esc(destinationId)}`).addTo(viewer.flowLayer);
    });
  }

  async function waitForLeaflet() { for (let i = 0; i < 100 && !window.L; i += 1) await new Promise((resolve) => setTimeout(resolve, 50)); if (!window.L) throw new Error('Map library could not be loaded.'); }
  async function renderNetwork() {
    viewer.stopMarkers = new Map();
    const stopsFile = findFile('network/stops.csv') || findFile('stops.csv'); if (!stopsFile) return;
    try {
      const locationManifest = await measurementLocationIndex();
      const unmatched = locationManifest?.unmatched_stop_ids || [];
      const missingIds = Number(locationManifest?.missing_stop_rows) || 0;
      if (unmatched.length || missingIds) {
        const messages = [];
        if (unmatched.length) messages.push(`${unmatched.length} measurement stop ID(s) do not exactly match network/stops.csv and cannot be selected on the map. Examples: ${unmatched.slice(0, 20).join(', ')}.`);
        if (missingIds) messages.push(`${missingIds} measurement row(s) have a blank stop ID and are not selectable from the map.`);
        $('measurement-unmatched-notice').textContent = messages.join(' ');
        $('measurement-unmatched-notice').classList.remove('hidden');
      }
    } catch (error) {
      $('measurement-unmatched-notice').textContent = `Measurement-location index could not be validated: ${error.message}`;
      $('measurement-unmatched-notice').classList.remove('hidden');
      $('map-measurement-mode').disabled = true;
    }
    viewer.stops = await new Promise((resolve, reject) => {
      const worker = new Worker(workerUrl); const rows = [];
      worker.onmessage = (event) => { if (event.data.type === 'rows') rows.push(...event.data.rows); else { worker.terminate(); resolve(rows); } };
      worker.onerror = () => { worker.terminate(); reject(new Error('network CSV worker failed')); };
      worker.postMessage({file: stopsFile, filters: {}, limit: 20000, chunkSize: 1024 * 1024});
    });
    viewer.stopsById = new Map(viewer.stops.map((stop) => [stop.stop_id, stop]));
    try { await waitForLeaflet(); } catch (error) { $('network-map').classList.remove('hidden'); $('map').textContent = `${error.message} Tables remain available without map tiles.`; return; }
    const points = viewer.stops.map((stop) => [num(stop.lat), num(stop.lon)]).filter(([lat, lon]) => lat !== null && lon !== null); if (!points.length) return;
    $('network-map').classList.remove('hidden'); viewer.map = L.map('map'); L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}).addTo(viewer.map); viewer.map.fitBounds(L.latLngBounds(points).pad(0.08)); viewer.markerLayer = L.layerGroup().addTo(viewer.map); viewer.flowLayer = L.layerGroup().addTo(viewer.map);
    const indexedStops = viewer.measurementLocationManifest?.stops || {};
    viewer.stops.forEach((stop) => {
      const lat = num(stop.lat); const lon = num(stop.lon); if (lat === null || lon === null) return;
      const hasMeasurements = Object.prototype.hasOwnProperty.call(indexedStops, stop.stop_id);
      const rowCount = hasMeasurements ? Number(indexedStops[stop.stop_id].row_count) || 0 : 0;
      const marker = L.circleMarker([lat, lon], stopMarkerStyle(stop.stop_id, hasMeasurements));
      marker.bindTooltip(`Stop ID: ${esc(stop.stop_id)}${stop.name ? ` · ${esc(stop.name)}` : ''}${hasMeasurements ? ` · ${rowCount} measurement row(s)` : ''}`)
        .on('click', (event) => {
          if (viewer.mapClickMode === 'measurement') {
            void selectMeasurementStop(stop.stop_id);
            return;
          }
          if (event.originalEvent) L.DomEvent.stopPropagation(event.originalEvent);
          updateOdMapSelectionStatus(`Choose an endpoint for ${formatSelectedStop(stop.stop_id)}.`);
          L.popup()
            .setLatLng(marker.getLatLng())
            .setContent(makeOdEndpointPopup(stop))
            .openOn(viewer.map);
        })
        .addTo(viewer.markerLayer);
      viewer.stopMarkers.set(stop.stop_id, {marker, hasMeasurements});
    });
    updateOdEndpointMarkers();
    updateOdMapSelectionStatus('Click a stop to choose an endpoint.');
    await renderLineGeometry();
  }

  async function renderLineGeometry() {
    const geometry = viewer.summary.line_geometry || {};
    if (!geometry.available || !geometry.geometry_path || !viewer.map || !window.L) return;
    const file = findFile(geometry.geometry_path);
    if (!file) return;
    try {
      const geojson = JSON.parse(await file.text());
      L.geoJSON(geojson, {
        style: {color: '#6a4fb3', weight: 4, opacity: .75},
        interactive: false,
      }).addTo(viewer.map);
    } catch (error) {
      setFlowStatus(`Line geometry unavailable: ${error.message}`, '');
    }
  }

  async function validateStartup(manifest) {
    const problems = [];
    for (const name of REQUIRED) if (!findFile(name)) problems.push(`missing required file: ${name}`);
    for (const [name, descriptor] of Object.entries(manifest.files || {})) { const file = findFile(name); if (!file) { problems.push(`missing manifest file: ${name}`); continue; } const size = descriptor.size_bytes ?? descriptor.size; if (size !== undefined && file.size !== size) problems.push(`size mismatch: ${name}`); }
    return problems;
  }
  async function load(files) {
    try {
      viewer.files = files; viewer.fileMap = new Map(files.map((file) => [file.webkitRelativePath || file.name, file])); setStatus('Reading manifest and small summary; large tables remain unopened…');
      viewer.manifest = await readJson('bundle_manifest.json'); if (viewer.manifest.bundle_type !== 'gravity_viewer_bundle') throw new Error(`Unexpected bundle type: ${viewer.manifest.bundle_type}`);
      const problems = await validateStartup(viewer.manifest); viewer.model = await readJson('model_specification.json'); viewer.report = await readJson('report.json'); viewer.summary = findFile('viewer_summary.json') ? await readJson('viewer_summary.json') : {};
      if (!viewer.summary.od) {
        const legacyRecord = viewer.manifest.files?.['full_od.csv'] || {};
        viewer.summary = {
          od: {available: true, partition_field: 'legacy_full_table', bins: ['all'], origins: [], destinations: [], partitions: [{key: 'all', path: 'full_od.csv', size_bytes: legacyRecord.size_bytes, sha256: legacyRecord.sha256}]},
          measurement_od_contributions: {available: false, message: 'Legacy bundle has no OD-by-measurement contribution dataset.'},
        };
      }
      if (!viewer.summary.measurement_od_contributions) viewer.summary.measurement_od_contributions = {available: false, message: 'This bundle predates OD-by-measurement contributions.'};
      await loadDepartureIntervalMetadata();
      viewer.cacheIdentity = `${viewer.manifest.model_fingerprint || ''}|${viewer.manifest.specification_fingerprint || ''}|${viewer.manifest.report_fingerprint || ''}|${JSON.stringify(viewer.manifest.provenance || {})}`;
      $('summary').classList.remove('hidden'); $('diagnostics').classList.remove('hidden'); $('tables').classList.remove('hidden'); $('flows').classList.remove('hidden');
      const counts = viewer.manifest.row_counts || {}; $('metrics').innerHTML = [metric('Bundle type', viewer.manifest.bundle_type), metric('Schema', viewer.manifest.schema_version), metric('OD rows', counts['full_od.csv'] ?? 'unknown'), metric('Measurements', counts['predicted_measurements.csv'] ?? 'unknown'), metric('OD partitions', viewer.summary.od?.partitions?.length ?? 'legacy'), metric('OD contribution map', viewer.summary.measurement_od_contributions?.available ? 'available' : 'absent'), metric('Identifiability', viewer.manifest.identifiability?.available ? 'available' : 'not available'), metric('Network files', Object.keys(viewer.manifest.files || {}).filter((key) => key.startsWith('network/')).length)].join('');
      $('model').innerHTML = renderModelSpecification(); $('provenance').innerHTML = renderRunDetails(); $('bundle-files').innerHTML = renderBundleFiles(); $('bundle-path-note').textContent = 'Included-file paths are relative to the selected bundle. Original source paths appear only if recorded in bundle metadata; browser security hides the laptop’s absolute folder path.'; const adequacy = viewer.report.adequacy || {}; $('diagnostic-metrics').innerHTML = [metric('Observed total', adequacy.observed_total), metric('Modeled total', adequacy.modeled_total), metric('MAE', adequacy.mae), metric('RMSE', adequacy.rmse), metric('Report fingerprint', viewer.report.report_fingerprint, 'fingerprint')].join('');
      $('table-content').innerHTML = '<p class="muted">Large tables are available through the partitioned explorer. No complete CSV is parsed at startup.</p>';
      const executive = findFile('executive_summary.md'); $('executive').innerHTML = executive ? renderMarkdown(await executive.text()) : '<p>No executive summary supplied.</p>'; populateFlowSelectors(); populateMeasurementOdControls(); updateIntervalTimeNotes(); await renderNetwork(); await refreshFlows(); await updateRoutePathBrowser({clearResults: true});
      if (!viewer.summary.od) { $('interactive-notice').textContent = 'This is a legacy bundle without partition metadata. The validated tables remain available; lazy time-slot exploration requires a refreshed export.'; $('interactive-notice').classList.remove('hidden'); }
      if (problems.length) setStatus(`Bundle loaded with startup validation warnings:\n- ${problems.join('\n- ')}`, 'bad'); else setStatus('Bundle manifest and summary loaded. Partition checksums are verified when selected.', 'ok');
    } catch (error) { $('summary').classList.add('hidden'); $('diagnostics').classList.add('hidden'); $('flows').classList.add('hidden'); setStatus(`Could not load bundle: ${error.message}`, 'bad'); }
  }
  function metric(label, value, extra = '') { return `<div class="metric"><label>${esc(label)}</label><strong class="${extra}">${esc(value)}</strong></div>`; }
  function humanize(value) { return String(value ?? '').replace(/^model(\d+)[_-]/i, 'Model $1: ').replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim().replace(/\b[a-z]/g, (letter) => letter.toUpperCase()); }
  function displayValue(value) { if (value === null || value === undefined || value === '') return 'Not recorded'; if (typeof value === 'boolean') return value ? 'Yes' : 'No'; return String(value); }
  function formatCount(value) { const number = Number(value); return Number.isFinite(number) ? new Intl.NumberFormat().format(number) : 'Not recorded'; }
  function formatBytes(value) { const bytes = Number(value); if (!Number.isFinite(bytes) || bytes < 0) return 'size unavailable'; if (bytes < 1024) return `${formatCount(bytes)} B`; const units = ['KB', 'MB', 'GB', 'TB']; let amount = bytes / 1024; let unit = units[0]; for (let index = 0; amount >= 1024 && index < units.length - 1; index += 1) { amount /= 1024; unit = units[index + 1]; } return `${new Intl.NumberFormat(undefined, {maximumFractionDigits: 1}).format(amount)} ${unit}`; }
  function formatDate(value) { if (!value) return 'Not recorded'; const date = new Date(value); return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString(undefined, {dateStyle: 'medium', timeStyle: 'short'}); }
  function renderDefinitionList(entries) { const rows = entries.filter(([, value]) => value !== undefined && value !== null && value !== '').map(([label, value]) => `<dt>${esc(label)}</dt><dd>${esc(displayValue(value))}</dd>`); return rows.length ? `<dl class="summary-list">${rows.join('')}</dl>` : '<p>No readable details were recorded.</p>'; }
  function readableEffects(specification) {
    const entries = [...(Array.isArray(specification.components) ? specification.components.map((item) => ({...item, kind: 'Component'})) : []), ...(Array.isArray(specification.terms) ? specification.terms.map((item) => ({...item, kind: 'Term'})) : [])];
    if (!entries.length && specification.legacy && typeof specification.legacy === 'object') for (const [key, value] of Object.entries(specification.legacy)) if (key.endsWith('_scope') && value && value !== 'none') entries.push({name: key.replace(/_scope$/, ''), scope: value, kind: 'Effect'});
    if (!entries.length) return '<p>The detailed structure is in <code>model_specification.json</code>.</p>';
    return `<ul class="effect-list">${entries.map((item) => { const details = [item.target ? humanize(item.target) : '', item.scope ? `${humanize(item.scope)} effect` : '', item.source ? `source: ${humanize(item.source)}` : '', item.grouping ? `grouped by ${humanize(item.grouping)}` : '', item.parameterization ? `${humanize(item.parameterization)} parameterization` : '', item.deviation?.scope ? `${humanize(item.deviation.scope)} deviations` : ''].filter(Boolean).join(' · '); return `<li><strong>${esc(humanize(item.name || item.kind))}</strong><span class="muted small">${esc(details || item.kind)}</span></li>`; }).join('')}</ul>`;
  }
  function renderModelSpecification() {
    const specification = viewer.model.specification && typeof viewer.model.specification === 'object' ? viewer.model.specification : viewer.model;
    const likelihoodValue = specification.likelihood ?? viewer.model.likelihood;
    const likelihood = likelihoodValue && typeof likelihoodValue === 'object' ? likelihoodValue.family || likelihoodValue.name : likelihoodValue;
    const entries = [['Model', humanize(specification.model_name || specification.preset || 'Gravity model')], ['Preset', specification.preset ? humanize(specification.preset) : null], ['Likelihood', likelihood ? humanize(likelihood) : null], ['Production input', specification.production_source ? humanize(specification.production_source) : null], ['Destination-attractiveness input', specification.destination_attractiveness_source ? humanize(specification.destination_attractiveness_source) : null]];
    const time = specification.time && typeof specification.time === 'object' ? Object.entries(specification.time).filter(([, value]) => ['string', 'number', 'boolean'].includes(typeof value) && value !== '').map(([key, value]) => [humanize(key), humanize(value)]) : [];
    return `${renderDefinitionList([...entries, ...time])}<h4>Included model effects</h4>${readableEffects(specification)}<p class="small muted">Full specification: <code>${esc(viewer.manifest.model_specification?.path || 'model_specification.json')}</code>.</p>`;
  }
  function selectedBundleName() { const manifestPath = [...viewer.fileMap.keys()].find((path) => path === 'bundle_manifest.json' || path.endsWith('/bundle_manifest.json')); const parent = manifestPath ? manifestPath.slice(0, -'bundle_manifest.json'.length).replace(/\/$/, '') : ''; return parent ? parent.split('/').pop() : 'Selected viewer bundle'; }
  function collectTechnicalIdentifiers() {
    const identifiers = new Map();
    const visit = (value, prefix = '') => { if (!value || typeof value !== 'object') return; for (const [key, item] of Object.entries(value)) { const label = prefix ? `${prefix}.${key}` : key; if (item && typeof item === 'object') visit(item, label); else if (/(fingerprint|sha256|checksum|revision|commit)/i.test(key) && item !== null && item !== undefined && item !== '') identifiers.set(label, String(item)); } };
    visit(viewer.manifest.provenance || {}); visit(viewer.manifest.fit_provenance || {}); visit(viewer.manifest.validation_provenance || {}); visit(viewer.manifest.report_provenance || {}); visit(viewer.report.provenance || {});
    for (const [key, value] of Object.entries(viewer.manifest)) if (/(fingerprint|sha256|checksum|revision|commit)/i.test(key) && value !== null && value !== undefined && typeof value !== 'object') identifiers.set(key, String(value));
    if (viewer.model.specification_fingerprint) identifiers.set('model_specification_fingerprint', String(viewer.model.specification_fingerprint));
    return [...identifiers.entries()].sort(([left], [right]) => left.localeCompare(right));
  }
  function renderRunDetails() {
    const fit = viewer.manifest.fit_provenance || {}; const metadata = viewer.manifest.metadata && typeof viewer.manifest.metadata === 'object' ? viewer.manifest.metadata : {};
    const entries = [['Selected bundle', selectedBundleName()], ['Bundle created', formatDate(viewer.manifest.created_at_utc)], ['Fit outcome', fit.status ? humanize(fit.status) : null], ['Optimizer', fit.optimizer ? humanize(fit.optimizer) : null], ['Fit acceptance', fit.acceptance], ['Time zone', viewer.manifest.time_zone], ...Object.entries(metadata).filter(([key, value]) => !/(fingerprint|sha256|checksum|revision)$/i.test(key) && ['string', 'number', 'boolean'].includes(typeof value) && value !== '').map(([key, value]) => [humanize(key), value])];
    const technical = collectTechnicalIdentifiers(); const details = technical.length ? `<details class="technical-details"><summary>Technical fingerprints and version identifiers (${formatCount(technical.length)})</summary><div class="fingerprint">${technical.map(([key, value]) => `<div><b>${esc(humanize(key))}:</b> ${esc(value)}</div>`).join('')}</div></details>` : '';
    return `${renderDefinitionList(entries)}${details}`;
  }
  function bundleFileDescription(path) {
    const base = fileName(path); const descriptions = {'bundle_manifest.json': 'Bundle index, run details, and file checksums', 'viewer_summary.json': 'Index of time partitions and on-demand datasets', 'model_specification.json': 'Model inputs, likelihood, and modeled effects', 'report.json': 'Fit diagnostics and aggregate performance measures', 'executive_summary.md': 'Human-readable report summary', 'full_od.csv': 'Estimated demand by origin, destination, and departure interval', 'predicted_measurements.csv': 'Observed counts and model-predicted counts', 'residuals.csv': 'Measurement-level observed-minus-modeled differences', 'grouped_residuals.csv': 'Residual diagnostics aggregated by group', 'parameters.csv': 'Estimated model parameters', 'identifiability.csv': 'Parameter identifiability classifications', 'od_map.csv': 'Map-ready OD demand summaries', 'stop_summary.csv': 'Observed and modeled values summarized by stop', 'measurement_location_manifest.json': 'Index linking map stops to measurement rows', 'report.html': 'Formatted detailed report'};
    if (path.startsWith('network/')) return 'Network snapshot used for map display'; if (path.startsWith('partitions/od/')) return 'Estimated OD demand for one time partition'; if (path.startsWith('partitions/measurement_od_contributions/')) return 'OD contributions to measured rows'; if (path.startsWith('partitions/measurement_locations/')) return 'Map-stop measurement lookup data'; if (path.startsWith('partitions/')) return 'Optional partitioned data included in the bundle'; if (path.startsWith('identifiability/')) return 'Parameter identifiability report data'; return descriptions[base] || 'Supporting data included in the viewer bundle';
  }
  function bundleFileCategory(path) {
    if (path === 'bundle_manifest.json' || !path.includes('/')) return ['Summary and report files', 'Summary, model, and report']; if (path.startsWith('network/')) return ['Network snapshot', 'Stops, lines, trips, and stop times'];
    if (path.startsWith('partitions/od/')) return ['Estimated OD demand partitions', 'Partitioned demand data'];
    if (path.startsWith('partitions/measurement_od_contributions/')) return ['OD contribution partitions', 'Measured-row decomposition data'];
    if (path.startsWith('partitions/measurement_locations/')) return ['Measurement lookup partitions', 'Map-stop lookup data'];
    if (path.startsWith('partitions/')) return ['Additional partition data', 'Optional partitioned files'];
    if (path.startsWith('identifiability/')) return ['Identifiability diagnostics', 'Parameter diagnostic data']; return [`${humanize(path.split('/')[0])} files`, 'Supporting files'];
  }
  function renderBundleFiles() {
    const records = {...(viewer.manifest.files || {})}; if (!records['bundle_manifest.json']) records['bundle_manifest.json'] = {size_bytes: findFile('bundle_manifest.json')?.size}; const groups = new Map();
    for (const [path, record] of Object.entries(records)) { const [title, subtitle] = bundleFileCategory(path); if (!groups.has(title)) groups.set(title, {subtitle, files: []}); groups.get(title).files.push([path, record || {}]); }
    return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([title, group]) => {
      group.files.sort(([a], [b]) => a.localeCompare(b)); const totalBytes = group.files.reduce((sum, [, record]) => sum + (Number(record.size_bytes ?? record.size) || 0), 0); const totalRows = group.files.reduce((sum, [, record]) => sum + (Number(record.row_count) || 0), 0); const shown = group.files.slice(0, 250);
      const rows = shown.map(([path, record]) => `<li><code>${esc(path)}</code><span class="small muted">${esc(bundleFileDescription(path))}</span><small>${esc(formatBytes(record.size_bytes ?? record.size))}${record.row_count === undefined ? '' : ` · ${esc(formatCount(record.row_count))} rows`}</small></li>`).join(''); const remainder = group.files.length > shown.length ? `<p class="file-limit-note small muted">Showing ${formatCount(shown.length)} of ${formatCount(group.files.length)} files. The complete file index is in <code>bundle_manifest.json</code>.</p>` : ''; const rowNote = totalRows ? ` · ${formatCount(totalRows)} rows total` : '';
      return `<details><summary>${esc(title)} — ${formatCount(group.files.length)} files · ${esc(formatBytes(totalBytes))}${rowNote}</summary><p class="small muted">${esc(group.subtitle)}</p><ul class="bundle-file-list">${rows}</ul>${remainder}</details>`;
    }).join('') || '<p class="muted">No file inventory was recorded.</p>';
  }
  function markdownTableCells(line) { return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((cell) => cell.trim().replace(/\\\|/g, '|')); }
  function isMarkdownTableStart(lines, index) {
    if (index + 1 >= lines.length || !lines[index].includes('|')) return false;
    const header = markdownTableCells(lines[index]); const separators = markdownTableCells(lines[index + 1]);
    return separators.length === header.length && separators.every((cell) => /^:?-{3,}:?$/.test(cell));
  }
  function inlineMarkdown(source) {
    let value = esc(source);
    value = value.replace(/`([^`]+)`/g, '<code>$1</code>');
    value = value.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/\*([^*]+)\*/g, '<em>$1</em>');
    return value.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  }
  function renderMarkdown(source) {
    const lines = String(source ?? '').replace(/\r\n?/g, '\n').split('\n').map((line) => line.replace(/^\\(?=#{1,6}\s|\|)/, '').replace(/\\$/, ''));
    const blocks = []; let index = 0;
    while (index < lines.length) {
      if (!lines[index].trim()) { index += 1; continue; }
      if (/^\s*```/.test(lines[index])) {
        const code = []; index += 1;
        while (index < lines.length && !/^\s*```/.test(lines[index])) code.push(lines[index++]);
        if (index < lines.length) index += 1;
        blocks.push(`<pre><code>${esc(code.join('\n'))}</code></pre>`); continue;
      }
      if (isMarkdownTableStart(lines, index)) {
        const header = markdownTableCells(lines[index]); const separators = markdownTableCells(lines[index + 1]);
        const alignment = separators.map((cell) => cell.startsWith(':') ? (cell.endsWith(':') ? 'center' : 'left') : (cell.endsWith(':') ? 'right' : 'left'));
        const rows = []; index += 2;
        while (index < lines.length && lines[index].trim() && lines[index].includes('|')) rows.push(markdownTableCells(lines[index++]));
        const headerHtml = `<thead><tr>${header.map((cell, column) => `<th style="text-align:${alignment[column]}">${inlineMarkdown(cell)}</th>`).join('')}</tr></thead>`;
        const bodyHtml = `<tbody>${rows.map((row) => `<tr>${header.map((_, column) => `<td style="text-align:${alignment[column]}">${inlineMarkdown(row[column] ?? '')}</td>`).join('')}</tr>`).join('')}</tbody>`;
        blocks.push(`<div class="table-wrap markdown-table"><table>${headerHtml}${bodyHtml}</table></div>`); continue;
      }
      const heading = lines[index].match(/^\s{0,3}(#{1,6})\s+(.+)$/);
      if (heading) { const level = Math.min(heading[1].length, 4); blocks.push(`<h${level}>${inlineMarkdown(heading[2].replace(/\s+#+\s*$/, ''))}</h${level}>`); index += 1; continue; }
      if (/^\s*[-*+]\s+/.test(lines[index])) {
        const items = []; while (index < lines.length && /^\s*[-*+]\s+/.test(lines[index])) items.push(`<li>${inlineMarkdown(lines[index++].replace(/^\s*[-*+]\s+/, ''))}</li>`);
        blocks.push(`<ul>${items.join('')}</ul>`); continue;
      }
      if (/^\s*\d+\.\s+/.test(lines[index])) {
        const items = []; while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) items.push(`<li>${inlineMarkdown(lines[index++].replace(/^\s*\d+\.\s+/, ''))}</li>`);
        blocks.push(`<ol>${items.join('')}</ol>`); continue;
      }
      const paragraph = []; while (index < lines.length && lines[index].trim() && !/^\s*```/.test(lines[index]) && !/^\s{0,3}#{1,6}\s+/.test(lines[index]) && !isMarkdownTableStart(lines, index)) paragraph.push(inlineMarkdown(lines[index++]));
      blocks.push(`<p>${paragraph.join('<br>')}</p>`);
    }
    return blocks.join('\n');
  }
  $('folder').addEventListener('change', (event) => load([...event.target.files])); $('flow-refresh').addEventListener('click', () => refreshFlows()); $('flow-more').addEventListener('click', () => loadMoreFlows()); $('flow-all').addEventListener('click', () => loadMoreFlows(true)); $('flow-page-previous').addEventListener('click', () => { viewer.flowPage = Math.max(0, viewer.flowPage - 1); renderTable(viewer.flowRows, viewer.flowTitle, viewer.flowMatchedCount); }); $('flow-page-next').addEventListener('click', () => { viewer.flowPage += 1; renderTable(viewer.flowRows, viewer.flowTitle, viewer.flowMatchedCount); }); $('flow-clear-cache').addEventListener('click', clearCache); ['flow-bin', 'flow-origin', 'flow-destination'].forEach((id) => $(id).addEventListener('change', (event) => { if (event.currentTarget.id !== 'flow-bin') { viewer.selectedOd = null; updateOdEndpointMarkers(); updateOdMapSelectionStatus(); } cancelRoutePathQuery(); void refreshFlows(); void updateRoutePathBrowser({clearResults: true}); }));
  $('route-path-service')?.addEventListener('change', () => { cancelRoutePathQuery(); void updateRoutePathBrowser({clearResults: true}); });
  $('route-path-load')?.addEventListener('click', loadRoutePaths);
  $('route-path-threshold')?.addEventListener('change', () => {
    cancelRoutePathQuery();
    $('route-path-summary').textContent = '';
    $('route-path-results').replaceChildren();
    if ($('flow-origin').value && $('flow-destination').value) {
      $('route-path-status').textContent = 'Threshold updated. Request paths to apply the new minimum probability.';
    }
  });
  $('map-od-mode')?.addEventListener('click', () => setMapClickMode('od')); $('map-measurement-mode')?.addEventListener('click', () => setMapClickMode('measurement'));
  $('measurement-od-direction')?.addEventListener('change', () => {
    updateMeasurementOdDirection();
    viewer.measurementContributionPage = 0;
    viewer.measurementOdSortColumn = 'modeled_mean_contribution';
    viewer.measurementOdSortDirection = 'desc';
    void refreshMeasurementOdForCurrentSelection();
  });
  $('measurement-od-load')?.addEventListener('click', refreshMeasurementOd);
  $('measurement-row-select')?.addEventListener('change', () => {
    const row = viewer.measurementRows.find((item) => String(item.measurement_index) === $('measurement-row-select').value);
    if (row) renderSelectedMeasurementRow(row);
  });
  $('measurement-row-previous')?.addEventListener('click', () => changeMeasurementRowPage(-1)); $('measurement-row-next')?.addEventListener('click', () => changeMeasurementRowPage(1));
  $('measurement-contribution-previous')?.addEventListener('click', () => { viewer.measurementContributionPage = Math.max(0, viewer.measurementContributionPage - 1); refreshMeasurementOd(); });
  $('measurement-contribution-next')?.addEventListener('click', () => { viewer.measurementContributionPage += 1; refreshMeasurementOd(); });
  $('measurement-od-limit')?.addEventListener('change', () => {
    viewer.measurementContributionPage = 0;
    void refreshMeasurementOdForCurrentSelection();
  });
  ['measurement-bin', 'measurement-origin', 'measurement-destination'].forEach((id) => $(id)?.addEventListener('change', () => {
    if ($('measurement-od-direction').value === 'od') void refreshMeasurementOdForCurrentSelection();
  }));
})();
