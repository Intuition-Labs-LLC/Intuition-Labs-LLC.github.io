import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { resolve, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const out = resolve(process.argv[2] || '/tmp/estate-holding-20261007/recombinant');
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const bundle = await build({ absWorkingDir: root, entryPoints: ['src/holding/runtime.js'], bundle: true, write: false,
  format: 'iife', target: 'es2020', minify: true, metafile: true, legalComments: 'inline' });
const script = bundle.outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const fontPath = 'public/fonts/site/bricolage-grotesque-latin.woff2';
const font = await readFile(resolve(root, fontPath));
const template = await readFile(resolve(root, 'src/holding/index.html'), 'utf8');
const style = (await readFile(resolve(root, 'src/holding/style.css'), 'utf8')).replace('__FONT__', font.toString('base64'));
const paths = [...new Set([...Object.keys(bundle.metafile.inputs), fontPath, 'src/holding/index.html', 'src/holding/style.css', 'scripts/holding/build.mjs'])].sort();
const sources = [];
for (const path of paths) sources.push({ path, sha256: sha(await readFile(resolve(root, path))) });
const sourceDigest = sha(JSON.stringify({ version: 1, sources, script: sha(script), style: sha(style) }));
const html = template.replace('__ARTIFACT__', sourceDigest).replace('__STYLE__', style).replace('__SCRIPT__', script);
const scriptHash = createHash('sha256').update(script).digest('base64');
const csp = `default-src 'none'; script-src 'sha256-${scriptHash}'; style-src 'unsafe-inline'; font-src data:; connect-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'`;
const config = { version: 3, routes: [
  { src: '/holding-source/(.*)', dest: '/holding-source/$1', headers: { 'Cache-Control': 'public, max-age=3600', 'X-Content-Type-Options': 'nosniff' } },
  { src: '/.*', dest: '/index.html', status: 503, headers: { 'Cache-Control': 'no-store, max-age=0', 'Retry-After': '86400', 'X-Robots-Tag': 'noindex, nofollow', 'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': csp, 'X-Terminals-State': 'closed-for-reconstruction', 'X-Holding-Artifact': sourceDigest } },
] };
await mkdir(resolve(out, 'static/holding-source'), { recursive: true });
await writeFile(resolve(out, 'static/index.html'), html);
await writeFile(resolve(out, 'static/404.html'), html);
await writeFile(resolve(out, 'config.json'), JSON.stringify(config, null, 2) + '\n');
for (const { path } of sources.filter(({ path }) => path.endsWith('.js') || path.endsWith('.mjs') || path.endsWith('.css') || path.endsWith('.html'))) {
  const destination = resolve(out, 'static/holding-source', path);
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(resolve(root, path), destination);
}
const manifest = { version: 1, source_digest: sourceDigest, sources, artifact: { html_sha256: sha(html), config_sha256: sha(JSON.stringify(config)), script_sha256: sha(script), html_bytes: Buffer.byteLength(html) },
  primitive_owners: ['src/worlds/kernel/law.js', 'src/worlds/kernel/septet.js', 'src/gl/field.js', 'src/ds/limen.js', 'src/ds/optics.js', 'src/ds/spring.js'],
  runtime_inputs: ['explicit UTC-day and rotation snapshot from device clock', 'visible active elapsed time', 'viewport and device pixel ratio', 'transient pointer displacement', 'motion preference and pause'],
  network: 'none; ordinary link to intuitionlabs.tech only', license: 'First-party source. Preserve original source notices; AGPL-3.0-or-later where declared.' };
await writeFile(resolve(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
await writeFile(resolve(out, 'static/holding-source/manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify({ out, source_digest: sourceDigest, html_bytes: manifest.artifact.html_bytes, input_files: sources.length }));
