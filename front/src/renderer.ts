/**
 * This file will automatically be loaded by webpack and run in the "renderer" context.
 * To learn more about the differences between the "main" and the "renderer" context in
 * Electron, visit:
 *
 * https://electronjs.org/docs/latest/tutorial/process-model
 *
 * By default, Node.js integration in this file is disabled. When enabling Node.js integration
 * in a renderer process, please be aware of potential security implications. You can read
 * more about security risks here:
 *
 * https://electronjs.org/docs/tutorial/security
 *
 * To enable Node.js integration in this file, open up `main.js` and enable the `nodeIntegration`
 * flag:
 *
 * ```
 *  // Create the browser window.
 *  mainWindow = new BrowserWindow({
 *    width: 800,
 *    height: 600,
 *    webPreferences: {
 *      nodeIntegration: true
 *    }
 *  });
 * ```
 */

import * as d3 from 'd3';
import videojs from "video.js";
import './index.css';
import 'video.js/dist/video-js.css';

// player.ready(function() {
//   // get the ProgressControl element (contains .vjs-progress-holder)
//   var progressHolderEl = player.controlBar.progressControl.el().firstChild;

//   // create overlay and insert as first child of the progress control
//   var overlay = document.createElement('div');
//   overlay.className = 'my-above-overlay';
//   overlay.textContent = 'My overlay content'; // put whatever you like here

//   // insert it so it shares the same horizontal bounds as the seek bar
//   progressHolderEl.insertBefore(overlay, progressHolderEl.firstChild);

//   // If you want the overlay to be interactive (clicks), remove pointer-events:none
//   // and translate clicks into times (example below).
// });
function getVideoMimeType(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase();

  const map: Record<string, string> = {
    mp4: "video/mp4",
    webm: "video/webm",
    mkv: "video/x-matroska",
    mov: "video/quicktime",
    avi: "video/x-msvideo"
  };

  return map[ext ?? ""] ?? "video/mp4";
}

const player = videojs("my-video")

async function handleUpload(e: Event) {
  
  e.preventDefault();  
  const path = await (window as any).electronAPI.openFile();
  if (!path) return;
  
  const normalized = path.replace(/\\/g, '/');
  const url = 'video://' + encodeURI(normalized)
    
  player.src({
    src: url,
    type: getVideoMimeType(url)
  });
  
  player.play();

  //TODO: connect better
  const response = await fetch("http://localhost:8000", {
    method: "POST",
    body: JSON.stringify({
      "path": encodeURI(normalized)
    }),
    headers: {
      "Content-Type" : "application/json"
    }
  })

  // TODO: Handle resizing
  let probs = (await response.json())["probs"];
  const colorScale = d3.scaleSequential(d3.interpolateRdYlGn).domain([1, 0]);

  const progressHolderEl = (player as any).controlBar.progressControl.el().firstChild;
  const canvas = document.createElement("canvas");
  const canvas_container = document.createElement("div");

  canvas.width = progressHolderEl.offsetWidth;
  canvas.height = 24;
  canvas.style.filter = "blur(4px)";

  canvas_container.style.overflow = "hidden";
  canvas_container.style.height = "12px";
  canvas_container.style.position = "absolute";
  canvas_container.style.left = "0";
  canvas_container.style.bottom = "calc(100% + 10px)";
  canvas_container.style.zIndex = "10";
  canvas_container.style.pointerEvents = "none";


  progressHolderEl.insertBefore(canvas_container, progressHolderEl.firstChild);
  canvas_container.insertBefore(canvas, null);

  const ctx = canvas.getContext("2d")!;

  const barWidth = canvas.width / probs.length;

  probs.forEach((v: number, i: number) => {
    ctx.fillStyle = colorScale(v);
    ctx.fillRect(i * barWidth, 0, barWidth, canvas.height);
  });
}

(window as any).handleUpload = handleUpload;
