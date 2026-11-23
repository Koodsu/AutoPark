import { createStore } from "zustand/vanilla";
import { useStore } from "zustand";
import type { FrameMessage, SceneMessage, ServerMessage } from "../net/protocol";
import type { ConnectionStatus } from "../net/client";
import { decodeGrid } from "../render/grid";

export interface GridState {
  version: number;
  data: Uint8Array;
}

export interface AppState {
  status: ConnectionStatus;
  statusDetail: string;
  scene: SceneMessage | null;
  frame: FrameMessage | null;
  grid: GridState | null;
  lastAck: string;
  debug: boolean;
  applyMessage: (msg: ServerMessage) => void;
  setStatus: (status: ConnectionStatus, detail?: string) => void;
  setDebug: (on: boolean) => void;
  reset: () => void;
}

/**
 * Holds the latest server state. The canvas renderer reads it with getState() every
 * animation frame; React panels subscribe to small slices only.
 */
export function createAppStore() {
  return createStore<AppState>()((set, get) => ({
    status: "idle",
    statusDetail: "",
    scene: null,
    frame: null,
    grid: null,
    lastAck: "",
    debug: false,
    applyMessage: (msg) => {
      switch (msg.type) {
        case "scene":
          set({ scene: msg, frame: null, grid: null });
          break;
        case "frame": {
          const scene = get().scene;
          let grid = get().grid;
          if (msg.grid && scene && (!grid || msg.grid.version !== grid.version)) {
            grid = {
              version: msg.grid.version,
              data: decodeGrid(msg.grid.data, scene.grid.nx * scene.grid.ny),
            };
          }
          set({ frame: msg, grid });
          break;
        }
        case "ack":
          set({ lastAck: msg.message });
          break;
        case "error":
          set({ lastAck: `error: ${msg.message}` });
          break;
      }
    },
    setStatus: (status, detail = "") => set({ status, statusDetail: detail }),
    setDebug: (on) => set({ debug: on }),
    reset: () => set({ scene: null, frame: null, grid: null, lastAck: "" }),
  }));
}

export type AppStore = ReturnType<typeof createAppStore>;

export const appStore = createAppStore();

export function useApp<T>(selector: (s: AppState) => T): T {
  return useStore(appStore, selector);
}
